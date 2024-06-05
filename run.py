"""
The copyright in this software is being made available under this Software
Copyright License. This software may be subject to other third party and
contributor rights, including patent rights, and no such rights are
granted under this license.
Copyright (c) 1995 - 2024 Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V.
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted for purpose of testing the functionalities of
this software provided that the following conditions are met:
*     Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
*     Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
*     Neither the names of the copyright holders nor the names of its
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
NO EXPRESS OR IMPLIED LICENSES TO ANY PATENT CLAIMS, INCLUDING
WITHOUT LIMITATION THE PATENTS OF THE COPYRIGHT HOLDERS AND
CONTRIBUTORS, ARE GRANTED BY THIS SOFTWARE LICENSE. THE
COPYRIGHT HOLDERS AND CONTRIBUTORS PROVIDE NO WARRANTY OF PATENT
NON-INFRINGEMENT WITH RESPECT TO THIS SOFTWARE.
"""

import argparse
import copy

import torch
import torchvision
import os, random
import numpy as np
import wandb
from collections import OrderedDict
from nncmodel import NNC_PYT_Model
import models, datasets, ecqxtools
from evaluation import evaluate_model
from train import train_ecqx
from nncodec import nnc

parser = argparse.ArgumentParser(description='ECQx_NNCodec')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training (default=64)')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train (default: 10)')
parser.add_argument('--pretrain_epochs', type=int, default=0, help='Number of epochs to pretrain (default: 0)')
parser.add_argument('--max_batches', type=int, default=None, help='Max num of batches to process (default: 0, i.e., all)')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
parser.add_argument('--model', type=str, default='resnet20', metavar=f'any of {models.__all__} or {torchvision.models.list_models(torchvision.models)}')
parser.add_argument('--model_path', type=str, default=None, metavar='./models/pretrained/resnet20.pt')
parser.add_argument('--model_rand_int', action="store_true", help='model randomly initialized, i.e., w/o loading pre-trained weights')
parser.add_argument('--dataset', type=str, default='cifar10', metavar=f"Any of {datasets.__all__}")
parser.add_argument('--data_path', type=str, default='../data')
parser.add_argument('--results', type=str, default='./results')
parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers (default: 4), if 0 debugging mode enabled')
parser.add_argument("--wandb", action="store_true", help='Use Weights & Biases for data logging')
parser.add_argument('--wandb_key', type=str, default='', help='Authentication key for Weights & Biases API account ')
parser.add_argument('--wandb_run_name', type=str, default='ECQx_NNCodec', help='Identifier for current run')
parser.add_argument("--verbose", action="store_true", help='Stdout process information.')
parser.add_argument('--bitwidth', type=int, default=4, help='Number of bits for quantization, i.e., weight precision (default: 4)')
parser.add_argument('--Lambda', type=float, default=0.05, help='Entropy constraint for quantization. Higher values introduce more sparsity. (default: 0.05)')
parser.add_argument("--lrp", action="store_true", help='Use LRP as quantization criterium.')
parser.add_argument("--lrp_output", default="output", help='Output type that is backpropagated for LRP (default: "output" of ["ones", "probability", "output"]).', type=str)
parser.add_argument("--lrp_composite", default="epsilon_plus_flat_bn_pass", help='Composite for LRP calculations (default: "epsilon_plus_flat_bn_pass").', type=str)
parser.add_argument("--lrp_w_rule", default="epsilon", help='Rule for LRP relevances regarding weights (default: "epsilon").', type=str)
parser.add_argument("--canonizer", default="resnet", help='Canonizer for LRP model (default: "resnet"), ["vgg", "resnet", "resnetcifar", "mobilenet"].', type=str)
parser.add_argument("--p", default=0.1, help='Intensity of LRP regularizer (default: 0.1).', type=float)
parser.add_argument("--alpha", default=0.1, help='Amount of previous relevance used for batch relevance (default: 0.1).', type=float)

def main():
    args = parser.parse_args()
    nnc_compression = True # compresses the integer representation of the ECQ(x) obtained weights using NNCodec and saves it to a *.nnc bitstream
    stopping_patience, min_delta = 4, 0.1  # early stopping config

    ### model determinism / reproducibility
    torch.manual_seed(808)
    random.seed(909)
    np.random.seed(303)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(909)
    elif int(torch.version.cuda.split(".")[0]) > 10 or \
            (int(torch.version.cuda.split(".")[0]) == 10 and int(torch.version.cuda.split(".")[1]) >= 2):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if not os.path.exists(args.results):
        os.makedirs(args.results)

    if args.wandb:
        if isinstance(args.wandb_key, str) and len(args.wandb_key) == 40:
            os.environ["WANDB_API_KEY"] = args.wandb_key
        else:
            assert 0, "incompatible W&B authentication key"

    if args.model in models.__all__:
        model = models.init_model(args.model, num_classes=100)
    elif args.model in torchvision.models.list_models(torchvision.models):
        model = torchvision.models.get_model(args.model, weights="DEFAULT" if not args.model_rand_int else None)
    elif args.model in torchvision.models.segmentation.deeplabv3.__all__:
        model = torchvision.models.get_model(args.model, weights="DEFAULT" if not args.model_rand_int else None)
    else:
        assert 0, f"Model not specified and not available in torchvision model zoo" \
                  f"{torchvision.models.list_models(torchvision.models)})"

    run_name = (f"{args.model}_{args.dataset}_ECQ{'x' if args.lrp else ''}_{args.bitwidth}bit_Lambda{args.Lambda}"
                f"{f'_composite_{args.lrp_composite}' if args.lrp else ''}"
                f"{f'_canonizer_{args.canonizer}' if args.lrp else ''}"
                f"{f'_p_{args.p}' if args.lrp else ''}"
                f"{f'_alpha_{args.alpha}' if args.lrp else ''}")

    savedir = os.path.join(args.results, run_name)

    if args.wandb:
        wandb.init(
            config=args,
            project=f"{args.dataset}_{args.wandb_run_name}",
            name=run_name,
            save_code=True,
            dir=f"{args.results}"
        )

    train_loader, test_loader = datasets.init_dataloaders(args.dataset, data_dir=args.data_path, batch_size=args.batch_size,
                                                          num_workers=args.workers, validation_data_required=False)

    if args.model_path is None:
        model_path = f'./models/pretrained/{args.model}_{args.dataset}.pt'
    else:
        model_path = args.model_path

    working_mdl = NNC_PYT_Model(model=model, train_loader=train_loader, test_loader=test_loader, mdl_path=model_path)

    ini_perf = evaluate_model(model, device=working_mdl.device, testloader=test_loader, verbose=args.verbose, max_batches=args.max_batches)

    ##################################### Pre-Training ##########################################
    pretrain_best = ini_perf[0] # change to 1 for mIoU or 2 for loss
    if args.pretrain_epochs > 0:
        for epoch in range(args.pretrain_epochs):
            print('Pre-train Epoch: %d' % epoch)
            train_ecqx(model=working_mdl.model,
                       trainloader=train_loader,
                       optimizer=torch.optim.Adam(model.parameters(), lr=args.lr),
                       device=working_mdl.device,
                       max_batches=args.max_batches,
                       )
            pretrain_perf = evaluate_model(model=working_mdl.model,
                                       testloader=test_loader,
                                       device=working_mdl.device,
                                       max_batches=args.max_batches,
                                       verbose=args.verbose
                                       )
            if epoch > 0 and pretrain_perf[0] < pretrain_best + min_delta: # early stopping
                patience_counter += 1
            else:
                patience_counter = 0
            if pretrain_perf[0] > pretrain_best:
                pretrain_best = pretrain_perf[0]
                print(f'saving pretrained model epoch {epoch} to {savedir+f"_pretrain.pt"}')
                torch.save(working_mdl.model.state_dict(), savedir+f"_pretrain.pt")
            if patience_counter == stopping_patience:
                print("early stopping pre-training")
                break
        if os.path.exists(savedir+f"_pretrain.pt"):
            working_mdl.load_model(savedir+f"_pretrain.pt")
        else:
            working_mdl.load_model(model_path)

    mdl_params = {k: np.float32(v.cpu().detach().numpy()) for k, v in working_mdl.model.state_dict().items() if v.shape != torch.Size([])}
    mdl_size = sum([v.numel() for k, v in working_mdl.model.state_dict().items() if v.shape != torch.Size([])]) * 4
    print(f"Uncompressed model size: {mdl_size/1e6:.4f} MB")

    ######################################### Initialization for ECQ  ###############################################
    working_mdl.initialize_STE(lr=args.lr, Lambda=args.Lambda)

    working_mdl.centroids, step_sizes = ecqxtools.initialize_centroids(mdl_params,
                                                                       bitwidth=args.bitwidth,
                                                                       device=working_mdl.device)
    mdl_params_ini_quant = ecqxtools.ecq(mdl_params,
                                         centroids=working_mdl.centroids,
                                         Lambda=args.Lambda,
                                         keepdims=True,
                                         device=working_mdl.device,
                                         p=args.p
                                         )

    quant_mdl_dict = working_mdl.np_to_torch(mdl_params_ini_quant)
    working_mdl.model.load_state_dict(quant_mdl_dict)
    #################################################################################################################

    c_stats = ecqxtools.get_centroid_stats(mdl_params_ini_quant)
    print(c_stats)
    sparsity_log = ecqxtools.get_sparsity(mdl_params_ini_quant)
    print('Overall network sparsity due to quantization: ', sparsity_log)

    if args.wandb:
        # wandb.watch(working_mdl.model, log="all", log_graph=True)
        wandb.log({"initial_sparsity": sparsity_log, "initial_miou": pretrain_best, "uncompressed_mdl_size": mdl_size})

    ##################################### Start (quantization-aware) Training ##########################################
    best_perf = 0
    for epoch in range(args.epochs):
        print('Epoch: %d' % epoch)

        # train
        train_ecqx(model=working_mdl.model,
                   trainloader=train_loader,
                   optimizer=working_mdl.optimizer,
                   opt_fp=working_mdl.opt_fp,
                   device=working_mdl.device,
                   centroids=working_mdl.centroids,
                   Lambda=args.Lambda,
                   lrp=args.lrp,
                   lrp_w_rule=args.lrp_w_rule,
                   lrp_composite=args.lrp_composite,
                   canonizer=args.canonizer,
                   max_batches=args.max_batches,
                   p=args.p,
                   alpha=args.alpha
                   )

        # test
        test_perf = evaluate_model(model=working_mdl.model,
                                   testloader=test_loader,
                                   device=working_mdl.device,
                                   max_batches=args.max_batches,
                                   verbose=args.verbose
                                   )

        # log stats
        mdl_params = {k: np.float32(v.cpu().detach().numpy()) for k, v in working_mdl.model.state_dict().items() if
                      v.shape != torch.Size([])}
        c_stats = ecqxtools.get_centroid_stats(mdl_params)
        sparsity_log = ecqxtools.get_sparsity(mdl_params)
        print('Overall network sparsity due to initial quantization: ', sparsity_log)
        if args.wandb:
            wandb.log({"centroid_stats": c_stats,
                       "sparsity": sparsity_log,
                       "test_top1": test_perf[0], "test_miou": test_perf[1], "test_loss": test_perf[2]})

        if nnc_compression:
            integer_params, float_params = {}, {}
            for name, param in mdl_params.items():
                if name in step_sizes:
                    integer_params[name] = np.int32(mdl_params[name] / step_sizes[name])
                else:
                    float_params[name] = mdl_params[name]

            int_bs, int_ogs = nnc.compress(integer_params, bitstream_path=f"{savedir}_integer_bitstream.nnc",
                                           use_dq=False, return_bitstream=True)
            rec_int_params = nnc.decompress(int_bs)
            rec_params = {k: rec_int_params[k] * step_sizes[k] for k in rec_int_params}

            float_bs, float_ogs = nnc.compress(float_params, bitstream_path=f"{savedir}_float_bitstream.nnc",
                                               qp=-40, nonweight_qp=-75, use_dq=True, return_bitstream=True)
            rec_float_params = nnc.decompress(float_bs)
            rec_params.update(rec_float_params)
            rec_state_dict = OrderedDict({k: torch.tensor(v) for k, v in rec_params.items()})

            test_mdl = copy.deepcopy(working_mdl.model)
            test_mdl.load_state_dict(rec_state_dict)
            rec_test_perf = evaluate_model(model=test_mdl, testloader=test_loader, device=working_mdl.device,
                                           max_batches=args.max_batches, verbose=args.verbose)
            overall_bs_size = len(int_bs) + len(float_bs) + (len(step_sizes)*32)
            og_size = int_ogs + float_ogs
            print(f"Overall bs_size: {overall_bs_size} bit ({overall_bs_size/1e6:.4f} MB)")
            print(f"Original size: {og_size} bit (CR = {(overall_bs_size / og_size)*100:.2f}%)")

            if args.wandb:
                wandb.log({"int_bs_size": len(int_bs), "float_bs_size": len(float_bs), "total_bs_size": overall_bs_size,
                        "original_size": og_size, "cr": overall_bs_size / og_size,
                        "rec_test_top1": rec_test_perf[0], "rec_test_miou": test_perf[1], "rec_test_loss": rec_test_perf[2]})


        if epoch > 0 and test_perf[0] < best_perf + min_delta:  # early stopping
            patience_counter += 1
        else:
            patience_counter = 0
        if test_perf[0] > best_perf:
            best_perf = test_perf[0]
            torch.save(working_mdl.model.state_dict(), savedir+".pt")
            print(f"Saved model to {savedir}")
        if patience_counter == stopping_patience:
            print("early stopping ECQ(x)")
            break

if __name__ == '__main__':
    main()