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

import copy
from collections import OrderedDict
import torch
from ecqxtools import get_grads, ecq, LRP

def freeze_batch_norm_layers(model):
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.eval()

def train_ecqx(model, trainloader, device, optimizer, opt_fp=None, centroids=None, Lambda=0.0, lrp=None, p=0.1,
               lrp_output="output", lrp_w_rule="epsilon", lrp_composite="epsilon_flat", canonizer='vgg', alpha=0,
               max_batches=None, exclude_last_layer=False, freeze_batch_norm=False):

    model.to(device)
    model.train()
    if freeze_batch_norm:
        freeze_batch_norm_layers(model)
    train_loss = []
    correct = 0
    total = 0
    lrp_relevances = OrderedDict()
    total_iterations = max_batches or len(trainloader)

    iterator = enumerate(trainloader)

    DeepLab_condition = model.__class__.__name__ == "DeepLabV3" if not (isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel)) \
                            else model.module.__class__.__name__ == "DeepLabV3"

    if DeepLab_condition:
        from torchmetrics import JaccardIndex
        num_of_classes = model.classifier[len(model.classifier) - 1].weight.shape[0] if not (isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel)) else model.module.classifier[len(model.module.classifier) - 1].weight.shape[0]
        jaccard = JaccardIndex(task="multiclass", num_classes=num_of_classes, average="macro", ignore_index=-100).to(device)

    for batch_idx, (inputs, targets) in iterator:

        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        if DeepLab_condition:
            targets = targets.long()
            outputs = outputs['out']

        loss = torch.nn.CrossEntropyLoss()(outputs, targets)

        loss.backward()
        train_loss.append(loss.item())

        _, predicted = outputs.max(1)

        total += targets.size(0) if not DeepLab_condition else targets.numel()
        correct += predicted.eq(targets).sum().item()

        if DeepLab_condition:
            jaccard.update(predicted, targets)

        if batch_idx % 100 == 0:
            print('Running Train Accuracy (batch {}/{}): {}'.format(batch_idx, total_iterations, correct * 100.0 / total))
            if DeepLab_condition:
                print('Running Train mIOU (batch {}/{}): {}'.format(batch_idx, total_iterations, jaccard.compute() * 100))

        del predicted, loss, _

        if opt_fp:
            opt_fp.zero_grad()
            quant_weights = optimizer.param_groups[1]['params']

            for i in range(len(quant_weights)):
                opt_fp.param_groups[0]['params'][i].grad = get_grads(quant_weights[i])

            opt_fp.step()
            optimizer.step()

            if lrp:
                Model = copy.deepcopy(model)
                lrp_ = LRP(Model, data=inputs, target=targets, output_type=lrp_output,
                           composite=lrp_composite, weight_rule=lrp_w_rule, canonizer=canonizer)

                del Model, inputs, targets, outputs
                torch.cuda.empty_cache()

                relevances = lrp_.get_relevances_weights()
                # lrp_.plot_explanation('./', {'lrp_heatmap_batch': batch_idx})
                del lrp_

                for param in relevances:
                    excluded_layer_names_including = ["downsample", "aux_classifier"]
                    if exclude_last_layer:
                        excluded_layer_names_including.append([n for n in relevances if len(relevances[n].shape) > 1][-1])
                    if len(relevances[param].shape) > 1 and not any(n in param for n in excluded_layer_names_including):  # remove 1-dim params (BN, bias, ...)
                        # mix relevance for batch with relevance from previous batches
                        prev_relevance = lrp_relevances[param] if param in lrp_relevances.keys() else relevances[param]
                        new_relevance = relevances[param]
                        lrp_relevances[param] = (1-alpha)*new_relevance + alpha*prev_relevance
                del relevances
            else:
                lrp_relevances = None

            quant_weights_update = ecq(opt_fp.param_groups[0]['params'],
                                       centroids=centroids,
                                       Lambda=Lambda,
                                       lrp=lrp_relevances,
                                       device=device,
                                       p=p)

            for i in range(len(quant_weights)):
                optimizer.param_groups[1]['params'][i].data = quant_weights_update[i].to(device)

            del quant_weights_update
            torch.cuda.empty_cache()

        else:
            optimizer.step()

        if batch_idx == max_batches:
            break

    print(f"Final acc. of epoch: {correct * 100.0 / total}")

    del lrp_relevances
