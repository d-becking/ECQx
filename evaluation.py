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

import torch
import numpy as np

def evaluate_model(model, testloader, max_batches=None, device=0, verbose=False, rsem19=False):

    model = model.to(device)
    model.eval()
    test_loss = []
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0
    top5_acc = 0

    DeepLab_condition = model.__class__.__name__ == "DeepLabV3" if not (isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel)) else model.module.__class__.__name__  == "DeepLabV3"

    if DeepLab_condition:
        from torchmetrics import JaccardIndex
        num_of_classes = model.classifier[len(model.classifier) - 1].weight.shape[0] if not (isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel)) else model.module.classifier[len(model.module.classifier) - 1].weight.shape[0]
        jaccard = JaccardIndex(task="multiclass", num_classes=num_of_classes, average="macro", ignore_index=-100).to(device)

    total_iterations = max_batches or len(testloader)
    iterator = enumerate(testloader)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in iterator:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            if DeepLab_condition:
                outputs = outputs['out']
                targets = targets.long()

            loss = torch.nn.CrossEntropyLoss()(outputs, targets)

            test_loss.append(loss.item())
            _, predicted = outputs.max(1)
            total += targets.size(0) if not DeepLab_condition else targets.numel()
            correct += predicted.eq(targets).sum().item()
            all_predictions.append(np.array(predicted.cpu()))
            all_labels.append(np.array(targets.cpu()))

            acc = 100. * correct / total

            if DeepLab_condition:
                jaccard.update(predicted, targets)
                if verbose and batch_idx % 50 == 0:
                    print('Running Test/Val mIOU (batch {}/{}): {}'.format(batch_idx, total_iterations, jaccard.compute() * 100))

            if batch_idx == max_batches:
                break

            if verbose and batch_idx % 50 == 0:
                print('Running Test/Val Accuracy (batch {}/{}): {}'.format(batch_idx, total_iterations, acc))

        acc = 100. * correct / total

        if verbose:
            print('\t- Test loss: %0.4f' % np.mean(test_loss))
            print('\t- Test accuracy: %0.4f%%' % acc)

        if DeepLab_condition:
            final_mIoU = jaccard.compute() * 100
            print('\t- Test mIoU: %0.4f%%' % final_mIoU)
            return acc, final_mIoU.item(), np.mean(test_loss)
        else:
            return acc, float(top5_acc), np.mean(test_loss)
