"""
The copyright in this software is being made available under this Software
Copyright License. This software may be subject to other third party and
contributor rights, including patent rights, and no such rights are
granted under this license.
Copyright (c) 1995 - 2024 Fraunhofer-Gesellschaft zur Förderung der
angewandten Forschung e.V. (Fraunhofer)
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
from torch.utils.data import dataloader
import numpy as np
import random
from .cifar import cifar10, cifar100
from .voc2012 import voc

__all__ = ['cifar10', 'cifar100', 'voc']

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
g = torch.Generator()
g.manual_seed(0)

def init_dataloaders(dataset, data_dir, batch_size, num_workers, validation_data_required=False):

    if dataset == 'cifar10':
        train_set = cifar10(data_dir, split="train", val_data_required=validation_data_required)
        test_set = cifar10(data_dir, split="test")
        val_set = cifar10(data_dir, split="val")
    elif dataset == 'cifar100':
        train_set = cifar100(data_dir, split="train", val_data_required=validation_data_required)
        test_set = cifar100(data_dir, split="test")
        val_set = cifar100(data_dir, split="val")
    elif dataset == 'voc':
        train_set = voc(data_dir, split="train")
        test_set = voc(data_dir, split="test")
        val_set = voc(data_dir, split="val")
    else:
        assert 0, "unknown dataset"

    train_loader = torch.utils.data.DataLoader(
                            train_set,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True,
                            collate_fn=getattr(train_set, "collate_fn", dataloader.default_collate),
                            sampler=getattr(train_set, "sampler", None),
                            worker_init_fn=seed_worker,
                            generator=g
    )

    test_loader = torch.utils.data.DataLoader(
                            test_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            collate_fn=getattr(test_set, "collate_fn", dataloader.default_collate),
                            sampler=getattr(test_set, "sampler", None),
                            worker_init_fn=seed_worker,
                            generator=g
    )

    if validation_data_required:
        val_loader = torch.utils.data.DataLoader(
                                    val_set,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=True,
                                    collate_fn=getattr(val_set, "collate_fn", dataloader.default_collate),
                                    sampler=getattr(val_set, "sampler", None),
                                    worker_init_fn=seed_worker,
                                    generator=g
        )

        return train_loader, val_loader, test_loader

    return train_loader, test_loader