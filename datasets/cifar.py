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
from torchvision import datasets, transforms

def cifar100(root, split='test', val_data_required=False):

    normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                     std=[0.2673, 0.2564, 0.2762])

    train_trafo = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize
            ])

    val_trafo = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

    if split == 'train':
        train_data = datasets.CIFAR100(root=root, train=True, transform=train_trafo, download=True)
        if val_data_required:
            train_data.data = train_data.data[10000:, :, :, :]
            train_data.targets = train_data.targets[10000:]
        return train_data

    elif split == 'val':
        val_data = datasets.CIFAR100(root=root, train=True, transform=val_trafo, download=True)
        val_data.data = val_data.data[:10000, :, :, :]
        val_data.targets = val_data.targets[:10000]
        return val_data

    elif split == 'test':
        test_data = datasets.CIFAR100(root=root, train=False, transform=val_trafo, download=True)
        return test_data


def cifar10(root, split='test', val_data_required=False):

    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    train_trafo = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize
            ])

    val_trafo = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

    if split == 'train':
        train_data = datasets.CIFAR10(root=root, train=True, transform=train_trafo, download=True)
        if val_data_required:
            train_data.data = train_data.data[10000:, :, :, :]
            train_data.targets = train_data.targets[10000:]
        return train_data

    elif split == 'val':
        val_data = datasets.CIFAR10(root=root, train=True, transform=val_trafo, download=True)
        val_data.data = val_data.data[:10000, :, :, :]
        val_data.targets = val_data.targets[:10000]
        return val_data

    elif split == 'test':
        test_data = datasets.CIFAR10(root=root, train=False, transform=val_trafo, download=True)
        return test_data