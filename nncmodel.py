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

import copy
from collections import OrderedDict
import torch
from abc import ABC


class NNC_PYT_Model(ABC):

    def __init__(self,
                 model,
                 train_loader,
                 test_loader,
                 mdl_path=None):

        self.model = model

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if mdl_path is not None:
            self.load_model(mdl_path)
            print(f"loaded {mdl_path}")


        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = torch.nn.DataParallel(self.model)


        self.model.to(self.device)

        self.train_loader = train_loader
        self.test_loader = test_loader

    def load_model(self,
                   model_path
                   ):

        model_file = torch.load(model_path, map_location=self.device)  ##loads the state_dict

        try:
            model_parameter_dict = None

            # model state_dict
            if isinstance(model_file, OrderedDict):
                model_parameter_dict = model_file

            # checkpoint including state_dict
            elif isinstance(model_file, dict):
                for key in model_file.keys():
                    if isinstance(model_file[key], OrderedDict):
                        model_parameter_dict = model_file[key]
                        print("Loaded weights from state_dict '{}' from checkpoint elements {}".format(key,
                                                                                                       model_file.keys()))
                        break
                if not model_parameter_dict:
                    assert 0, "Checkpoint does not include a state_dict in {}".format(model_file.keys())

            # whole model (in general not recommended)
            elif isinstance(model_file, torch.nn.Module):
                model_parameter_dict = model_file.state_dict()

            # multi-GPU parallel trained models (torch.nn.DataParallel)
            if not isinstance(self.model, torch.nn.DataParallel):
                if all(i[:7] == 'module.' for i in model_parameter_dict.keys()):
                    print("Removing 'module.' prefixes from state_dict keys resulting from saving torch.nn.DataParallel "
                          "models not in the recommended way, that is torch.save(model.module.state_dict()")
                    new_state_dict = OrderedDict()
                    for n, t in model_parameter_dict.items():
                        name = n[7:]  # remove `module.`
                        new_state_dict[name] = t
                    model_parameter_dict = new_state_dict

        except:
            raise SystemExit("Can't read model: {}".format(model_path))

        if hasattr(self, 'model'):
            self.model.load_state_dict(model_parameter_dict)

    def initialize_STE(self,
                       lr=1e-4,
                       Lambda=0.0,
                       exclude_last_layer=False):

        excluded_layer_names_including = ["downsample"]
        if exclude_last_layer:
            excluded_layer_names_including.append([n for n, v in self.model.named_parameters()
                                                   if len(v.shape) > 1][-1])

        if self.device.type == "cuda":
            fp_weights = [
                param.cuda(self.device).clone().detach().requires_grad_(True)
                for name, param in self.model.named_parameters()
                if len(param.shape) > 1 and not any(n in name for n in excluded_layer_names_including)
            ]
        else:
            fp_weights = [
                param.requires_grad_(True)
                for name, param in self.model.named_parameters()
                if len(param.shape) > 1 and not any(n in name for n in excluded_layer_names_including)
            ]

        self.opt_fp = torch.optim.Adam(fp_weights, lr=lr)

        params = [
            {'params': [param for name, param in self.model.named_parameters()
                        if len(param.shape) < 2 or any(n in name for n in excluded_layer_names_including)],
                        'weight_decay': 5e-6},
            {'params': [param for name, param in self.model.named_parameters()
                        if len(param.shape) > 1 and not any(n in name for n in excluded_layer_names_including)]}
        ]
        self.optimizer = torch.optim.Adam(params, lr=lr)

        self.Lambda = Lambda

    def np_to_torch(self, parameter_dict):
        return {name: torch.tensor(copy.deepcopy(parameter_dict[name])) for name in parameter_dict}
