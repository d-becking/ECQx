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
import numpy as np
from torch import Tensor
from torch.nn import Module
from typing import Dict, Tuple
from zennit.composites import COMPOSITES
from zennit.torchvision import VGGCanonizer, ResNetCanonizer, MobileNetCanonizer
from zennit.rules import RULES

# percentile to calculate quantization stepsize
PERCENTILE = 99.9

def get_grads(quant_opt):
    """
    Returns resulting gradients for the full precision copy of the model.

    Parameters:
    -----------
        quant_grad:
            Gradients of the (ternary) quantized layer
    Returns:
    --------
        Gradient for the full precision weights
    """

    if quant_opt.grad == None:
        return None

    quant_grad = quant_opt.grad.data
    quant_weights = quant_opt.data

    centroids = torch.unique(quant_weights)
    if centroids.numel() > 2**8:
        return quant_grad
        # assert 0, f"Too many centroids detected: {centroids.numel()}. Probably a tensor is not quantized correctly."

    fp_zero_grad = torch.ones_like(quant_weights)
    fp_grad = torch.zeros_like(quant_weights)
    for c in range(centroids.numel()):
        if centroids[int(c)] != 0:
            fp_grad += centroids[c] * (quant_weights == centroids[c]) * quant_grad
            fp_zero_grad -= 1 * (quant_weights == centroids[c])
    fp_grad += fp_zero_grad * quant_grad

    return fp_grad

def get_pmf(prelim_assignment, zero_idx, num_centroids, biased=True, device=0):
    """
    Calculates the probabilitiy mass function (pmf) which here is simplified the number of weights assigned to a
    specific centroid devided by th number of all weights. The preliminary assignment considers only the minimal
    distance from  centroids to weights as cost.
    With "spars_bound" we ensure that at least 50% of all weights would be assigned to the zero-centroid w_0 such that
    the entropy score (information content) for w_0 is always the lowest.

    Parameters:
    -----------
        prelim_assignment:
            Minimal arguments for all 3 centroid distances to all layer weights

    Returns:
    --------
        pmf_prelim:
            Percentage frequencies of -only distance dependent- centroid assignments (w_n, w_0, w_p)
            with pmf[w_n] + pmf[w_0] + pmf[w_p] = 1 and  pmf[w_0] always > 0.5
    """

    C_val, C_cts = torch.unique(prelim_assignment, return_counts=True)

    zero_idx = torch.where(C_val==zero_idx)[0].item()
    # Ensuring that the probability of w_0 is at least 50%
    spars_bound = 0.5 if biased else 0.0

    pmf_prelim = torch.div(C_cts.type(torch.float32), torch.numel(prelim_assignment))

    if pmf_prelim[zero_idx] < spars_bound:
        nonzero_sum = 0
        for i in range(pmf_prelim.size(0)):
            if i != zero_idx:
                nonzero_sum += pmf_prelim[i]
        for i in range(pmf_prelim.size(0)):
            pmf_prelim[i] -= (pmf_prelim[i] / nonzero_sum) * (spars_bound - pmf_prelim[zero_idx])
        pmf_prelim[zero_idx] = spars_bound

    if len(pmf_prelim) < num_centroids:
        pmf_prelim_extended_idx = torch.arange(num_centroids).to(device)
        pmf_prelim_extended = torch.zeros_like(pmf_prelim_extended_idx).type(torch.float32)
        pmf_prelim_idx = 0
        for idx in pmf_prelim_extended_idx:
            if idx in C_val:
                pmf_prelim_extended[idx] = pmf_prelim[pmf_prelim_idx]
                pmf_prelim_idx += 1
        pmf_prelim = pmf_prelim_extended

    return pmf_prelim

def apply_entropy_constraint(weights, centroids, largest_layer_size, lambda_max_divider, lrp=None, device=0, p=0.1):
    """
    Applies the entropy constraint to the quantizer's cost function. The function
    finds the maximum Lambda (for which almost all weights are assigned to w_0) scales
    it with lambda_decay and lambda_max_divider. The returned cost can be described as:

        d(w_i, w_c) + (lambda_max_divider * lambda_decay * Lambda * information_content_c)

            c ... element of the centroids [w_n, w_0, w_p],
            i ... element of i layer weights and
            d() ... the squared distance of layer weights to the centroids.

    Parameters:
    -----------
        weights:
            Full precision weights of the given layer.
        w_p, w_n:
            Negative and positive centroid values of a layer.
        lambda_decay:
            Decreases Lambda according to the number of layer weights. Smaller layers won't be as sparse as huge layers.
        lambda_max_divider:
            Lambda_max is defined as the Lambda for which almost all layer
            weights would be assigned to w_0. The greater lambda_max_divider the sparser will be the resulting ternary
            network. We want to find the maximal lambda_max_divider, i.e. the sparsest network, which still maintains
            the initial accuracy (to a specified level).

    Returns:
    --------
        cost:
            Final cost for all centroids with lambda applied.
    """
    zero_idx = torch.where(centroids == 0)[0].item()
    # Calculating distances from layer weights to centroids
    dist = get_distances(weights, centroids)
    # Preliminary assignment depending only on the distance as cost
    prelim_assignment = torch.argmin(dist, dim=0)
    # Calculating probability mass function approximation, given the preliminary assignment
    pmf = get_pmf(prelim_assignment, zero_idx, num_centroids=len(dist), device=device)

    # Centroid's information content
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        I = -torch.log2(pmf).to(device)
    else:
        I = -torch.log2(pmf)

    # Extrude information content I such that it has the same num of dimensions as "weights"
    I_extruded = torch.unsqueeze(I, 1)
    for i in range(1, dist.shape.__len__() - 1):
        I_extruded = torch.unsqueeze(I_extruded, 1) # Iteratively adding a new axis for the layers shape's length


    # Calculate the maximal Lambdas for which all weights would be assigned to either [w_0, w_n] or [w_0, w_p]
    Lambda_n_max = ((centroids[prelim_assignment.min()] - weights.min()) ** 2 - (0 - weights.min())) ** 2 / \
                   (I[prelim_assignment.min()] - I[zero_idx])

    Lambda_p_max = ((centroids[prelim_assignment.max()] - weights.max()) ** 2 - (0 - weights.max())) ** 2 / \
                   (I[prelim_assignment.max()] - I[zero_idx])

    if Lambda_p_max < 0 or Lambda_n_max < 0:
        print('lambda max < 0')

    # From the maximal Lambdas choose the greater value
    if Lambda_p_max > Lambda_n_max:
        Lambda = Lambda_p_max
    else:
        Lambda = Lambda_n_max

    sustain = 0.05
    lambda_decay = ((weights.numel() / largest_layer_size) + sustain) / (1 + sustain)

    # Multiplying Lambda with its scaling factors
    Lambda *= lambda_max_divider
    Lambda *= lambda_decay

    # changing cost with LRP
    entropy = Lambda * I_extruded
    cost = torch.add(dist, entropy)

    assignment_entropy = torch.argmin(cost, dim=0)
    sparsity_entropy = assignment_entropy[assignment_entropy == zero_idx].numel() / assignment_entropy.numel()

    sparsity_lrp = 0
    iteration = 0

    if isinstance(lrp, torch.Tensor):
        beta = torch.log(torch.tensor(0.5)) / torch.log(lrp.mean())

        # find best beta value
        cost_ = cost.clone()
        while sparsity_lrp > sparsity_entropy or sparsity_lrp < sparsity_entropy - p:
            # leave loop if sparsity_entropy - p < sparsity_lrp < sparsity_entropy (or iteration >= 50)

            lrp_ = (lrp + (torch.full_like(lrp, 1e-12)))  ** beta

            cost_ = cost.clone()
            cost_[zero_idx] *= 2*lrp_

            assignment_lrp = torch.argmin(cost_, dim=0)
            sparsity_lrp = assignment_lrp[assignment_lrp == zero_idx].numel() / assignment_lrp.numel()

            if iteration >= 40:
                beta *= (1 + .75*np.sign(sparsity_entropy-sparsity_lrp))
            elif iteration >= 20:
                beta *= (1 + .5*np.sign(sparsity_entropy-sparsity_lrp))
            else:
                beta *= (1 + .1*np.sign(sparsity_entropy-sparsity_lrp))  # either make beta larger or smaller
            beta = min(beta, 1)
            if iteration >= 50:
                break
            iteration += 1

        cost = cost_

    return cost, prelim_assignment
#
def get_distances(param, centroids):
    """
    Calculates the squared distances from all param elements to all centroids
    Parameters:
    -----------
        param:
            Weights of the given layer.
        centroids:
            Centroid values of layer.
    Returns:
    --------
        squared distances to all centroids in a new tensor axis
    """
    G = torch.unsqueeze(centroids, 1)
    for i in range(1, param.shape.__len__()): # Iteratively adding a new axis for the layers shape's length
        G = torch.unsqueeze(G, 1)
    return (G.sub(param)) ** 2
#
def ecq(orig_model_params, centroids, Lambda=0.0, lrp=None, ap_info=None, keepdims=False, device=0, exclude_last_layer=False, p=0.1):

    excluded_layer_names_including = ["downsample", "aux_classifier"]
    if exclude_last_layer: ## only once required in initial quantization (not while quantization aware training)
        excluded_layer_names_including.append([n for n in orig_model_params if len(orig_model_params[n].shape) > 1][-1])

    if lrp and isinstance(lrp, OrderedDict):
        param_names = []
        for name in lrp:
            param_names.append(name)

    if isinstance(orig_model_params, dict) or isinstance(orig_model_params, OrderedDict):
        param_identifier = []
        model_params = []
        for name in orig_model_params:
            if len(orig_model_params[name].shape) > 1 and not any(n in name for n in excluded_layer_names_including):
                param_identifier.append(name)
                if isinstance(orig_model_params[name], np.ndarray):
                    model_params.append(torch.tensor(copy.deepcopy(orig_model_params[name])).to(device))
                else:
                    model_params.append(copy.deepcopy(orig_model_params[name]))
        rec_dict = True

    elif isinstance(orig_model_params, list):
        model_params = copy.deepcopy(orig_model_params)
        rec_dict = False

    param_sizes = []
    [param_sizes.append(prmtr.numel()) for prmtr in model_params]

    for t_idx, (tensor, name) in enumerate(zip(model_params, centroids)):
        T = tensor.to(device)
        if len(T.shape) < 2 or not T.any() or any(n in name for n in excluded_layer_names_including):
            model_params[t_idx] = tensor
        else:
            if ap_info != None and "skc_codebook" in ap_info.approx_info:
                ap_info.approx_info["skc_codebook"][tensor] = 1

            if lrp:
                relevance = (lrp[param_names[t_idx]].abs() / (lrp[param_names[t_idx]].abs().max() + 1e-12)).to(device)
            else:
                relevance = None

            if Lambda != 0:
                cost, distance_assignment = apply_entropy_constraint(T,
                                                                     centroids[name].to(device),
                                                                     largest_layer_size=max(param_sizes),
                                                                     lambda_max_divider=Lambda,
                                                                     lrp=relevance,
                                                                     device=device,
                                                                     p=p)
            else:
                cost = get_distances(T, centroids[name])

            # Assigning weights to centroids with minimal distance
            assignment = torch.argmin(cost, dim=0)
            del cost

            # Weight values assigned to each cluster
            assigned_values = [(assignment == c) * centroids[name][c].to(device) for c in range(len(centroids[name]))]
            del assignment

            quantized_tensor = torch.zeros_like(assigned_values[0])
            for i in range(len(assigned_values)):
                quantized_tensor += assigned_values[i]

            model_params[t_idx] = quantized_tensor
            del quantized_tensor, assigned_values

    if rec_dict:
        m_dict = OrderedDict()
        assert len(param_identifier) == len(model_params), "num of quantized params != num of given param names"
        for name, param in zip(param_identifier, model_params):
            m_dict[name] = np.float32(param.cpu().detach().numpy())
        model_params = m_dict

    if keepdims:
        model_params_keepdims = OrderedDict()
        for module_name in orig_model_params:
            if module_name in model_params:
                model_params_keepdims[module_name] = model_params[module_name]
            else:
                model_params_keepdims[module_name] = orig_model_params[module_name]
        model_params = model_params_keepdims

    return model_params
#
def initialize_centroids(parameters, bitwidth, uint_support=True, device=0, exclude_last_layer=False):
    model_params = copy.deepcopy(parameters)
    centroids = OrderedDict()
    step_sizes = OrderedDict()

    excluded_layer_names_including = ["downsample", "aux_classifier"]
    if exclude_last_layer:
        excluded_layer_names_including.append([n for n in parameters if len(parameters[n].shape) > 1][-1])

    for tensor in model_params:
        if len(model_params[tensor].shape) > 1 and not any(n in tensor for n in excluded_layer_names_including):

            if torch.cuda.is_available() or torch.backends.mps.is_available():
                T = torch.tensor(model_params[tensor]).to(device)
            else:
                T = torch.tensor(model_params[tensor])

            if uint_support and len(T[T < 0]) == 0:
                denom = 2 ** bitwidth - 1
            else:
                denom = 2 ** (bitwidth - 1) - 1

            step_size = np.percentile(abs(T.cpu().detach().numpy()), PERCENTILE) / denom
            if step_size == 0:
                step_size = np.max(abs(T.cpu().detach().numpy())) / denom

            if uint_support and len(T[T < 0]) == 0:
                centroids[tensor] = (torch.arange(0, 2 ** bitwidth) * step_size).type(torch.float32).to(device)

            else:
                # layer statistics
                absmax = torch.max(T).abs()
                absmin = torch.min(T).abs()
                num_pos = T[T > 0].numel()
                num_neg = T[T < 0].numel()
                mean_pos = torch.mean(T[T > 0]).abs()
                mean_neg = torch.mean(T[T < 0]).abs()
                sum_pos = (T[T > 0]).abs().sum()
                sum_neg = (T[T < 0]).abs().sum()
                std_pos = torch.std(T[T > 0])
                std_neg = torch.std(T[T < 0])

                pos_ctr = 0
                neg_ctr = 0
                if absmax > absmin:
                    pos_ctr += 1
                else:
                    neg_ctr += 1
                if mean_pos > mean_neg:
                    pos_ctr += 1
                else:
                    neg_ctr += 1
                if num_pos > num_neg:
                    pos_ctr += 1
                else:
                    neg_ctr += 1
                if sum_pos > sum_neg:
                    pos_ctr += 1
                else:
                    neg_ctr += 1
                if std_pos > std_neg:
                    pos_ctr += 1
                else:
                    neg_ctr += 1

                if neg_ctr > pos_ctr:
                    centroids[tensor] = (torch.arange(-2 ** (bitwidth - 1), 2 ** (bitwidth - 1))
                                         * step_size).type(torch.float32).to(device)
                elif pos_ctr > neg_ctr:
                    centroids[tensor] = (torch.arange(-2 ** (bitwidth - 1) + 1, 2 ** (bitwidth - 1) + 1)
                                         * step_size).type(torch.float32).to(device)

            step_sizes[tensor] = step_size

    return centroids, step_sizes
#
def get_layer_entropy(tensor, bitwidth, uint_support=True, device=0):

    if torch.cuda.is_available() or torch.backends.mps.is_available():
        T = torch.tensor(tensor).to(device)
    else:
        T = torch.tensor(tensor)

    if uint_support and len(T[T < 0]) == 0:
        denom = 2 ** bitwidth - 1
    else:
        denom = 2 ** (bitwidth - 1) - 1

    step_size = np.percentile(abs(T.cpu().detach().numpy()), PERCENTILE) / denom
    if step_size == 0:
        step_size = np.max(abs(T.cpu().detach().numpy())) / denom

    if uint_support and len(T[T < 0]) == 0:
        centroids = (torch.arange(0, 2 ** bitwidth) * step_size).type(torch.float32).to(device)

    else:
        # layer statistics
        absmax = torch.max(T).abs()
        absmin = torch.min(T).abs()
        num_pos = T[T > 0].numel()
        num_neg = T[T < 0].numel()
        mean_pos = torch.mean(T[T > 0]).abs()
        mean_neg = torch.mean(T[T < 0]).abs()
        sum_pos = (T[T > 0]).abs().sum()
        sum_neg = (T[T < 0]).abs().sum()
        std_pos = torch.std(T[T > 0])
        std_neg = torch.std(T[T < 0])

        pos_ctr = 0
        neg_ctr = 0
        if absmax > absmin:
            pos_ctr += 1
        else:
            neg_ctr += 1
        if mean_pos > mean_neg:
            pos_ctr += 1
        else:
            neg_ctr += 1
        if num_pos > num_neg:
            pos_ctr += 1
        else:
            neg_ctr += 1
        if sum_pos > sum_neg:
            pos_ctr += 1
        else:
            neg_ctr += 1
        if std_pos > std_neg:
            pos_ctr += 1
        else:
            neg_ctr += 1

        if neg_ctr > pos_ctr:
            centroids = (torch.arange(-2 ** (bitwidth - 1), 2 ** (bitwidth - 1)) * step_size).type(
                torch.float32).to(device)
        elif pos_ctr > neg_ctr:
            centroids = (torch.arange(-2 ** (bitwidth - 1) + 1, 2 ** (bitwidth - 1) + 1) * step_size).type(
                torch.float32).to(device)

    zero_idx = torch.where(centroids == 0)[0].item()
    # Calculating distances from layer weights to centroids
    dist = get_distances(T, centroids)
    # Preliminary assignment depending only on the distance as cost
    prelim_assignment = torch.argmin(dist, dim=0)
    # Calculating probability mass function approximation, given the preliminary assignment
    pmf = get_pmf(prelim_assignment, zero_idx, num_centroids=len(dist), biased=False, device=device)

    H = []
    for P in pmf:
        if P != 0:
            H.append(-P * torch.log2(P))

    return sum(H)

def get_num_nonzeros(net):
    num_non_zeros = 0
    for param in net:
        num_non_zeros += net[param][net[param] != 0].size
    return num_non_zeros

def get_num_zeros(net):
    num_zeros = 0
    for param in net:
        num_zeros += net[param][net[param] == 0].size
    return num_zeros

def get_sparsity(net):
    num_zeros = 0
    num_params = 0
    for param in net:
        num_zeros += net[param][net[param] == 0].size
        num_params += net[param].size
    return num_zeros / num_params

def get_centroid_stats(model_diff, print_all=False, return_trainables_only=True, ignore_one_dim_params=True):
    results_trainable = []
    results = []
    num_sparse_layers = 0
    num_layers = 0
    for module_name in model_diff:
        if ignore_one_dim_params and (len(model_diff[module_name].shape) < 2 or "downsample" in module_name or "aux_classifier" in module_name):
            continue
        module = copy.deepcopy(model_diff[module_name])
        vals, cts = np.unique(module, return_counts=True)
        l_spars = module[module == 0].size / module.size
        if print_all:
            results.append([module_name, ["%.6f"%item for item in vals.tolist()], cts.tolist(), l_spars])
        elif module.any() and ("running_var" not in module_name and "running_mean" not in module_name):
            results_trainable.append([module_name, ["%.6f"%item for item in vals.tolist()], cts.tolist(), l_spars])
        elif module.any() and ("running_var" in module_name or "running_mean" in module_name):
            results.append([module_name, ["%.6f"%item for item in vals.tolist()], cts.tolist(), l_spars])
        elif not module.any() and ".weight" in module_name:
            num_sparse_layers += 1
        if module.any() and ".weight" in module_name:
            num_layers += 1
    if not print_all:
        results_trainable.append([num_layers+num_sparse_layers, "total layers"])
        results_trainable.append([num_sparse_layers, "sparse layers"])
    if return_trainables_only:
        return results_trainable
    else:
        return results_trainable, results

class LRP:
    """
    LRP Class to calculate relevances of intermediate layers regarding input/output and/or weights.
    Currently, the model is restricted to classification tasks.

    Examples::
        >>> lrp = LRP(model, data, target)
        >>> relevances = lrp.get_relevances()
    """
    def __init__(self,
                 model: Module,
                 data: Tensor,
                 target: Tensor,
                 output: Tensor = None,
                 output_type: str = "output",
                 composite: str = "epsilon_flat",
                 weight_rule: str = "epsilon",
                 canonizer: str = "vgg"):
        """

        Parameters
        ----------
        model: Module
            Pytorch model that is to be analyzed (classification model)
        data: Tensor
            Input data that is used to generate a prediction
        target: Tensor
            True label representing the class to be predicted of shape (batch_size, 1)
        output_type: str
            Output type that is backpropagated. Possible types are ones (propagate ones),
            output (propagate output), probability (propagate probabilities)
        composite: str
            Composite name for LRP calculations
        weight_rule: str
            Rule for relevance calculations regarding weights.
        """

        self.model = model.eval()
        self.data = data
        self.target = target
        if torch.cuda.device_count() > 1:
            self.model = model.to("cuda:1")
            self.data = data.to("cuda:1")
            self.target = target.to("cuda:1")
        self.device = self.data.device


        if (isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model, torch.nn.parallel.DistributedDataParallel)):
            self.model = self.model.module.to(self.device)

        half_precision = False
        if half_precision:
            self.model = self.model.half()
            self.data = self.data.half()


        self.output = output if isinstance(output, Tensor) else self.model(self.data)

        self.output_relevance = None
        self.composite = composite
        assert composite in COMPOSITES.keys(), "Not a valid composite for relevance calculation of inputs."
        self.weight_rule = weight_rule
        assert weight_rule in RULES.keys(), "Not a valid rule for relevance calculation of weights."
        self.output_type = output_type
        assert output_type in ["ones", "probability", "output"]
        self.canonizer = canonizer
        assert canonizer in ["vgg", "resnet", "resnetcifar", "mobilenet"]

    def get_relevances(self) -> Tuple[Dict, OrderedDict, OrderedDict]:
        """ Get relevances regarding weights, inputs, and output for all intermediate layers as dictionaries.

        Returns
        -------
        Tuple[Dict, OrderedDict, OrderedDict]
            Relevances regarding weights, inputs and outputs
        """
        if self.output_relevance is None:
            self.do_backpropagation()

        return self.get_relevances_weights(), self.get_relevances_inputs(), self.get_relevances_output()

    def get_relevances_weights(self) -> Dict:
        """ Get relevances from all parameters in the same format as the state dictionary.

        Returns
        -------
        Dict
            Relevance regarding parameters
        """
        if self.output_relevance is None:
            self.do_backpropagation()

        relevance_dict = {}
        for name, parameter in self.model.named_parameters():
            if hasattr(parameter, "relevance"):
                relevance_dict[name] = parameter.relevance

        return relevance_dict

    def get_relevances_inputs(self) -> OrderedDict:
        """ Get relevances regarding input from all intermediate layers in the format of an ordered dict.

        Returns
        -------
        OrderedDict
            Relevance regarding inputs
        """
        if self.output_relevance is None:
            self.do_backpropagation()

        relevance_dict = OrderedDict()

        for name, layer in self.model.named_modules():
            if hasattr(layer, "input_relevance"):
                relevance_dict[name] = layer.input_relevance

        return relevance_dict

    def get_relevances_output(self) -> OrderedDict:
        """ Get relevances regarding output from all intermediate layers in the format of an ordered dict.

        Returns
        -------
        OrderedDict
            Relevance regarding output
        """
        if self.output_relevance is None:
            self.do_backpropagation()

        relevance_dict = OrderedDict()

        for name, layer in self.model.named_modules():
            if hasattr(layer, "output_relevance"):
                relevance_dict[name] = layer.output_relevance

        return relevance_dict

    def do_backpropagation(self):
        """ Backpropagation of relevance by modifying model and autograd
        """
        self.create_output_relevance()

        if self.canonizer == 'vgg':
            composite_kwargs = {
                'canonizers': [VGGCanonizer()],
                'weight_rule': RULES[self.weight_rule]()
            }
        elif self.canonizer == 'resnet':
            composite_kwargs = {
                'canonizers': [ResNetCanonizer()],
                'weight_rule': RULES[self.weight_rule]()
            }
        elif self.canonizer == 'resnetcifar':
            composite_kwargs = {
                'canonizers': [VGGCanonizer()],
                'weight_rule': RULES[self.weight_rule]()
            }
        elif self.canonizer == 'mobilenet':
            composite_kwargs = {
                'canonizers': [MobileNetCanonizer()],
                'weight_rule': RULES[self.weight_rule]()
            }

        composite = COMPOSITES[self.composite](**composite_kwargs)

        DeepLab_condition = self.model.__class__.__name__ == "DeepLabV3" if not (isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model, torch.nn.parallel.DistributedDataParallel)) \
            else self.model.module.__class__.__name__ == "DeepLabV3"

        with composite.context(self.model) as modified:
            data = self.data
            data.requires_grad_()
            out = modified(data)
            if DeepLab_condition:
                out = out['out']
            torch.autograd.backward((out,), (self.output_relevance.to(self.device),))
            # del out

    def create_output_relevance(self):
        """ Defining the relevance output tensor that is used to propagate towards input.
        """
        DeepLab_condition = self.model.__class__.__name__ == "DeepLabV3" if not (isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model, torch.nn.parallel.DistributedDataParallel))\
                            else self.model.module.__class__.__name__ == "DeepLabV3"
        if DeepLab_condition:
            self.output = self.output['out']

        output_relevance = torch.zeros_like(self.output)

        if self.output_type == "ones":
            # use one as output for true class for each sample
            outputs = torch.sign(self.output).abs()
        elif self.output_type == "output":
            # keep output logit of true class for each sample
            outputs = self.output
        elif self.output_type == "probability":
            # keep output probability of true class for each sample
            outputs = torch.nn.functional.softmax(self.output, dim=1)
        else:
            outputs = torch.nn.functional.softmax(self.output, dim=1)

        # TODO: implement without using loop
        for sample in range(output_relevance.shape[0]):
            if DeepLab_condition:
                num_classes = self.output.shape[1]
                # create mask from target.shape (W, H) to (C, W, H) to get the right indices of output_relevance
                mask = torch.stack([self.target[sample] == i for i in range(num_classes)])
                output_relevance[sample, mask] = outputs[sample, mask]
            else:
                output_relevance[sample, self.target[sample]] = outputs[sample, self.target[sample]]

        # normalize output
        output_relevance = output_relevance/output_relevance.shape[0]
        self.output_relevance = output_relevance
