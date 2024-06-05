'''
The copyright in this software is being made available under the Clear BSD
License, included below. No patent rights, trademark rights and/or 
other Intellectual Property Rights other than the copyrights concerning 
the Software are granted under this license.

The Clear BSD License

Copyright (c) 2019-2023, Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V. & The NNCodec Authors.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted (subject to the limitations in the disclaimer below) provided that
the following conditions are met:

     * Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.

     * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

     * Neither the name of the copyright holder nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
'''
    
import sys
assert sys.version_info >= (3, 6)

import os
import numpy as np
from timeit import default_timer as timer

import nnc_core
from nnc_core import nnr_model

def __print_output_line( outputString, verbose=True ):
    if verbose:
        sys.stdout.write(outputString)
        sys.stdout.flush()

def compress( 
    parameter_dict,
    bitstream_path="./bitstream.nnc",
    qp=-38,
    qp_density=2,
    nonweight_qp=-75,
    qp_per_tensor=None,
    use_dq=True,
    codebook_mode=0,
    scan_order=0,
    lambda_scale=0,
    param_opt=True,
    cabac_unary_length_minus1=10,
    opt_qp=False,
    ioq=False,
    bnf=False,
    lsa=False,
    fine_tune=False,
    block_id_and_param_type=None,
    model=None,
    model_executer=None,
    verbose=False,
    return_bitstream=False
    ):

    try:
        start = timer()
        start_overall = start
        __print_output_line("INITIALIZE APPROXIMATOR AND ENCODER...", verbose=verbose)
        if isinstance(parameter_dict, dict) and all( [isinstance(a, np.ndarray) for a in parameter_dict.values()] ) and (all([ (a.dtype==np.float32 or a.dtype==np.int32) for a in parameter_dict.values()])):
            model_parameters = parameter_dict
            
            if isinstance(model, nnc_core.nnr_model.NNRModel):
                nnc_mdl = model
            else:
                nnc_mdl = nnc_core.nnr_model.NNRModel(parameter_dict)

            if model_executer is not None:
                assert isinstance( model_executer, nnc_core.nnr_model.ModelExecute ), "model_executer must be of type ModelExecute!"
        else:
            raise SystemExit("Parameter dict must be a dict (key-value pairs). The keys shall be stings, specifying the tensor names. The values shalls be numpy arrays (ndarray) of type float32 or int32!")
    except:
        raise SystemExit("Can not read parameter_dict: {}".format(parameter_dict))

    if block_id_and_param_type is not None:
        blkIdParamTypeOk = nnc_core.nnr_model.sanity_check_block_id_and_param_type( block_id_and_param_type, parameter_dict )
        if blkIdParamTypeOk:
            nnc_core.nnr_model.set_block_id_and_param_type( nnc_mdl.model_info , block_id_and_param_type )
        else:
            print("INFO: Sanity check for block_id_and_param_type failed! block_id_and_param_type has been set to 'None', and the flags 'lsa' and 'bnf' have been set to 'False'!")
            block_id_and_param_type = None
            lsa = False
            bnf = False
            
    
    if model_executer:
        if lsa and not model_executer.has_tune_lsa():
            print("INFO: Tuning (training) of LSA parameters (tune_model) not implemented by model_executer! 'lsa' has been set to 'False'!")
            lsa = False
        if fine_tune and not model_executer.has_tune_ft():
            print("INFO: Fine tuning (training) of parameters (tune_model) not implemented by model_executer! 'fine_tune' has been set to 'False'!")
            fine_tune = False
        if ioq and not model_executer.has_eval():
            print("INFO: Evaluation (inference on a reduced dataset) of parameters (eval_model) not implemented by model_executer! ioq' has been set to 'False'!")
            ioq = False
                    
    ##INITIALIZATION
    approx_data =  nnc_core.approximator.init_approx_data(  model_parameters,
                                                            nnc_mdl.model_info, 
                                                            qp_density=qp_density, 
                                                            scan_order=scan_order
                                                         )

    ApproxInfoO = nnc_core.approximator.ApproxInfo( approx_data,
                                                    nnc_mdl.model_info,
                                                    "uniform" if codebook_mode==0 else "codebook",
                                                    codebook_mode,
                                                    qp,
                                                    opt_qp,
                                                    not use_dq,
                                                    cabac_unary_length_minus1,
                                                    lambda_scale,
                                                    nonweight_qp=nonweight_qp,
                                                    qp_per_tensor=qp_per_tensor
                                                )
    approx_info = ApproxInfoO.approx_info

    enc_info = {
            "cabac_unary_length_minus1" : cabac_unary_length_minus1,
            "param_opt_flag"     : param_opt,
        }
    end = timer()
    __print_output_line("DONE in {:.4f} s\n".format(end-start), verbose=verbose)

    ##PREPROCESSING
    if ioq:
        assert model_executer is not None, "model_executer must be available in order to run IOQ!"
        start = timer()
        __print_output_line("PREPROCESSING, IOQ...\n", verbose=verbose) 
        nnc_core.approximator.inference_based_qp_opt(
            approx_info,
            nnc_mdl.model_info,
            model_executer,
            approx_data,
            enc_info["param_opt_flag"],
            enc_info["cabac_unary_length_minus1"],
            verbose=verbose,
        )
        end = timer()
        __print_output_line("DONE in {:.4f} s\n".format( end-start ), verbose=verbose)   

    ##LSA and FT
    if (lsa or fine_tune):
        assert model_executer is not None, "model_executer must be available in order to run LSA and/or FT!"
        start = timer()
        __print_output_line("PREPROCESSING, LSA/FT...\n", verbose=verbose) 
        nnc_core.approximator.run_ft_and_lsa(
            nnc_mdl.model_info,
            approx_data,
            ApproxInfoO,
            model_executer,
            block_id_and_param_type,
            lsa,
            fine_tune,
            use_dq,
            verbose
        )
        end = timer()
        __print_output_line("DONE in {:.4f} s\n".format( end-start ), verbose=verbose)  
    ##BNF
    if bnf:
        start = timer()
        __print_output_line("PREPROCESSING, BNF...", verbose=verbose)    
        nnc_core.approximator.fold_bn(nnc_mdl.model_info, approx_data, ApproxInfoO)
        end = timer()
        __print_output_line("DONE in {:.4f} s\n".format(end-start), verbose=verbose)


    #####QUANTIZATION AND ENCODING
    start = timer() 
    __print_output_line("APPROXIMATING WITH METHOD {}...".format(approx_info["approx_method"]), verbose=verbose)
    approx_data_enc = nnc_core.approximator.approx( approx_info,
                                                nnc_mdl.model_info,
                                                approx_data,
                                                enc_info["param_opt_flag"]
                                               )
    end = timer()
    __print_output_line("DONE in {:.4f} s\n".format( end-start ), verbose=verbose)

    start = timer()
    __print_output_line("ENCODING...", verbose=verbose)
    bitstream, bs_sizes = nnc_core.coder.encode(  enc_info,
                                                     nnc_mdl.model_info,
                                                     approx_data_enc,
                                                     return_bs_sizes=True
                                                     )
    end = timer()
    __print_output_line("DONE in {:.4f} s\n".format( end-start ), verbose=verbose)

    original_size = nnc_mdl.model_info["original_size"]

    __print_output_line("COMPRESSED FROM {} BYTES TO {} BYTES ({:.2f} KB, {:.2f} MB, COMPRESSION RATIO: {:.2f} %) in {:.4f} s\n".format(original_size, len(bitstream), len(bitstream)/1000.0, len(bitstream)/1000000.0, len(bitstream)/original_size*100, end-start_overall), verbose=True)
    
    if bitstream_path is not None:
        with open( bitstream_path, "wb" ) as br_file:
            br_file.write( bitstream )

    if return_bitstream:
        return bitstream, original_size


def decompress( bitstream_or_path, 
                block_id_and_param_type=None, 
                return_model_information=False, 
                verbose=False,
                reconstruct_lsa=False,
                reconstruct_bnf=False
                ):

    dec_model_info  = {'parameter_type': {},
                      'parameter_dimensions': {},
                      'parameter_index': {},
                      'block_identifier': {},
                      'topology_storage_format' : None,
                      'topology_compression_format' : None,
                      'performance_maps' : { "mps" : {}, "lps" : {}},
                      'performance_map_flags' : { "mps_sparsification_flag" : {}, "lps_sparsification_flag" : {},
                                                  "mps_pruning_flag" : {}, "lps_pruning_flag" : {},
                                                  "mps_unification_flag" : {}, "lps_unification_flag" : {},
                                                  "mps_decomposition_performance_map_flag" : {}, "lps_decomposition_performance_map_flag" : {},
                                                } 
                      }

    model_information = { 'topology_storage_format' : None,
                          'performance_maps' : {},
                          'performance_map_flags' : {}
                        }

    if block_id_and_param_type is not None:
        blkIdParamTypeOk = nnc_core.nnr_model.sanity_check_block_id_and_param_type( block_id_and_param_type )
        if blkIdParamTypeOk == False:
            print("INFO: Sanity check for block_id_and_param_type failed! block_id_and_param_type has been set to 'None'!")
            block_id_and_param_type = None
        else:
            nnc_core.nnr_model.set_block_id_and_param_type( dec_model_info, block_id_and_param_type )

    hls_bytes = {}
    start = timer()
    __print_output_line("DECODING...", verbose=verbose)
    if isinstance(bitstream_or_path, bytearray):
        bitstream = bitstream_or_path
    elif os.path.exists(os.path.expanduser(bitstream_or_path)):
        with open( os.path.expanduser(bitstream_or_path), "rb" ) as br_file:
            bitstream = br_file.read()
    else:
        raise SystemExit( "Could not read bitstream or bitstream_path: {}".format(bitstream_or_path) )

    dec_approx_data = nnc_core.coder.decode(bitstream, dec_model_info, hls_bytes)
    end = timer()
    __print_output_line("DONE in {:.4f} s\n".format( end-start ), verbose=verbose)

    start = timer()
    rec_approx_data = dec_approx_data
    __print_output_line("RECONSTRUCTING...", verbose=verbose)
    nnc_core.approximator.rec(rec_approx_data )
    if reconstruct_bnf: ## TODO: check if there are cases where must be dis/enabled
        nnc_core.approximator.unfold_bn(dec_model_info, rec_approx_data)
    if reconstruct_lsa: ## TODO: check if there are cases where must be dis/enabled
        nnc_core.approximator.apply_lsa(dec_model_info, rec_approx_data)
    rec_approx_data = nnc_core.approximator.recompose_params( dec_model_info, rec_approx_data)
    end = timer()
    __print_output_line("DONE in {:.4f} s\n".format( end-start ), verbose=verbose)
    
    if return_model_information:
        model_information["topology_storage_format"] = dec_model_info["topology_storage_format"]
        model_information["performance_maps"]        = dec_model_info["performance_maps"]
        model_information["performance_map_flags"]   = dec_model_info["performance_map_flags"]

        return rec_approx_data["parameters"], model_information
    else:
        return rec_approx_data["parameters"]
