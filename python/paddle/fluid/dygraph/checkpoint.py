# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import os
import collections
from ..framework import Variable, default_main_program, in_dygraph_mode, dygraph_only, Parameter, ParamBase, _varbase_creator, _dygraph_tracer
import pickle
import six
from . import learning_rate_scheduler
import warnings
from .. import core
from paddle.fluid.dygraph.io import VARIABLE_FILENAME, EXTRA_VAR_INFO_FILENAME, _load_persistable_vars

__all__ = [
    'save_dygraph',
    'load_dygraph',
]


@dygraph_only
def save_dygraph(state_dict, model_path):
    '''
    :api_attr: imperative

    Save Layer's state_dict to disk. This will generate a file with suffix ".pdparams"
    
    The state_dict is get from Layers.state_dict function
    
    Args:
        state_dict(dict) : The state dict to be saved.
        model_path(str) : the file prefix to save the state_dict. The format is "dirname/file_prefix". If file_prefix is empty str. A exception will be raised

    Returns:
        None

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            with fluid.dygraph.guard():
                emb = fluid.dygraph.Embedding([10, 10])

                state_dict = emb.state_dict()
                fluid.save_dygraph( state_dict, "paddle_dy")

                adam = fluid.optimizer.Adam( learning_rate = fluid.layers.noam_decay( 100, 10000),
                                             parameter_list = emb.parameters() )

                state_dict = adam.state_dict()
                fluid.save_dygraph( state_dict, "paddle_dy")

    '''

    base_name = os.path.basename(model_path)
    assert base_name != "", "The input model_path MUST be format of dirname/filename [dirname\\filename in Windows system], but received filename is empty string."

    suffix = ".pdparams"
    assert len(state_dict) > 0, "state_dict is empty, no need to save"

    param_num = 0
    for k, v in state_dict.items():
        if isinstance(v, ParamBase):
            param_num += 1

    if param_num == 0:
        suffix = ".pdopt"

    model_dict = {}
    name_table = {}
    for k, v in state_dict.items():
        if isinstance(v, (Variable, core.VarBase)):
            model_dict[k] = v.numpy()
            name_table[k] = v.name
        else:
            model_dict[k] = v
    model_dict["StructuredToParameterName@@"] = name_table

    file_name = model_path + suffix
    dir_name = os.path.dirname(file_name)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open(file_name, 'wb') as f:
        pickle.dump(model_dict, f, protocol=2)


# TODO(qingqing01): remove dygraph_only to support loading static model.
# maybe need to unify the loading interface after 2.0 API is ready.
#@dygraph_only
def load_dygraph(model_path, keep_name_table=False):
    '''
    :api_attr: imperative
    
    Load parameter state_dict from disk.

    Args:
        model_path(str) : The file prefix store the state_dict. (The path should Not contain suffix '.pdparams') 
        keep_name_table(bool, optional) : Whether keep structed name to parameter name conversion table in output dict. 
                                          Default : False

    Returns:
        state_dict(dict) : the dict store the state_dict

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            
            with fluid.dygraph.guard():
                emb = fluid.dygraph.Embedding([10, 10])

                state_dict = emb.state_dict()
                fluid.save_dygraph( state_dict, "paddle_dy")

                adam = fluid.optimizer.Adam( learning_rate = fluid.layers.noam_decay( 100, 10000),
                                             parameter_list = emb.parameters() )
                state_dict = adam.state_dict()
                fluid.save_dygraph( state_dict, "paddle_dy")

                para_state_dict, opti_state_dict = fluid.load_dygraph( "paddle_dy")

    '''

    model_prefix = model_path
    if model_prefix.endswith(".pdparams"):
        model_prefix = model_prefix[:-9]
    elif model_prefix.endswith(".pdopt"):
        model_prefix = model_prefix[:-6]

    para_dict = None
    opti_dict = None
    params_file_path = model_prefix + ".pdparams"
    opti_file_path = model_prefix + ".pdopt"
    if not os.path.exists(params_file_path) and not os.path.exists(
            opti_file_path):
        # Load state dict by `jit.save` save format
        # TODO(chenweihang): [Why not support `io.save_infernece_model` save format here]
        # The model saved by `save_inference_model` does not completely correspond to 
        # the information required by the `state_dict` under the dygraph. 
        # Although we reluctantly restore the `state_dict` in some scenarios, 
        # this may not be complete and there are some limitations, so this function 
        # will be considered later. The limitations include:
        #   1. `save_inference_model` not save structured name, we need to remind 
        # the user to configure the `use_structured_name` argument when `set_dict`, 
        # but this argument is currently not public
        #   2. if `save_inference_model` save all persistable variables in a single file,
        # user need to give the variable name list to load `state_dict`

        # 1. check model path
        if not os.path.isdir(model_prefix):
            raise ValueError("Model saved directory '%s' is not exists." %
                             model_prefix)
        # 2. load `__variables.info__`
        var_info_path = os.path.join(model_prefix, EXTRA_VAR_INFO_FILENAME)
        if not os.path.exists(var_info_path):
            raise RuntimeError(
                "No target can be loaded. Now only supports loading `state_dict` from "
                "the result saved by `imperative.save` and `imperative.jit.save`."
            )
        with open(var_info_path, 'rb') as f:
            extra_var_info = pickle.load(f)
        # 3. load `__variables__`
        # TODO(chenweihang): now only supports loading from default save format:
        # - all persistable vars saved in one file named `__variables__`
        # for other case, we may need to modify the arguments of this API
        var_file_path = os.path.join(model_prefix, VARIABLE_FILENAME)
        if not os.path.exists(var_file_path):
            raise RuntimeError(
                "The parameter file to be loaded was not found. "
                "Now only supports loading from the default save format, "
                "and does not support custom params_filename and "
                "save parameters separately.")
        # 4. load all persistable vars
        load_var_list = []
        for name in sorted(extra_var_info):
            var = _varbase_creator(name=name, persistable=True)
            load_var_list.append(var)
        _dygraph_tracer().trace_op(
            type='load_combine',
            inputs={},
            outputs={'Out': load_var_list},
            attrs={'file_path': var_file_path})
        # 5. construct state_dict
        para_dict = dict()
        for var in load_var_list:
            structured_name = extra_var_info[var.name].get('structured_name',
                                                           None)
            if structured_name is None:
                raise RuntimeError(
                    "Cannot find saved variable (%s)'s structured name in saved model.",
                    var.name)
            para_dict[structured_name] = var.numpy()
        # NOTE: `jit.save` doesn't save optimizer state
    else:
        # Load state dict by `save_dygraph` save format
        if os.path.exists(params_file_path):
            with open(params_file_path, 'rb') as f:
                para_dict = pickle.load(f) if six.PY2 else pickle.load(
                    f, encoding='latin1')

        if not keep_name_table and "StructuredToParameterName@@" in para_dict:
            del para_dict["StructuredToParameterName@@"]

        if os.path.exists(opti_file_path):
            with open(opti_file_path, 'rb') as f:
                opti_dict = pickle.load(f) if six.PY2 else pickle.load(
                    f, encoding='latin1')

    return para_dict, opti_dict
