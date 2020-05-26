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
from ..framework import Variable, default_main_program, in_dygraph_mode, dygraph_only, Parameter, ParamBase
import pickle
import six
from . import learning_rate_scheduler
import warnings
from .. import core

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

    for k, v in state_dict.items():
        if not isinstance(v, ParamBase):
            suffix = ".pdopt"
        break

    model_dict = {}
    name_table = {}
    for k, v in state_dict.items():
        if isinstance(v, (Variable, core.VarBase)):
            model_dict[k] = v.numpy()
        else:
            model_dict[k] = v
        name_table[k] = v.name
    model_dict["StructuredToParameterName@@"] = name_table

    file_name = model_path + suffix
    dir_name = os.path.dirname(file_name)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open(file_name, 'wb') as f:
        pickle.dump(model_dict, f, protocol=2)


@dygraph_only
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

    params_file_path = model_prefix + ".pdparams"
    if not os.path.exists(params_file_path):
        raise RuntimeError("Parameter file [ {} ] not exists".format(
            params_file_path))

    with open(params_file_path, 'rb') as f:
        para_dict = pickle.load(f) if six.PY2 else pickle.load(
            f, encoding='latin1')

    if not keep_name_table and "StructuredToParameterName@@" in para_dict:
        del para_dict["StructuredToParameterName@@"]
    opti_dict = None
    opti_file_path = model_prefix + ".pdopt"
    if os.path.exists(opti_file_path):
        with open(opti_file_path, 'rb') as f:
            opti_dict = pickle.load(f) if six.PY2 else pickle.load(
                f, encoding='latin1')

    return para_dict, opti_dict
