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
from ..framework import Variable, default_main_program, in_dygraph_mode, dygraph_only, Parameter
import pickle
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
                emb = fluid.dygraph.Embedding( "emb", [10, 10])

                state_dict = emb.state_dict()
                fluid.save_dygraph( state_dict, "paddle_dy")

                adam = fluid.optimizer.Adam( learning_rate = fluid.layers.noam_decay( 100, 10000) )

                state_dict = adam.state_dict()
                fluid.save_dygraph( state_dict, "paddle_dy")

    '''

    base_name = os.path.basename(model_path)
    assert base_name != "", "model_path MUST be format of dirname/filename [dirname\\filename in Window], Now filename is empty str"

    suffix = ".pdparams"
    assert len(state_dict) > 0, "state_dict is empty, no need to save"

    for k, v in state_dict.items():
        if not isinstance(v, Parameter):
            suffix = ".pdopt"
        break

    core._save_dygraph_dict(model_path + suffix, state_dict)


@dygraph_only
def load_dygraph(model_path):
    '''
    Load parameter state_dict from disk.

    Args:
        model_path(str) : The file prefix store the state_dict. (The path should Not contain suffix '.pdparams') 

    Returns:
        state_dict(dict) : the dict store the state_dict

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            
            with fluid.dygraph.guard():
                emb = fluid.dygraph.Embedding( "emb", [10, 10])

                state_dict = emb.state_dict()
                fluid.save_dygraph( state_dict, "paddle_dy")

                adam = fluid.optimizer.Adam( learning_rate = fluid.layers.noam_decay( 100, 10000) )
                state_dict = adam.state_dict()
                fluid.save_dygraph( state_dict, "padle_dy")

                para_state_dict, opti_state_dict = fluid.load_dygraph( "paddle_dy")

    '''

    params_file_path = model_path + ".pdparams"
    if not os.path.exists(params_file_path):
        raise RuntimeError("Parameter file [ {} ] not exists".format(
            params_file_path))

    para_dict = core._load_dygraph_dict(params_file_path)

    opti_dict = None
    opti_file_path = model_path + ".pdopt"
    if os.path.exists(opti_file_path):
        opti_dict = core._load_dygraph_dict(opti_file_path)

    return para_dict, opti_dict


@dygraph_only
def load_optimizer(model_path):
    '''
    Load optimizer state_dict from disk.

    Args:
        model_path(str) : The file prefix store the state_dict. (The path should Not contain shuffix '.pdparams')

    Returns:
        state_dict(dict) : the dict store the state_dict

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            with fluid.dygraph.guard():
                adam = fluid.optimizer.Adam(0.001)

                state_dict = adam.state_dict()
                fluid.save_optimizer( state_dict, "opt_adam")

                fluid.load_optimizer( "opt_adam")

    '''

    assert in_dygraph_mode(), "save_optimizer only work in dygraph mode"
    opt_file_path = model_path + ".pdopt"
    if not os.path.exists(opt_file_path):
        raise RuntimeError("Optimizer file [ {} ] not exists".format(
            opt_file_path))
    return core._load_dygraph_dict(opt_file_path)
