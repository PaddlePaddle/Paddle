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
import functools
from ..framework import Variable, default_main_program, in_dygraph_mode, dygraph_only, Parameter, ParamBase, _varbase_creator, _dygraph_tracer
import pickle
import six
from . import learning_rate_scheduler
import warnings
from .. import core
from .base import guard
from paddle.fluid.dygraph.jit import SaveLoadConfig, deprecate_save_load_configs
from paddle.fluid.dygraph.io import _construct_program_holders, _construct_params_and_buffers

__all__ = [
    'save_dygraph',
    'load_dygraph',
]


# NOTE(chenweihang): deprecate load_dygraph's argument keep_name_table,
# ensure compatibility when user still use keep_name_table argument
def deprecate_keep_name_table(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        def __warn_and_build_configs__(keep_name_table):
            warnings.warn(
                "The argument `keep_name_table` has deprecated, please use `SaveLoadConfig.keep_name_table`.",
                DeprecationWarning)
            config = SaveLoadConfig()
            config.keep_name_table = keep_name_table
            return config

        # deal with arg `keep_name_table`
        if len(args) > 1 and isinstance(args[1], bool):
            args = list(args)
            args[1] = __warn_and_build_configs__(args[1])
        # deal with kwargs
        elif 'keep_name_table' in kwargs:
            kwargs['config'] = __warn_and_build_configs__(kwargs[
                'keep_name_table'])
            kwargs.pop('keep_name_table')
        else:
            # do nothing
            pass

        return func(*args, **kwargs)

    return wrapper


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
# @dygraph_only
@deprecate_save_load_configs
@deprecate_keep_name_table
def load_dygraph(model_path, config=None):
    '''
    :api_attr: imperative
    
    Load parameter state dict from disk.

    .. note::
        Due to some historical reasons, if you load ``state_dict`` from the saved 
        result of `paddle.io.save_inference_model`, the structured variable name 
        will cannot be restored. You need to set the argument `use_structured_name=False` 
        when using `Layer.set_state_dict` later.

    Args:
        model_path(str) : The file prefix store the state_dict. 
            (The path should Not contain suffix '.pdparams') 
        config (SaveLoadConfig, optional): :ref:`api_imperative_jit_saveLoadConfig`
            object that specifies additional configuration options, these options 
            are for compatibility with ``jit.save/io.save_inference_model`` formats. 
            Default None.

    Returns:
        state_dict(dict) : the dict store the state_dict

    Examples:
        .. code-block:: python

            import paddle
            
            paddle.disable_static()

            emb = paddle.nn.Embedding([10, 10])

            state_dict = emb.state_dict()
            paddle.save(state_dict, "paddle_dy")

            scheduler = paddle.optimizer.lr_scheduler.NoamLR(
                d_model=0.01, warmup_steps=100, verbose=True)
            adam = paddle.optimizer.Adam(
                learning_rate=scheduler,
                parameters=emb.parameters())
            state_dict = adam.state_dict()
            paddle.save(state_dict, "paddle_dy")

            para_state_dict, opti_state_dict = paddle.load("paddle_dy")

    '''
    # deal with argument `model_path`
    model_prefix = model_path
    if model_prefix.endswith(".pdparams"):
        model_prefix = model_prefix[:-9]
    elif model_prefix.endswith(".pdopt"):
        model_prefix = model_prefix[:-6]

    para_dict = None
    opti_dict = None
    params_file_path = model_prefix + ".pdparams"
    opti_file_path = model_prefix + ".pdopt"

    # deal with argument `configs`
    configs = config
    if configs is None:
        configs = SaveLoadConfig()

    if not os.path.exists(params_file_path) and not os.path.exists(
            opti_file_path):
        # Load state dict by `jit.save/io.save_inference_model` save format
        # NOTE(chenweihang): [ Compatibility of save_inference_model save format ]
        # The model saved by `save_inference_model` does not completely correspond to 
        # the information required by the `state_dict` under the dygraph. 
        # `save_inference_model` not save structured name, we need to remind 
        # the user to configure the `use_structured_name` argument when `set_state_dict`
        # NOTE(chenweihang): `jit.save` doesn't save optimizer state 

        # 1. check model path
        if not os.path.isdir(model_prefix):
            raise ValueError("Model saved directory '%s' is not exists." %
                             model_prefix)

        # 2. load program desc & construct _ProgramHolder
        programs = _construct_program_holders(model_path,
                                              configs.model_filename)

        # 3. load layer parameters & buffers
        # NOTE: using fluid.dygraph.guard() here will cause import error in py2
        with guard():
            persistable_var_dict = _construct_params_and_buffers(
                model_prefix,
                programs,
                configs.separate_params,
                configs.params_filename,
                append_suffix=False)

            # 4. construct state_dict
            para_dict = dict()
            for var_name in persistable_var_dict:
                para_dict[var_name] = persistable_var_dict[var_name].numpy()
    else:
        # Load state dict by `save_dygraph` save format
        para_dict = {}
        if os.path.exists(params_file_path):
            with open(params_file_path, 'rb') as f:
                para_dict = pickle.load(f) if six.PY2 else pickle.load(
                    f, encoding='latin1')

        if not configs.keep_name_table and "StructuredToParameterName@@" in para_dict:
            del para_dict["StructuredToParameterName@@"]

        if os.path.exists(opti_file_path):
            with open(opti_file_path, 'rb') as f:
                opti_dict = pickle.load(f) if six.PY2 else pickle.load(
                    f, encoding='latin1')

    return para_dict, opti_dict
