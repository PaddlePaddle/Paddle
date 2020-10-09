# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import pickle
import six
import warnings

import paddle

# deprecated module import
from paddle import fluid
from paddle.fluid import core
from paddle.fluid.framework import Variable, _varbase_creator, _dygraph_tracer
from paddle.fluid.dygraph.io import _construct_program_holders, _construct_params_and_buffers, EXTRA_VAR_INFO_FILENAME

__all__ = [
    'save',
    'load',
]


def _build_saved_state_dict(state_dict):
    save_dict = {}
    name_table = {}
    for key, value in state_dict.items():
        if isinstance(value, (Variable, core.VarBase)):
            save_dict[key] = value.numpy()
            name_table[key] = value.name
        else:
            save_dict[key] = value
    save_dict["StructuredToParameterName@@"] = name_table

    return save_dict


def _load_state_dict_from_save_inference_model(model_path, config):
    # 1. load program desc & construct _ProgramHolder
    programs = _construct_program_holders(model_path, config.model_filename)

    # 2. load layer parameters & buffers
    with fluid.dygraph.guard():
        persistable_var_dict = _construct_params_and_buffers(
            model_path,
            programs,
            config.separate_params,
            config.params_filename,
            append_suffix=False)

        # 3. construct state_dict
        load_param_dict = dict()
        for var_name in persistable_var_dict:
            load_param_dict[var_name] = persistable_var_dict[var_name].numpy()

        # if __variables.info__ exists, we can recover structured_name
        var_info_path = os.path.join(model_path, EXTRA_VAR_INFO_FILENAME)
        if os.path.exists(var_info_path):
            with open(var_info_path, 'rb') as f:
                extra_var_info = pickle.load(f)
            structured_para_dict = dict()
            for var_name in load_param_dict:
                structured_name = extra_var_info[var_name].get(
                    'structured_name', None)
                assert structured_name is not None, "Cannot find saved variable (%s)'s structured name in saved model." % var_name
                structured_para_dict[structured_name] = load_param_dict[
                    var_name]
            load_param_dict = structured_para_dict

    return load_param_dict


def _load_state_dict_from_save_params(model_path):
    # Try to load all the files in the directory in VarBase format, 
    # the file name is used as the name of VarBase
    load_var_list = []

    # 1. load file names
    var_name_list = []
    for root, _, files in os.walk(model_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            tmp_var_name = os.path.relpath(file_path, model_path)
            var_name = tmp_var_name.replace("\\", "/")
            var_name_list.append(var_name)

    # 2. create and load VarBase
    with fluid.dygraph.guard():
        for name in var_name_list:
            new_var = _varbase_creator(name=name, persistable=True)
            _dygraph_tracer().trace_op(
                type='load',
                inputs={},
                outputs={'Out': new_var},
                attrs={'file_path': os.path.join(model_path, name)})
            load_var_list.append(new_var)

    # 3. construct state_dict
    load_param_dict = dict()
    for var in load_var_list:
        load_param_dict[var.name] = var.numpy()

    return load_param_dict


def save(obj, path):
    '''
    Save an object to the specified path.
    
    .. note::
        Now only supports save ``state_dict`` of Layer or Optimizer.
    
    Args:
        obj(Object) : The object to be saved.
        path(str) : The path of the object to be saved. 
          If saved in the current directory, the input path string will be used as the file name. 

    Returns:
        None

    Examples:
        .. code-block:: python

            import paddle

            paddle.disable_static()

            emb = paddle.nn.Embedding(10, 10)
            layer_state_dict = emb.state_dict()
            paddle.save(layer_state_dict, "emb.pdparams")

            scheduler = paddle.optimizer.lr_scheduler.NoamLR(	
                d_model=0.01, warmup_steps=100, verbose=True)
            adam = paddle.optimizer.Adam(
                learning_rate=scheduler,
                parameters=emb.parameters())
            opt_state_dict = adam.state_dict()
            paddle.save(opt_state_dict, "adam.pdopt")
    '''

    # 1. input check
    if not isinstance(obj, dict):
        raise NotImplementedError(
            "Now only supports save state_dict of Layer or Optimizer, "
            "expect dict, but received %s." % type(obj))

    if len(obj) == 0:
        warnings.warn("The input state dict is empty, no need to save.")

    filename = os.path.basename(path)
    if filename == "":
        raise ValueError("The input path MUST be format of dirname/filename "
                         "[dirname\\filename in Windows system], but received "
                         "filename is empty string.")

    # 2. save object
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)

    # TODO(chenweihang): supports save other object
    saved_obj = _build_saved_state_dict(obj)

    with open(path, 'wb') as f:
        pickle.dump(saved_obj, f, protocol=2)


def load(path, config=None):
    '''
    Load an object can be used in paddle from specified path.

    .. note::
        Now only supports load ``state_dict`` of Layer or Optimizer.

    .. note::
        ``paddle.load`` supports loading ``state_dict`` from the result of several 
        paddle1.x save APIs in static mode, but due to some historical reasons, 
        if you load ``state_dict`` from the saved result of 
        ``paddle.static.save_inference_model/paddle.fluid.io.save_params/paddle.fluid.io.save_persistables`` , 
        the structured variable name will cannot be restored. You need to set the argument 
        ``use_structured_name=False`` when using ``Layer.set_state_dict`` later.

    Args:
        path(str) : The path to load the target object. Generally, the path is the target 
            file path, when compatible with loading the saved results of 
            ``paddle.jit.save/paddle.static.save_inference_model`` , the path is a directory. 
        config (SaveLoadConfig, optional): :ref:`api_imperative_jit_saveLoadConfig`
            object that specifies additional configuration options, these options 
            are for compatibility with ``paddle.jit.save/paddle.static.save_inference_model`` 
            formats. Default None.

    Returns:
        Object(Object): a target object can be used in paddle

    Examples:
        .. code-block:: python

            import paddle
            
            paddle.disable_static()

            emb = paddle.nn.Embedding(10, 10)
            layer_state_dict = emb.state_dict()
            paddle.save(layer_state_dict, "emb.pdparams")

            scheduler = paddle.optimizer.lr_scheduler.NoamLR(	
                d_model=0.01, warmup_steps=100, verbose=True)
            adam = paddle.optimizer.Adam(
                learning_rate=scheduler,
                parameters=emb.parameters())
            opt_state_dict = adam.state_dict()
            paddle.save(opt_state_dict, "adam.pdopt")

            load_layer_state_dict = paddle.load("emb.pdparams")
            load_opt_state_dict = paddle.load("adam.pdopt")
    '''
    # 1. input check
    if not os.path.exists(path):
        error_msg = "The path `%s` does not exist."
        # if current path is a prefix, and the path.pdparams or path.pdopt
        # is exist, users may want use `paddle.load` load the result of 
        # `fluid.save_dygraph`, we raise error here for users
        params_file_path = path + ".pdparams"
        opti_file_path = path + ".pdopt"
        if os.path.exists(params_file_path) or os.path.exists(opti_file_path):
            error_msg += " If you want to load the results saved by `fluid.save_dygraph`, " \
                "please specify the full file name, not just the file name prefix. For " \
                "example, it should be written as `paddle.load('model.pdparams')` instead of " \
                "`paddle.load('model')`."
        raise ValueError(error_msg % path)

    if config is None:
        config = paddle.SaveLoadConfig()

    # 2. load target
    load_result = None
    if os.path.isfile(path):
        # we think path is file means this file is created by paddle.save
        with open(path, 'rb') as f:
            load_result = pickle.load(f) if six.PY2 else pickle.load(
                f, encoding='latin1')

        if not config.keep_name_table and "StructuredToParameterName@@" in load_result:
            del load_result["StructuredToParameterName@@"]
    elif os.path.isdir(path):
        # we think path is directory means compatible with loading 
        # store results of static mode related save APIs

        # check whether model file exists
        if config.model_filename is None:
            model_filename = '__model__'
        else:
            model_filename = config.model_filename
        model_file_path = os.path.join(path, model_filename)

        if os.path.exists(model_file_path):
            # Load state dict by `jit.save/io.save_inference_model` save format
            # NOTE(chenweihang): [ Compatibility of save_inference_model save format ]
            # The model saved by `save_inference_model` does not completely correspond to 
            # the information required by the `state_dict` under the dygraph. 
            # `save_inference_model` not save structured name, we need to remind 
            # the user to configure the `use_structured_name` argument when `set_state_dict`
            # NOTE(chenweihang): `jit.save` doesn't save optimizer state 
            load_result = _load_state_dict_from_save_inference_model(path,
                                                                     config)
        else:
            # load state dict by `io.save_params/persistables` save format
            # TODO(chenweihang): [ Now only supports loading parameters seperately ]
            # If users save all parameters as one file, the [ variable.name -> variable ]
            # mapping info will lost, so users need to give variable list, but users build 
            # variable list in dygraph mode is difficult, we recommend users to use
            # paddle.static.load_program_state in this case
            load_result = _load_state_dict_from_save_params(path)
    else:
        raise ValueError(
            "Unsupported path format, now only supports file or directory.")

    return load_result
