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

import collections
import copyreg
import os
import pickle
import sys
import threading
import warnings
from collections.abc import Iterable

import numpy as np

import paddle

# deprecated module import
from paddle import base
from paddle.base import core
from paddle.base.framework import (
    EagerParamBase,
    Program,
    Variable,
    _create_tensor,
    _current_expected_place,
    _current_expected_place_,
    _dygraph_tracer,
    in_dygraph_mode,
    in_pir_mode,
)

from .io_utils import (
    _is_file_path,
    _is_memory_buffer,
    _legacy_static_save,
    _open_file_buffer,
    _pack_loaded_dict,
    _pickle_loads_mac,
    _unpack_saved_dict,
)

__all__ = []
async_save_queue = []


def clear_async_save_task_queue():
    '''
    wait until all async save task to be done.
    '''
    while len(async_save_queue) > 0:
        task = async_save_queue.pop()
        if task and task.is_alive():
            task.join()


def async_save(obj, path, protocol=4, sync_other_task=False, **configs):
    '''
    async version of paddle.save.
    Note:
        currently only support dygraph mode.
    Note:
        any argument passed through configs will be overridden by default setting.
    Args:
        obj(Object) : The object to be saved.
        path(str|BytesIO) : The path/buffer of the object to be saved.
          If saved in the current directory, the input path string will be used as the file name.
        protocol(int, optional): The protocol version of pickle module must be greater than 1 and less than 5.
                                 Default: 4
        sync_other_task(bool) : Determine whether to wait other async save task to be finished before this one be put in queue.
        **configs(dict, optional): compatible argument to paddle.save, but will be overridden by default setting.
    Examples:
        .. code-block:: python
            :name: code-example-1

            import paddle
            emb = paddle.nn.Embedding(10, 10)
            layer_state_dict = emb.state_dict()

            # call paddle.async_save with the same style of paddle.save
            paddle.async_save(layer_state_dict, "emb.pdparams")
            for i in range(10):
                # do some calculations here
            # wait if any async_save task has not been done
            paddle.clear_async_task_queue()
    '''
    if not in_dygraph_mode():
        raise ValueError(
            "async_save currently is not supported in static mode."
        )
    if len(configs) > 0:
        warnings.warn(
            "configs are not supported in async mode, will be overridden by default settings."
        )

    # TODO: make this part async
    def move_state_dict_to_cpu(sd):
        for k, v in sd.items():
            if isinstance(v, dict):
                move_state_dict_to_cpu(v)
            elif isinstance(v, core.eager.Tensor):
                sd[k] = v.pin_memory() if core.is_compiled_with_cuda() else v

    if isinstance(obj, dict):
        move_state_dict_to_cpu(obj)
    elif isinstance(obj, core.eager.Tensor):
        obj = obj.pin_memory() if core.is_compiled_with_cuda() else obj
    else:
        # other types are currently not supported
        raise TypeError(
            f"currently async_save does not support this type: {type(obj)}"
        )
    if sync_other_task:
        clear_async_save_task_queue()
    t = threading.Thread(target=save, args=(obj, path, protocol))
    t.start()
    async_save_queue.append(t)


def _build_saved_state_dict(state_dict):
    save_dict = {}
    name_table = {}
    for key, value in state_dict.items():
        if isinstance(value, (Variable, core.eager.Tensor)):
            if value.type == core.VarDesc.VarType.VOCAB:
                save_dict[key] = value.value().get_map_tensor()
            else:
                if not value.value().get_tensor()._is_initialized():
                    raise ValueError(
                        "The saved tensor is not initialized. If you used group sharded, please use save_group_sharded_model."
                    )
                if value.is_dense() and value.place.is_custom_place():
                    value = paddle._C_ops.npu_identity(value, -1)
                save_dict[key] = np.array(value.cpu())
            name_table[key] = value.name
        else:
            save_dict[key] = value
    save_dict["StructuredToParameterName@@"] = name_table

    return save_dict


def _load_state_dict_from_save_inference_model(model_path, config):
    # 1. load program desc & construct _ProgramHolder
    # TODO(GGBond8488):From a long-term perspective, it is inappropriate for the framework to
    # rely on jit. It is necessary to migrate the dependency from jit to the framework in the future
    if in_pir_mode():
        from paddle.jit.pir_translated_layer import (
            _construct_params_and_buffers,
            _construct_program_holders,
        )

        programs = _construct_program_holders(model_path, config.model_filename)

    else:
        from paddle.jit.translated_layer import (
            _construct_params_and_buffers,
            _construct_program_holders,
        )

        programs = _construct_program_holders(model_path, config.model_filename)

    # 2. load layer parameters & buffers
    with base.dygraph.guard():
        persistable_var_dict = _construct_params_and_buffers(
            model_path, programs, config.params_filename
        )

        # 3. construct state_dict
        load_param_dict = {}
        for var_name in persistable_var_dict:
            tmp_var = persistable_var_dict[var_name]
            if tmp_var.is_dense() and tmp_var.place.is_custom_place():
                load_param_dict[var_name] = np.array(
                    paddle._C_ops.npu_identity(tmp_var, -1).cpu()
                )
            else:
                load_param_dict[var_name] = np.array(tmp_var.cpu())

        # if *.info exists, we can recover structured_name
        var_info_filename = str(config.params_filename) + ".info"
        var_info_path = os.path.join(model_path, var_info_filename)
        if os.path.exists(var_info_path):
            with open(var_info_path, 'rb') as f:
                extra_var_info = pickle.load(f)
            structured_para_dict = {}
            for var_name in load_param_dict:
                structured_name = extra_var_info[var_name].get(
                    'structured_name', None
                )
                assert structured_name is not None, (
                    "Cannot find saved variable (%s)'s structured name in saved model."
                    % var_name
                )
                structured_para_dict[structured_name] = load_param_dict[
                    var_name
                ]
            load_param_dict = structured_para_dict

    return load_param_dict


def _load_state_dict_from_save_params(model_path):
    # Try to load all the files in the directory in Tensor format,
    # the file name is used as the name of Tensor
    load_var_list = []

    # 1. load file names
    var_name_list = []
    for root, _, files in os.walk(model_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            tmp_var_name = os.path.relpath(file_path, model_path)
            var_name = tmp_var_name.replace("\\", "/")
            var_name_list.append(var_name)

    # 2. create and load Tensor
    with base.dygraph.guard():
        for name in var_name_list:
            new_var = _create_tensor(name=name, persistable=True)
            _dygraph_tracer().trace_op(
                type='load',
                inputs={},
                outputs={'Out': new_var},
                attrs={'file_path': os.path.join(model_path, name)},
            )
            load_var_list.append(new_var)

    # 3. construct state_dict
    load_param_dict = {}
    for var in load_var_list:
        if var.is_dense() and var.place.is_custom_place():
            var = paddle._C_ops.npu_identity(var, -1)
        load_param_dict[var.name] = np.array(var.cpu())

    return load_param_dict


# NOTE(chenweihang): [ Handling of use cases of API paddle.load ]
# `paddle.load` may be used to load saved results of:
# 1. Expected cases:
#   - need [full filename] when loading
#       - paddle.save
#       - paddle.static.save
#   - need [prefix] when loading [compatible for paddle 2.x]
#       - paddle.jit.save
#       - paddle.static.save_inference_model
#   - need [directory] when loading [compatible for paddle 1.x]
#       - paddle.base.io.save_inference_model
#       - paddle.base.io.save_params/save_persistable
# 2. Error cases:
#   - no error case
def _build_load_path_and_config(path, config):
    # NOTE(chenweihang): If both [prefix save format] and [directory save format] exist,
    # raise error, avoid confusing behavior
    # TODO(GGBond8488):From a long-term perspective, it is inappropriate for the framework to
    # rely on jit. It is necessary to migrate the dependency from jit to the framework in the future
    from paddle.jit.pir_translated_layer import (
        PIR_INFER_MODEL_SUFFIX,
    )
    from paddle.jit.translated_layer import (
        INFER_MODEL_SUFFIX,
        INFER_PARAMS_SUFFIX,
    )

    if in_pir_mode():
        prefix_format_path = path + PIR_INFER_MODEL_SUFFIX
    else:
        prefix_format_path = path + INFER_MODEL_SUFFIX
    prefix_format_exist = os.path.exists(prefix_format_path)
    directory_format_exist = os.path.isdir(path)
    if prefix_format_exist and directory_format_exist:
        raise ValueError(
            f"The {path}.pdmodel and {path} directory exist at the same time, "
            "don't know which one to load, please make sure that the specified target "
            "of ``path`` is unique."
        )
    elif not prefix_format_exist and not directory_format_exist:
        error_msg = "The ``path`` (%s) to load model not exists."
        # if current path is a prefix, and the path.pdparams or path.pdopt
        # is exist, users may want use `paddle.load` load the result of
        # `base.save_dygraph`, we raise error here for users
        params_file_path = path + ".pdparams"
        opti_file_path = path + ".pdopt"
        if os.path.exists(params_file_path) or os.path.exists(opti_file_path):
            error_msg += (
                "please specify the full file name, not just the file name prefix. For "
                "example, it should be written as `paddle.load('model.pdparams')` instead of "
                "`paddle.load('model')`."
            )
        raise ValueError(error_msg % path)
    else:
        if prefix_format_exist:
            file_prefix = os.path.basename(path)
            model_path = os.path.dirname(path)
            if config.model_filename is not None:
                warnings.warn(
                    "When loading the result saved with the "
                    "specified file prefix, the ``model_filename`` config does "
                    "not take effect."
                )
            if in_pir_mode():
                config.model_filename = file_prefix + PIR_INFER_MODEL_SUFFIX
            else:
                config.model_filename = file_prefix + INFER_MODEL_SUFFIX
            if config.params_filename is not None:
                warnings.warn(
                    "When loading the result saved with the "
                    "specified file prefix, the ``params_filename`` config does "
                    "not take effect."
                )
            config.params_filename = file_prefix + INFER_PARAMS_SUFFIX
        else:
            # Compatible with the old save_inference_model format
            model_path = path

    return model_path, config


def _parse_load_config(configs):
    supported_configs = [
        'model_filename',
        'params_filename',
        'keep_name_table',
        'return_numpy',
    ]

    # input check
    for key in configs:
        if key not in supported_configs:
            raise ValueError(
                "The additional config (%s) of `paddle.load` is not supported."
                % key
            )

    # construct inner config
    # TODO(GGBond8488):From a long-term perspective, it is inappropriate for the framework to
    # rely on jit. It is necessary to migrate the dependency from jit to the framework in the future
    from paddle.jit.api import _SaveLoadConfig

    inner_config = _SaveLoadConfig()
    inner_config.model_filename = configs.get('model_filename', None)
    inner_config.params_filename = configs.get('params_filename', None)
    inner_config.keep_name_table = configs.get('keep_name_table', None)
    inner_config.return_numpy = configs.get('return_numpy', False)

    return inner_config


def _parse_save_config(configs):
    supported_configs = ['use_binary_format', 'pickle_protocol']

    # input check
    for key in configs:
        if key not in supported_configs:
            raise ValueError(
                "The additional config (%s) of `paddle.save` is not supported."
                % key
            )

    # construct inner config
    # TODO(GGBond8488):From a long-term perspective, it is inappropriate for the framework to
    # rely on jit. It is necessary to migrate the dependency from jit to the framework in the future
    from paddle.jit.api import _SaveLoadConfig

    inner_config = _SaveLoadConfig()
    inner_config.use_binary_format = configs.get('use_binary_format', False)
    inner_config.pickle_protocol = configs.get('pickle_protocol', None)

    return inner_config


def _pickle_save(obj, f, protocol):
    # TODO(weixin):add support for BytesIO.
    if not isinstance(protocol, int):
        raise ValueError(
            f"The 'protocol' MUST be `int`, but received {type(protocol)}"
        )

    if protocol < 2 or protocol > 4:
        raise ValueError(
            f"Expected 1<'protocol'<5, but received protocol={protocol}"
        )

    def reduce_varbase(self):
        if self.is_dense() and self.place.is_custom_place():
            data = np.array(paddle._C_ops.npu_identity(self, -1).cpu())
        else:
            data = np.array(self.cpu())
        name = self.name

        return (tuple, ((name, data),))

    def reduce_LoDTensor(self):
        p = core.Place()
        p.set_place(paddle.CPUPlace())
        if self._place().is_custom_place():
            data = np.array(paddle._C_ops.npu_identity(self, -1)._copy(p))
        else:
            data = np.array(self._copy(p))

        return (eval, ('data', {'data': data}))

    def reduce_Layer(self):
        raise ValueError(
            "paddle do not support saving `paddle.nn.Layer` object."
        )

    dispatch_table_layer = {}

    def create_layer_dispatch_table(layer):
        dispatch_table_layer[layer.__class__] = reduce_Layer
        return layer

    _parse_every_object(
        obj,
        lambda v: isinstance(v, paddle.nn.Layer),
        create_layer_dispatch_table,
    )

    def add_dispatch_table():
        # This is not a good method, because the pickle module has been modified.
        pickle.dispatch_table[core.eager.Tensor] = reduce_varbase
        pickle.dispatch_table[EagerParamBase] = reduce_varbase
        pickle.dispatch_table[core.LoDTensor] = reduce_LoDTensor
        pickle.dispatch_table.update(dispatch_table_layer)

    def pop_dispatch_table():
        pickle.dispatch_table.pop(core.LoDTensor)
        pickle.dispatch_table.pop(core.eager.Tensor)
        pickle.dispatch_table.pop(EagerParamBase)
        for k in dispatch_table_layer:
            pickle.dispatch_table.pop(k)

    # When value of dict is lager than 4GB ,there is a Bug on 'MAC python3'
    if sys.platform == 'darwin' and sys.version_info.major == 3:
        add_dispatch_table()
        pickle_bytes = pickle.dumps(obj)
        pop_dispatch_table()

        max_bytes = 2**30
        for i in range(0, len(pickle_bytes), max_bytes):
            f.write(pickle_bytes[i : i + max_bytes])
    else:
        pickler = pickle.Pickler(f, protocol)
        pickler.dispatch_table = copyreg.dispatch_table.copy()

        pickler.dispatch_table[core.LoDTensor] = reduce_LoDTensor
        pickler.dispatch_table[core.eager.Tensor] = reduce_varbase
        pickler.dispatch_table[EagerParamBase] = reduce_varbase
        pickler.dispatch_table.update(dispatch_table_layer)
        pickler.dump(obj)


def _contain_x(obj, condition_func):
    if isinstance(obj, core.SelectedRows):
        raise NotImplementedError(
            "`paddle.save` do not support saving 'SelectedRows'."
        )

    if condition_func(obj):
        return True
    elif type(obj) in (dict, collections.OrderedDict, list, tuple):
        if type(obj) in (dict, collections.OrderedDict):
            keys = list(obj.keys())
        else:
            keys = range(len(obj))
        flag = False
        for key in keys:
            flag |= _contain_x(obj[key], condition_func)
            if flag:
                return True
        return flag
    else:
        return False


def _is_state_dict(obj):
    if isinstance(obj, dict):

        def condition(obj):
            return isinstance(
                obj,
                (
                    paddle.nn.Layer,
                    Program,
                    core.eager.Tensor,
                    core.LoDTensor,
                    core.SelectedRows,
                ),
            )

        # If the value of a dict is a core.Tensor/LoDTensor or a dict
        # that does not contain a paddle type(Layer, Program, Tensor, LoDTensor, SelectedRows),
        # the dict is considered to be a state_ dict.
        for key, value in obj.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    if _contain_x(v, condition):
                        return False
            elif not isinstance(value, (core.eager.Tensor, core.LoDTensor)):
                return False
        return True

    return False


def _transformed_from_varbase(obj):
    # In paddle2.1 version, Tensor is saved as tuple(tensor.name, tensor.numpy()).
    # When executing paddle.load, use this function to determine whether to restore to Tensor/LoDTensor.
    if isinstance(obj, tuple) and len(obj) == 2:
        name_types = str
        if isinstance(obj[0], name_types) and isinstance(obj[1], np.ndarray):
            return True
    return False


def _transformed_from_lodtensor(obj):
    # In paddle2.1 version, LoDTensor is saved as np.array(tensor).
    # When executing paddle.load, use this function to determine whether to restore to Tensor/LoDTensor.
    if isinstance(obj, np.ndarray):
        return True
    return False


def _to_LodTensor(ndarray):
    if not isinstance(ndarray, np.ndarray):
        raise TypeError(
            f'Type of `ndarray` should be numpy.ndarray, but received {type(ndarray)}.'
        )
    t = core.LoDTensor()
    place = _current_expected_place_()
    t.set(ndarray, place)
    return t


def _tuple_to_tensor(obj, return_numpy):
    if return_numpy:
        return obj[1]
    if in_dygraph_mode():
        t = paddle.to_tensor(obj[1])
        # This function does modify the name of return value.
        # Loading the same variable multiple times may cause the same name.
        t.name = obj[0]
        return t
    else:
        return _to_LodTensor(obj[1])


def _ndarray_to_tensor(obj, return_numpy):
    if return_numpy:
        return obj
    if in_dygraph_mode():
        return paddle.to_tensor(obj)
    else:
        return _to_LodTensor(obj)


def _lod_tensor2varbase(tensor):
    return_var = _create_tensor()
    return_var.value().get_tensor().set(tensor, _current_expected_place())
    return return_var


def _parse_every_object(obj, condition_func, convert_func):
    if condition_func(obj):
        return convert_func(obj)
    elif type(obj) in (dict, collections.OrderedDict, list):
        if type(obj) == list:
            keys = range(len(obj))
        else:
            keys = list(obj.keys())
        for key in keys:
            if condition_func(obj[key]):
                obj[key] = convert_func(obj[key])
            else:
                obj[key] = _parse_every_object(
                    obj[key], condition_func, convert_func
                )
        return obj
    elif type(obj) == tuple:
        return tuple(
            _parse_every_object(list(obj), condition_func, convert_func)
        )
    elif type(obj) == set:
        return set(_parse_every_object(list(obj), condition_func, convert_func))
    else:
        if isinstance(obj, Iterable) and not isinstance(
            obj,
            (str, np.ndarray, core.eager.Tensor, core.LoDTensor),
        ):
            raise NotImplementedError(
                f"The iterable objects supported are tuple, list, dict, OrderedDict, string. But received {type(obj)}."
            )
        return obj


def _parse_load_result(obj, return_numpy):
    def is_layer(obj):
        return isinstance(obj, paddle.nn.Layer)

    def parse_layer(obj):
        temp_dict = _parse_load_result(obj.__dict__, False)
        obj.__dict__.update(temp_dict)
        return obj

    if _contain_x(obj, is_layer):
        if not in_dygraph_mode():
            raise ValueError(
                "Layer can only be loaded in dynamic graph mode, but now in static graph mode."
            )

        _parse_every_object(obj, is_layer, parse_layer)

    def tuple_to_tensor(obj):
        return _tuple_to_tensor(obj, return_numpy=return_numpy)

    def ndarray_to_tensor(obj):
        return _ndarray_to_tensor(obj, return_numpy=return_numpy)

    # tuple(name, ndarray) was converted from varbase of paddle2.1,
    # and all tuple(name, ndarray) are converted to tensor.
    if _contain_x(obj, _transformed_from_varbase):
        return _parse_every_object(
            obj, _transformed_from_varbase, tuple_to_tensor
        )
    # If there is no tuple(name, ndarray), it is considered to be saved by paddle2.0
    # or converted from LoDTensor, and all ndarrays are converted to tensor.
    else:
        return _parse_every_object(
            obj, _transformed_from_lodtensor, ndarray_to_tensor
        )


def _save_lod_tensor(tensor, file_name):
    if not tensor._is_initialized():
        raise ValueError(
            "The saved tensor is not initialized. If you used group sharded, please use save_group_sharded_model firstly."
        )
    if _is_file_path(file_name):
        _seek = core.save_lod_tensor(tensor, file_name)
        # '_seek' is the end position of this tensor in the file.

    elif _is_memory_buffer(file_name):
        tensor_bytes = core.save_lod_tensor_to_memory(tensor)

        with _open_file_buffer(file_name, 'wb') as f:
            f.write(tensor_bytes)
            _seek = f.tell()

    else:
        raise NotImplementedError(
            f'Only supports saving objects to file or BytesIO, but received {type(file_name)}'
        )
    return _seek


def _load_lod_tensor(file_name):
    temp_t = paddle.base.core.LoDTensor()
    if _is_file_path(file_name):
        # '_seek' is the end position of this tensor in the file.
        _seek = paddle.base.core.load_lod_tensor(temp_t, file_name)

    elif _is_memory_buffer(file_name):
        with _open_file_buffer(file_name, 'rb') as f:
            tensor_bytes = f.read()
            paddle.base.core.load_lod_tensor_from_memory(temp_t, tensor_bytes)
            _seek = f.tell()

    else:
        raise NotImplementedError(
            f'Only supports load objects from file or BytesIO, but received {type(file_name)}'
        )

    return temp_t, _seek


def _save_selected_rows(selected_rows, file_name):
    if not selected_rows.get_tensor()._is_initialized():
        raise ValueError("The saved tensor is not initialized.")
    if _is_file_path(file_name):
        # '_seek' is the end position of this SelectedRows in the file.
        _seek = core.save_selected_rows(selected_rows, file_name)

    elif _is_memory_buffer(file_name):
        selected_rows_bytes = core.save_selected_rows_to_memory(selected_rows)
        with _open_file_buffer(file_name, 'wb') as f:
            f.write(selected_rows_bytes)
            _seek = f.tell()
    else:
        raise NotImplementedError(
            f'Only supports saving objects to file or BytesIO, but received {type(file_name)}'
        )
    return _seek


def _load_selected_rows(file_name):
    temp_sr = core.SelectedRows()
    if _is_file_path(file_name):
        # '_seek' is the end position of this SelectedRows in the file.
        _seek = core.load_selected_rows(temp_sr, file_name)

    elif _is_memory_buffer(file_name):
        with _open_file_buffer(file_name, 'rb') as f:
            selected_rows_bytes = f.read()
            paddle.base.core.load_selected_rows_from_memory(
                temp_sr, selected_rows_bytes
            )
        _seek = f.tell()

    else:
        raise NotImplementedError(
            f'Only supports load objects from file or BytesIO, but received {type(file_name)}'
        )

    return temp_sr, _seek


def _save_binary_var(obj, path):
    if isinstance(obj, core.LoDTensor):
        _save_lod_tensor(obj, path)
    elif isinstance(obj, core.SelectedRows):
        _save_selected_rows(obj, path)
    elif isinstance(obj, core.eager.Tensor):
        _save_lod_tensor(obj.value().get_tensor(), path)
    else:
        # Since the concept of 'Tensor' is only exposed to users, the error message can only contain tensor instead of 'LoDTensor' or 'SelectedRows'
        raise NotImplementedError(
            f"When use_binary_format = True, `paddle.save`  expected Tensor, but received {type(obj)}."
        )


def save(obj, path, protocol=4, **configs):
    '''
    Save an object to the specified path.

    Note:
        Now supports saving ``state_dict`` of Layer/Optimizer, Tensor and nested structure containing Tensor, Program.

    Note:
        Different from ``paddle.jit.save``, since the save result of ``paddle.save`` is a single file,
        there is no need to distinguish multiple saved files by adding a suffix. The argument ``path``
        of ``paddle.save`` will be directly used as the saved file name instead of a prefix.
        In order to unify the saved file name format, we recommend using the paddle standard suffix:
        1. for ``Layer.state_dict`` , recommend to use ``.pdparams`` ;
        2. for ``Optimizer.state_dict`` , recommend to use ``.pdopt`` .
        For specific examples, please refer to API code examples.

    Args:
        obj(Object) : The object to be saved.
        path(str|BytesIO) : The path/buffer of the object to be saved.
          If saved in the current directory, the input path string will be used as the file name.
        protocol(int, optional): The protocol version of pickle module must be greater than 1 and less than 5.
                                 Default: 4
        **configs(dict, optional): optional keyword arguments. The following options are currently supported:
          use_binary_format(bool): When the saved object is static graph variable, you can specify ``use_binary_for_var``.
          If True, save the file in the c++ binary format when saving a single static graph variable; otherwise, save it in pickle format.
          Default: False

    Returns:
        None

    Examples:
        .. code-block:: python
            :name: code-example-1

            >>> # example 1: dynamic graph
            >>> import paddle
            >>> emb = paddle.nn.Embedding(10, 10)
            >>> layer_state_dict = emb.state_dict()

            >>> # save state_dict of emb
            >>> paddle.save(layer_state_dict, "emb.pdparams")

            >>> scheduler = paddle.optimizer.lr.NoamDecay(
            ...     d_model=0.01, warmup_steps=100, verbose=True)
            >>> adam = paddle.optimizer.Adam(
            ...     learning_rate=scheduler,
            ...     parameters=emb.parameters())
            >>> opt_state_dict = adam.state_dict()

            >>> # save state_dict of optimizer
            >>> paddle.save(opt_state_dict, "adam.pdopt")
            >>> # save weight of emb
            >>> paddle.save(emb.weight, "emb.weight.pdtensor")

        .. code-block:: python
            :name: code-example-2

            >>> # example 2: Save multiple state_dict at the same time
            >>> import paddle
            >>> from paddle import nn
            >>> from paddle.optimizer import Adam

            >>> layer = paddle.nn.Linear(3, 4)
            >>> adam = Adam(learning_rate=0.001, parameters=layer.parameters())
            >>> obj = {'model': layer.state_dict(), 'opt': adam.state_dict(), 'epoch': 100}
            >>> path = 'example/model.pdparams'
            >>> paddle.save(obj, path)

        .. code-block:: python
            :name: code-example-3

            >>> # example 3: static graph
            >>> import paddle
            >>> import paddle.static as static

            >>> paddle.enable_static()

            >>> # create network
            >>> x = paddle.static.data(name="x", shape=[None, 224], dtype='float32')
            >>> z = paddle.static.nn.fc(x, 10)

            >>> place = paddle.CPUPlace()
            >>> exe = paddle.static.Executor(place)
            >>> exe.run(paddle.static.default_startup_program())
            >>> prog = paddle.static.default_main_program()
            >>> for var in prog.list_vars():
            ...     if list(var.shape) == [224, 10]:
            ...         tensor = var.get_value()
            ...         break

            >>> # save/load tensor
            >>> path_tensor = 'temp/tensor.pdtensor'
            >>> paddle.save(tensor, path_tensor)

            >>> # save/load state_dict
            >>> path_state_dict = 'temp/model.pdparams'
            >>> paddle.save(prog.state_dict("param"), path_tensor)

        .. code-block:: python
            :name: code-example-4

            >>> # example 4: save program
            >>> import paddle

            >>> paddle.enable_static()

            >>> data = paddle.static.data(
            ...     name='x_static_save', shape=(None, 224), dtype='float32')
            >>> y_static = z = paddle.static.nn.fc(data, 10)
            >>> main_program = paddle.static.default_main_program()
            >>> path = "example/main_program.pdmodel"
            >>> paddle.save(main_program, path)

        .. code-block:: python
            :name: code-example-5

            >>> # example 5: save object to memory
            >>> from io import BytesIO
            >>> import paddle
            >>> from paddle.nn import Linear
            >>> paddle.disable_static()

            >>> linear = Linear(5, 10)
            >>> state_dict = linear.state_dict()
            >>> byio = BytesIO()
            >>> paddle.save(state_dict, byio)
            >>> paddle.seed(2023)
            >>> tensor = paddle.randn([2, 3], dtype='float32')
            >>> paddle.save(tensor, byio)

    '''
    if _is_file_path(path):
        # 1. input check
        filename = os.path.basename(path)
        if filename == "":
            raise ValueError(
                "The input path MUST be format of dirname/filename "
                "[dirname\\filename in Windows system], but received "
                "filename is empty string."
            )

        # 2. save object
        dirname = os.path.dirname(path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
    elif not _is_memory_buffer(path):
        raise ValueError(
            f"only supports saving objects to file and `BytesIO`, but got {type(path)}"
        )

    config = _parse_save_config(configs)

    if not isinstance(config.use_binary_format, bool):
        raise TypeError(
            f"Type of `use_binary_format` should be bool, but received {type(config.use_binary_format)}."
        )

    if config.use_binary_format:
        _save_binary_var(obj, path)
    else:
        # `protocol` need to be used, `pickle_protocol` is a deprecated arg.
        if config.pickle_protocol is not None:
            protocol = config.pickle_protocol
            warnings.warn(
                "'pickle_protocol' is a deprecated argument. Please use 'protocol' instead."
            )

        if isinstance(obj, paddle.static.Program):
            if in_pir_mode():
                paddle.core.serialize_pir_program(
                    obj, path, 1, True, False, True
                )
            else:
                obj.desc.flush()
                with _open_file_buffer(path, "wb") as f:
                    f.write(obj.desc.serialize_to_string())

        elif _is_state_dict(obj):
            if in_dygraph_mode():
                _legacy_save(obj, path, protocol)
            else:
                _legacy_static_save(obj, path, protocol)
        else:
            with _open_file_buffer(path, 'wb') as f:
                _pickle_save(obj, f, protocol)


def _legacy_save(obj, path, protocol=2):
    # 1. input check
    if not isinstance(obj, dict):
        raise NotImplementedError(
            "Now only supports save state_dict of Layer or Optimizer, "
            "expect dict, but received %s." % type(obj)
        )

    if len(obj) == 0:
        warnings.warn("The input state dict is empty, no need to save.")

    if not isinstance(protocol, int):
        raise ValueError(
            f"The 'protocol' MUST be `int`, but received {type(protocol)}"
        )

    if protocol < 2 or protocol > 4:
        raise ValueError(
            f"Expected 1<'protocol'<5, but received protocol={protocol}"
        )

    if _is_file_path(path):
        filename = os.path.basename(path)
        if filename == "":
            raise ValueError(
                "The input path MUST be format of dirname/filename "
                "[dirname\\filename in Windows system], but received "
                "filename is empty string."
            )
        # 2. save object
        dirname = os.path.dirname(path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

    if isinstance(obj, dict):
        saved_obj = _build_saved_state_dict(obj)

    saved_obj = _unpack_saved_dict(saved_obj, protocol)

    # When value of dict is lager than 4GB ,there is a Bug on 'MAC python3'
    if (
        _is_file_path(path)
        and sys.platform == 'darwin'
        and sys.version_info.major == 3
    ):
        pickle_bytes = pickle.dumps(saved_obj, protocol=protocol)
        with open(path, 'wb') as f:
            max_bytes = 2**30
            for i in range(0, len(pickle_bytes), max_bytes):
                f.write(pickle_bytes[i : i + max_bytes])
    else:
        with _open_file_buffer(path, 'wb') as f:
            pickle.dump(saved_obj, f, protocol=protocol)


def load(path, **configs):
    '''
    Load an object can be used in paddle from specified path.

    Note:
        Now supports loading ``state_dict`` of Layer/Optimizer, Tensor and nested structure containing Tensor, Program.

    Note:
        In order to use the model parameters saved by paddle more efficiently,
        ``paddle.load`` supports loading ``state_dict`` of Layer from the result of
        other save APIs except ``paddle.save`` , but the argument ``path`` format is
        different:
        1. loading from ``paddle.static.save`` or ``paddle.Model().save(training=True)`` ,
        ``path`` needs to be a complete file name, such as ``model.pdparams`` or
        ``model.pdopt`` ;
        2. loading from ``paddle.jit.save`` or ``paddle.static.save_inference_model``
        or ``paddle.Model().save(training=False)`` , ``path`` need to be a file prefix,
        such as ``model/mnist``, and ``paddle.load`` will get information from
        ``mnist.pdmodel`` and ``mnist.pdiparams`` ;
        3. loading from paddle 1.x APIs ``paddle.base.io.save_inference_model`` or
        ``paddle.base.io.save_params/save_persistables`` , ``path`` need to be a
        directory, such as ``model`` and model is a directory.

    Note:
        If you load ``state_dict`` from the saved result of static graph mode API such as
        ``paddle.static.save`` or ``paddle.static.save_inference_model`` ,
        the structured variable name in dynamic mode will cannot be restored.
        You need to set the argument ``use_structured_name=False`` when using
        ``Layer.set_state_dict`` later.

    Args:
        path(str|BytesIO) : The path/buffer to load the target object. Generally, the path is the target
            file path. When loading state_dict from the saved result of the API used to save
            the inference model, the path may be a file prefix or directory.
        **configs (dict, optional): other load configuration options for compatibility. We do not
            recommend using these configurations, they may be removed in the future. If not necessary,
            DO NOT use them. Default None.
            The following options are currently supported:
            (1) model_filename (str): The inference model file name of the paddle 1.x
            ``save_inference_model`` save format. Default file name is :code:`__model__` .
            (2) params_filename (str): The persistable variables file name of the paddle 1.x
            ``save_inference_model`` save format. No default file name, save variables separately
            by default.
            (3) return_numpy(bool): If specified as True, return tensor as numpy.ndarray, otherwise return tensor as paddle.Tensor.
            Default False.

    Returns:
        Object(Object): a target object can be used in paddle

    Examples:
        .. code-block:: python
            :name: code-example-1

            >>> # example 1: dynamic graph
            >>> import paddle
            >>> emb = paddle.nn.Embedding(10, 10)
            >>> layer_state_dict = emb.state_dict()

            >>> # save state_dict of emb
            >>> paddle.save(layer_state_dict, "emb.pdparams")

            >>> scheduler = paddle.optimizer.lr.NoamDecay(
            ...     d_model=0.01, warmup_steps=100, verbose=True)
            >>> adam = paddle.optimizer.Adam(
            ...     learning_rate=scheduler,
            ...     parameters=emb.parameters())
            >>> opt_state_dict = adam.state_dict()

            >>> # save state_dict of optimizer
            >>> paddle.save(opt_state_dict, "adam.pdopt")
            >>> # save weight of emb
            >>> paddle.save(emb.weight, "emb.weight.pdtensor")

            >>> # load state_dict of emb
            >>> load_layer_state_dict = paddle.load("emb.pdparams")
            >>> # load state_dict of optimizer
            >>> load_opt_state_dict = paddle.load("adam.pdopt")
            >>> # load weight of emb
            >>> load_weight = paddle.load("emb.weight.pdtensor")

        .. code-block:: python
            :name: code-example-2

            >>> # example 2: Load multiple state_dict at the same time
            >>> import paddle
            >>> from paddle import nn
            >>> from paddle.optimizer import Adam

            >>> layer = paddle.nn.Linear(3, 4)
            >>> adam = Adam(learning_rate=0.001, parameters=layer.parameters())
            >>> obj = {'model': layer.state_dict(), 'opt': adam.state_dict(), 'epoch': 100}
            >>> path = 'example/model.pdparams'
            >>> paddle.save(obj, path)
            >>> obj_load = paddle.load(path)

        .. code-block:: python
            :name: code-example-3

            >>> # example 3: static graph
            >>> import paddle
            >>> import paddle.static as static

            >>> paddle.enable_static()

            >>> # create network
            >>> x = paddle.static.data(name="x", shape=[None, 224], dtype='float32')
            >>> z = paddle.static.nn.fc(x, 10)

            >>> place = paddle.CPUPlace()
            >>> exe = paddle.static.Executor(place)
            >>> exe.run(paddle.static.default_startup_program())
            >>> prog = paddle.static.default_main_program()
            >>> for var in prog.list_vars():
            ...     if list(var.shape) == [224, 10]:
            ...         tensor = var.get_value()
            ...         break

            >>> # save/load tensor
            >>> path_tensor = 'temp/tensor.pdtensor'
            >>> paddle.save(tensor, path_tensor)
            >>> load_tensor = paddle.load(path_tensor)

            >>> # save/load state_dict
            >>> path_state_dict = 'temp/model.pdparams'
            >>> paddle.save(prog.state_dict("param"), path_tensor)
            >>> load_state_dict = paddle.load(path_tensor)

        .. code-block:: python
            :name: code-example-4

            >>> # example 4: load program
            >>> import paddle

            >>> paddle.enable_static()

            >>> data = paddle.static.data(
            ...     name='x_static_save', shape=(None, 224), dtype='float32')
            >>> y_static = z = paddle.static.nn.fc(data, 10)
            >>> main_program = paddle.static.default_main_program()
            >>> path = "example/main_program.pdmodel"
            >>> paddle.save(main_program, path)
            >>> load_main = paddle.load(path)

        .. code-block:: python
            :name: code-example-5

            >>> # example 5: save object to memory
            >>> from io import BytesIO
            >>> import paddle
            >>> from paddle.nn import Linear
            >>> paddle.disable_static()

            >>> linear = Linear(5, 10)
            >>> state_dict = linear.state_dict()
            >>> byio = BytesIO()
            >>> paddle.save(state_dict, byio)
            >>> paddle.seed(2023)
            >>> tensor = paddle.randn([2, 3], dtype='float32')
            >>> paddle.save(tensor, byio)
            >>> byio.seek(0)
            >>> # load state_dict
            >>> dict_load = paddle.load(byio)

    '''

    if _is_memory_buffer(path) or os.path.isfile(path):
        config = _parse_load_config(configs)
        exception_type = pickle.UnpicklingError
        try:
            with _open_file_buffer(path, 'rb') as f:
                # When value of dict is lager than 4GB ,there is a Bug on 'MAC python3'
                if (
                    _is_file_path(path)
                    and sys.platform == 'darwin'
                    and sys.version_info.major == 3
                ):
                    load_result = _pickle_loads_mac(path, f)
                else:
                    load_result = pickle.load(f, encoding='latin1')

                # TODO(weixin):If `obj` is any object, the judgment condition should be more precise.
                if isinstance(load_result, dict):
                    load_result = _pack_loaded_dict(load_result)
                    # paddle2.0: paddle.save/load
                    if "StructuredToParameterName@@" in load_result:
                        for key, name in load_result[
                            "StructuredToParameterName@@"
                        ].items():
                            if isinstance(load_result[key], np.ndarray):
                                load_result[key] = _ndarray_to_tensor(
                                    load_result[key], config.return_numpy
                                )
                                # default name is "generatedxxx" which is set in Tensor init, if not set
                                if not config.return_numpy and getattr(
                                    load_result[key], "name", ""
                                ):
                                    load_result[key].name = name

                        if (
                            not config.keep_name_table
                            and "StructuredToParameterName@@" in load_result
                        ):
                            del load_result["StructuredToParameterName@@"]
                    else:
                        # paddle2.1 static.save/load
                        load_result = _parse_load_result(
                            load_result, config.return_numpy
                        )

                else:
                    load_result = _parse_load_result(
                        load_result, config.return_numpy
                    )

        except exception_type as msg_pickle:
            try:
                tensor, _ = _load_selected_rows(path)
                return tensor
            except:
                try:
                    tensor, _ = _load_lod_tensor(path)
                    if config.return_numpy:
                        p = core.Place()
                        p.set_place(paddle.CPUPlace())
                        if tensor._place().is_custom_place():
                            return np.array(
                                paddle._C_ops.npu_identity(tensor, -1)._copy(p)
                            )
                        else:
                            return np.array(tensor._copy(p))
                    else:
                        if in_dygraph_mode():
                            return _lod_tensor2varbase(tensor)
                        return tensor
                except:
                    try:
                        if in_pir_mode():
                            program = paddle.static.Program()
                            paddle.core.deserialize_pir_program(
                                path, program, 1
                            )
                            return program
                        with _open_file_buffer(path, "rb") as f:
                            program_desc_str = f.read()
                            program = Program.parse_from_string(
                                program_desc_str
                            )
                            return program
                    except:
                        raise ValueError(
                            f"`paddle.load` can not parse the file:{path}."
                        )

    else:
        load_result = _legacy_load(path, **configs)

    return load_result


def _legacy_load(path, **configs):
    load_result = None
    config = _parse_load_config(configs)

    if os.path.isfile(path) or _is_memory_buffer(path):
        # we think path is file means this file is created by paddle.save
        with _open_file_buffer(path, 'rb') as f:
            load_result = pickle.load(f, encoding='latin1')
        load_result = _pack_loaded_dict(load_result)
        if (
            not config.keep_name_table
            and "StructuredToParameterName@@" in load_result
        ):
            del load_result["StructuredToParameterName@@"]
    else:
        # file prefix and directory are compatible cases
        model_path, config = _build_load_path_and_config(path, config)
        # check whether model file exists
        if config.model_filename is None:
            model_filename = '__model__'
        else:
            model_filename = config.model_filename
        model_file_path = os.path.join(model_path, model_filename)

        if os.path.exists(model_file_path):
            # Load state dict by `jit.save/io.save_inference_model` save format
            # NOTE(chenweihang): [ Compatibility of save_inference_model save format ]
            # The model saved by `save_inference_model` does not completely correspond to
            # the information required by the `state_dict` under the dygraph.
            # `save_inference_model` not save structured name, we need to remind
            # the user to configure the `use_structured_name` argument when `set_state_dict`
            # NOTE(chenweihang): `jit.save` doesn't save optimizer state
            load_result = _load_state_dict_from_save_inference_model(
                model_path, config
            )
        else:
            # load state dict by `io.save_params/persistables` save format
            # TODO(chenweihang): [ Now only supports loading parameters separately ]
            # If users save all parameters as one file, the [ variable.name -> variable ]
            # mapping info will lost, so users need to give variable list, but users build
            # variable list in dygraph mode is difficult, we recommend users to use
            # paddle.static.load_program_state in this case
            load_result = _load_state_dict_from_save_params(model_path)

    return load_result
