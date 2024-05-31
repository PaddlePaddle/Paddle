#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import errno
import inspect
import logging
import os
import pickle
import sys
import warnings

import numpy as np

import paddle
from paddle.base import (
    Program,
    Variable,
    core,
    default_main_program,
    program_guard,
    unique_name,
)
from paddle.base.executor import Executor, global_scope
from paddle.base.framework import (
    Parameter,
    dygraph_not_support,
    in_pir_mode,
    process_type_promotion,
    static_only,
)
from paddle.base.log_helper import get_logger
from paddle.framework.io_utils import (
    _clone_var_in_block_,
    _load_program_scope,
    _pack_loaded_dict,
    _pickle_loads_mac,
    _unpack_saved_dict,
    is_belong_to_optimizer,
    is_parameter,
    is_persistable,
)

from .io_utils import (
    _check_args,
    _check_vars,
    _get_valid_program,
    _normalize_path_prefix,
    _safe_load_pickle,
)
from .pir_io import (
    get_pir_parameters,
    load_pir,
    load_pir_inference_model,
    load_vars_pir,
    normalize_pir_program,
    save_pir,
    save_pir_inference_model,
    save_vars_pir,
)

__all__ = []

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)


def _clone_var_in_block(block, var):
    assert isinstance(var, Variable)
    if var.desc.type() == core.VarDesc.VarType.LOD_TENSOR:
        return block.create_var(
            name=var.name,
            shape=var.shape,
            dtype=var.dtype,
            type=var.type,
            lod_level=var.lod_level,
            persistable=True,
        )
    else:
        return block.create_var(
            name=var.name,
            shape=var.shape,
            dtype=var.dtype,
            type=var.type,
            persistable=True,
        )


def prepend_feed_ops(
    inference_program, feed_target_names, feed_holder_name='feed'
):
    if len(feed_target_names) == 0:
        return

    global_block = inference_program.global_block()
    feed_var = global_block.create_var(
        name=feed_holder_name,
        type=core.VarDesc.VarType.FEED_MINIBATCH,
        persistable=True,
    )

    for i, name in enumerate(feed_target_names):
        if not global_block.has_var(name):
            raise ValueError(
                f"The feeded_var_names[{i}]: '{name}' doesn't exist in pruned inference program. "
                f"Please check whether '{name}' is a valid feed_var name, or remove it from feeded_var_names "
                f"if '{name}' is not involved in the target_vars calculation."
            )
        out = global_block.var(name)
        global_block._prepend_op(
            type='feed',
            inputs={'X': [feed_var]},
            outputs={'Out': [out]},
            attrs={'col': i},
        )


def append_fetch_ops(
    inference_program, fetch_target_names, fetch_holder_name='fetch'
):
    global_block = inference_program.global_block()
    fetch_var = global_block.create_var(
        name=fetch_holder_name,
        type=core.VarDesc.VarType.FETCH_LIST,
        persistable=True,
    )

    for i, name in enumerate(fetch_target_names):
        global_block.append_op(
            type='fetch',
            inputs={'X': [name]},
            outputs={'Out': [fetch_var]},
            attrs={'col': i},
        )


def normalize_program(program, feed_vars, fetch_vars, **kwargs):
    """

    Normalize/Optimize a program according to feed_vars and fetch_vars.

    Args:
        program(Program): Specify a program you want to optimize.
        feed_vars(Tensor | list[Tensor]): Variables needed by inference.
        fetch_vars(Tensor | list[Tensor]): Variables returned by inference.
        kwargs: Supported keys including ``skip_prune_program``.
            - skip_prune_program(bool): whether to skip pruning program. Defaults to False.

    Returns:
        Program: Normalized/Optimized program.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.enable_static()

            >>> path_prefix = "./infer_model"

            # User defined network, here a softmax regression example
            >>> image = paddle.static.data(name='img', shape=[None, 28, 28], dtype='float32')
            >>> label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
            >>> predict = paddle.static.nn.fc(image, 10, activation='softmax')

            >>> loss = paddle.nn.functional.cross_entropy(predict, label)

            >>> exe = paddle.static.Executor(paddle.CPUPlace())
            >>> exe.run(paddle.static.default_startup_program())

            # normalize main program.
            >>> program = paddle.static.default_main_program()
            >>> normalized_program = paddle.static.normalize_program(program, [image], [predict])

    """
    if in_pir_mode():
        return normalize_pir_program(program, feed_vars, fetch_vars, **kwargs)
    if not isinstance(program, Program):
        raise TypeError(
            "program type must be `base.Program`, but received `%s`"
            % type(program)
        )
    if not isinstance(feed_vars, list):
        feed_vars = [feed_vars]
    if not all(isinstance(v, Variable) for v in feed_vars):
        raise TypeError(
            "feed_vars type must be a Variable or a list of Variable."
        )
    if not isinstance(fetch_vars, list):
        fetch_vars = [fetch_vars]
    if not all(isinstance(v, Variable) for v in fetch_vars):
        raise TypeError(
            "fetch_vars type must be a Variable or a list of Variable."
        )

    # remind users to set auc_states to 0 if auc op were found.
    for op in program.global_block().ops:
        # clear device of Op
        device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName()
        op._set_attr(device_attr_name, "")
        if op.type == 'auc':
            warnings.warn(
                "Be sure that you have set auc states to 0 before saving inference model."
            )
            break

    # fix the bug that the activation op's output as target will be pruned.
    # will affect the inference performance.
    # TODO(Superjomn) add an IR pass to remove 1-scale op.
    with program_guard(program):
        uniq_fetch_vars = []
        for i, var in enumerate(fetch_vars):
            if var.dtype != paddle.bool:
                var = paddle.scale(var, 1.0, name=f"save_infer_model/scale_{i}")
            uniq_fetch_vars.append(var)
        fetch_vars = uniq_fetch_vars

    # serialize program
    copy_program = program.clone()
    global_block = copy_program.global_block()
    remove_op_idx = []
    for i, op in enumerate(global_block.ops):
        op.desc.set_is_target(False)
        if op.type == "feed" or op.type == "fetch":
            remove_op_idx.append(i)

        if op.type == "pylayer":
            sub_blocks_ids = op._blocks_attr_ids("blocks")
            if len(sub_blocks_ids) > 1:
                # pylayer op ``blocks`` attr contains forward block id and backward block id
                backward_block_id = sub_blocks_ids[-1]
                # remove backward block
                copy_program.blocks.pop(backward_block_id)
                # update attrs ``blocks``
                reserved_blocks = []
                for block_id in sub_blocks_ids[:-1]:
                    reserved_blocks.append(copy_program.block(block_id))
                op._update_desc_attr("blocks", reserved_blocks)

    for idx in remove_op_idx[::-1]:
        global_block._remove_op(idx)
    copy_program.desc.flush()

    feed_var_names = [var.name for var in feed_vars]

    skip_prune_program = kwargs.get('skip_prune_program', False)
    if not skip_prune_program:
        copy_program = copy_program._prune_with_input(
            feeded_var_names=feed_var_names, targets=fetch_vars
        )
    copy_program = copy_program._inference_optimize(prune_read_op=True)
    fetch_var_names = [var.name for var in fetch_vars]
    prepend_feed_ops(copy_program, feed_var_names)
    append_fetch_ops(copy_program, fetch_var_names)
    copy_program.desc._set_version()
    return copy_program


@static_only
def serialize_program(feed_vars, fetch_vars, **kwargs):
    """

    Serialize default main program according to feed_vars and fetch_vars.

    Args:
        feed_vars(Tensor | list[Tensor]): Tensor needed by inference.
        fetch_vars(Tensor | list[Tensor]): Tensor returned by inference.
        kwargs: Supported keys including ``program``. Attention please, kwargs is used for backward compatibility mainly.

            - program(Program): specify a program if you don't want to use default main program.
            - legacy_format(bool): whether to save inference program in legacy format. Defaults to False.

    Returns:
        bytes: serialized program.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.enable_static()

            >>> path_prefix = "./infer_model"

            # User defined network, here a softmax regression example
            >>> image = paddle.static.data(name='img', shape=[None, 28, 28], dtype='float32')
            >>> label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
            >>> predict = paddle.static.nn.fc(image, 10, activation='softmax')

            >>> loss = paddle.nn.functional.cross_entropy(predict, label)

            >>> exe = paddle.static.Executor(paddle.CPUPlace())
            >>> exe.run(paddle.static.default_startup_program())

            # serialize the default main program to bytes.
            >>> serialized_program = paddle.static.serialize_program([image], [predict])

            # deserialize bytes to program
            >>> deserialized_program = paddle.static.deserialize_program(serialized_program)

    """
    # verify feed_vars
    _check_vars('feed_vars', feed_vars)
    # verify fetch_vars
    _check_vars('fetch_vars', fetch_vars)

    program = _get_valid_program(kwargs.get('program', None))
    program = normalize_program(program, feed_vars, fetch_vars)
    legacy_format = kwargs.get('legacy_format', False)
    return _serialize_program(program, legacy_format=legacy_format)


def _serialize_program(program, legacy_format=False):
    """
    serialize given program to bytes.
    """
    return program.desc.serialize_to_string(legacy_format=legacy_format)


@static_only
def serialize_persistables(feed_vars, fetch_vars, executor, **kwargs):
    """

    Serialize parameters using given executor and default main program according to feed_vars and fetch_vars.

    Args:
        feed_vars(Tensor | list[Tensor]): Tensor needed by inference.
        fetch_vars(Tensor | list[Tensor]): Tensor returned by inference.
        kwargs: Supported keys including ``program``. Attention please, kwargs is used for backward compatibility mainly.

            - program(Program): specify a program if you don't want to use default main program.

    Returns:
        bytes: serialized program.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.enable_static()

            >>> path_prefix = "./infer_model"

            # User defined network, here a softmax regression example
            >>> image = paddle.static.data(name='img', shape=[None, 28, 28], dtype='float32')
            >>> label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
            >>> predict = paddle.static.nn.fc(image, 10, activation='softmax')

            >>> loss = paddle.nn.functional.cross_entropy(predict, label)

            >>> exe = paddle.static.Executor(paddle.CPUPlace())
            >>> exe.run(paddle.static.default_startup_program())

            # serialize parameters to bytes.
            >>> serialized_params = paddle.static.serialize_persistables([image], [predict], exe)

            # deserialize bytes to parameters.
            >>> main_program = paddle.static.default_main_program()
            >>> deserialized_params = paddle.static.deserialize_persistables(main_program, serialized_params, exe)

    """
    # verify feed_vars
    _check_vars('feed_vars', feed_vars)
    # verify fetch_vars
    _check_vars('fetch_vars', fetch_vars)

    program = _get_valid_program(kwargs.get('program', None))
    program = normalize_program(program, feed_vars, fetch_vars)
    return _serialize_persistables(program, executor)


def _serialize_persistables(program, executor):
    """
    Serialize parameters using given program and executor.
    """
    vars_ = list(filter(is_persistable, program.list_vars()))
    # warn if no variable found in model
    if len(vars_) == 0:
        warnings.warn(
            "no variable in your model, please ensure there are any "
            "variables in your model to save"
        )
        return None
    # create a new program and clone persistable vars to it
    save_program = Program()
    save_block = save_program.global_block()
    save_var_map = {}
    for var in vars_:
        if var.type != core.VarDesc.VarType.RAW:
            var_copy = _clone_var_in_block(save_block, var)
            save_var_map[var_copy.name] = var

    # create in_vars and out_var, then append a save_combine op to save_program
    in_vars = []
    for name in sorted(save_var_map.keys()):
        in_vars.append(save_var_map[name])

    out_var_name = unique_name.generate("out_var")
    out_var = save_block.create_var(
        type=core.VarDesc.VarType.RAW, name=out_var_name
    )
    out_var.desc.set_persistable(True)
    save_block.append_op(
        type='save_combine',
        inputs={'X': in_vars},
        outputs={'Y': out_var},
        attrs={'file_path': '', 'save_to_memory': True},
    )
    # run save_program to save vars
    # NOTE(zhiqiu): save op will add variable kLookupTablePath to save_program.desc,
    # which leads to diff between save_program and its desc. Call _sync_with_cpp
    # to keep consistency.
    save_program._sync_with_cpp()
    executor.run(save_program)
    # return serialized bytes in out_var
    return global_scope().find_var(out_var_name).get_bytes()


def save_to_file(path, content):
    """
    Save content to given path.

    Args:
        path(str): Path to write content to.
        content(bytes): Content to write.

    Returns:
        None

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.enable_static()
            >>> path_prefix = "./infer_model"

            # 用户自定义网络，此处用 softmax 回归为例。
            >>> image = paddle.static.data(name='img', shape=[None, 28, 28], dtype='float32')
            >>> label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
            >>> predict = paddle.static.nn.fc(image, 10, activation='softmax')
            >>> loss = paddle.nn.functional.cross_entropy(predict, label)
            >>> exe = paddle.static.Executor(paddle.CPUPlace())
            >>> exe.run(paddle.static.default_startup_program())

            # 序列化参数
            >>> serialized_params = paddle.static.serialize_persistables([image], [predict], exe)

            # 将序列化之后的参数保存到文件
            >>> params_path = path_prefix + ".params"
            >>> paddle.static.save_to_file(params_path, serialized_params)
    """

    if not isinstance(content, bytes):
        raise ValueError("'content' type should be bytes.")
    with open(path, "wb") as f:
        f.write(content)


@static_only
def save_inference_model(
    path_prefix, feed_vars, fetch_vars, executor, **kwargs
):
    """
    Save current model and its parameters to given path. i.e.
    Given ``path_prefix = "PATH/modelname"``, after invoking
    ``save_inference_model(path_prefix, feed_vars, fetch_vars, executor)``,
    you will find two files named ``modelname.pdmodel`` and ``modelname.pdiparams``
    under ``PATH``, which represent your model and parameters respectively.

    Args:
        path_prefix(str): Directory path to save model + model name without suffix.
        feed_vars(Tensor | list[Tensor]): Variables needed by inference.
        fetch_vars(Tensor | list[Tensor]): Variables returned by inference.
        executor(Executor): The executor that saves the inference model. You can refer
                            to :ref:`api_guide_executor_en` for more details.
        kwargs: Supported keys including 'program' and "clip_extra". Attention please, kwargs is used for backward compatibility mainly.

            - program(Program): specify a program if you don't want to use default main program.

            - clip_extra(bool): the flag indicating whether to clip extra information for every operator. Default: True.

            - legacy_format(bool): whether to save inference model in legacy format. Default: False.

    Returns:
        None

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.enable_static()

            >>> path_prefix = "./infer_model"

            # User defined network, here a softmax regression example
            >>> image = paddle.static.data(name='img', shape=[None, 28, 28], dtype='float32')
            >>> label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
            >>> predict = paddle.static.nn.fc(image, 10, activation='softmax')

            >>> loss = paddle.nn.functional.cross_entropy(predict, label)

            >>> exe = paddle.static.Executor(paddle.CPUPlace())
            >>> exe.run(paddle.static.default_startup_program())

            # Feed data and train process

            # Save inference model. Note we don't save label and loss in this example
            >>> paddle.static.save_inference_model(path_prefix, [image], [predict], exe)

            # In this example, the save_inference_mode inference will prune the default
            # main program according to the network's input node (img) and output node(predict).
            # The pruned inference program is going to be saved in file "./infer_model.pdmodel"
            # and parameters are going to be saved in file "./infer_model.pdiparams".

    """

    if in_pir_mode():
        save_pir_inference_model(
            path_prefix, feed_vars, fetch_vars, executor, **kwargs
        )
        return

    # check path_prefix, set model_path and params_path
    path_prefix = _normalize_path_prefix(path_prefix)
    try:
        # mkdir may conflict if pserver and trainer are running on the same machine
        dirname = os.path.dirname(path_prefix)
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    model_path = path_prefix + ".pdmodel"
    params_path = path_prefix + ".pdiparams"
    if os.path.isdir(model_path):
        raise ValueError(f"'{model_path}' is an existing directory.")
    if os.path.isdir(params_path):
        raise ValueError(f"'{params_path}' is an existing directory.")

    # verify feed_vars
    _check_vars('feed_vars', feed_vars)
    # verify fetch_vars
    _check_vars('fetch_vars', fetch_vars)

    program = _get_valid_program(kwargs.get('program', None))

    # do type promotion
    program = process_type_promotion(program)

    clip_extra = kwargs.get('clip_extra', True)
    # serialize and save program

    program = normalize_program(
        program,
        feed_vars,
        fetch_vars,
        skip_prune_program=kwargs.get('skip_prune_program', False),
    )
    legacy_format = kwargs.get('legacy_format', False)
    program_bytes = _serialize_program(
        program._remove_training_info(clip_extra=clip_extra),
        legacy_format=legacy_format,
    )

    save_to_file(model_path, program_bytes)

    vars = list(filter(is_persistable, program.list_vars()))

    if len(list(vars)) == 0:
        warnings.warn(
            "no variable in your model, please ensure there are any variables in your model to save"
        )

    if len(vars) > 0:
        save_dirname = os.path.dirname(params_path)
        params_filename = os.path.basename(params_path)
        save_vars(
            executor,
            dirname=save_dirname,
            main_program=program,
            predicate=is_persistable,
            filename=params_filename,
        )


@static_only
def deserialize_program(data):
    """

    Deserialize given data to a program.

    Args:
        data(bytes): serialized program.

    Returns:
        Program: deserialized program.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.enable_static()

            >>> path_prefix = "./infer_model"

            # User defined network, here a softmax regression example
            >>> image = paddle.static.data(name='img', shape=[None, 28, 28], dtype='float32')
            >>> label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
            >>> predict = paddle.static.nn.fc(image, 10, activation='softmax')

            >>> loss = paddle.nn.functional.cross_entropy(predict, label)

            >>> exe = paddle.static.Executor(paddle.CPUPlace())
            >>> exe.run(paddle.static.default_startup_program())

            # serialize the default main program to bytes.
            >>> serialized_program = paddle.static.serialize_program([image], [predict])

            # deserialize bytes to program
            >>> deserialized_program = paddle.static.deserialize_program(serialized_program)

    """
    program = Program.parse_from_string(data)
    if not core._is_program_version_supported(program._version()):
        raise ValueError(
            "Unsupported program version: %d\n" % program._version()
        )
    return program


# NOTE(liuyuanle): Due to load from memory, deserialize_persistables does not support loading weights with file sizes exceeding 2GB.
@static_only
def deserialize_persistables(program, data, executor):
    """

    Deserialize given data to parameters according to given program and executor.

    Args:
        program(Program): program that contains parameter names (to deserialize).
        data(bytes): serialized parameters.
        executor(Executor): executor used to run load op.

    Returns:
        Program: deserialized program.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.enable_static()

            >>> path_prefix = "./infer_model"

            # User defined network, here a softmax regression example
            >>> image = paddle.static.data(name='img', shape=[None, 28, 28], dtype='float32')
            >>> label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
            >>> predict = paddle.static.nn.fc(image, 10, activation='softmax')

            >>> loss = paddle.nn.functional.cross_entropy(predict, label)

            >>> exe = paddle.static.Executor(paddle.CPUPlace())
            >>> exe.run(paddle.static.default_startup_program())

            # serialize parameters to bytes.
            >>> serialized_params = paddle.static.serialize_persistables([image], [predict], exe)

            # deserialize bytes to parameters.
            >>> main_program = paddle.static.default_main_program()
            >>> deserialized_params = paddle.static.deserialize_persistables(main_program, serialized_params, exe)


    """
    if not isinstance(program, Program):
        raise TypeError(
            "program type must be `base.Program`, but received `%s`"
            % type(program)
        )
    # load params to a tmp program
    load_program = Program()
    load_block = load_program.global_block()
    vars_ = list(filter(is_persistable, program.list_vars()))

    origin_shape_map = {}
    load_var_map = {}
    check_vars = []
    sparse_vars = []
    for var in vars_:
        assert isinstance(var, Variable)
        if var.type == core.VarDesc.VarType.RAW:
            continue
        if isinstance(var, Parameter):
            origin_shape_map[var.name] = tuple(var.desc.get_shape())
        if var.type == core.VarDesc.VarType.SELECTED_ROWS:
            sparse_vars.append(var)
            continue
        var_copy = _clone_var_in_block(load_block, var)
        check_vars.append(var)
        load_var_map[var_copy.name] = var_copy

    if data is None:
        assert (
            len(origin_shape_map) == 0
        ), "Required 'data' shall be not None if program contains parameter, but received 'data' is None."
        return

    # append load_combine op to load parameters,
    load_var_list = []
    for name in sorted(load_var_map.keys()):
        load_var_list.append(load_var_map[name])
    load_block.append_op(
        type='load_combine',
        inputs={},
        outputs={"Out": load_var_list},
        # if load from memory, file_path is data
        attrs={'file_path': data, 'model_from_memory': True},
    )
    executor.run(load_program)
    # check var shape
    for var in check_vars:
        if not isinstance(var, Parameter):
            continue
        var_tmp = paddle.base.global_scope().find_var(var.name)
        assert var_tmp is not None, "can't not find var: " + var.name
        new_shape = (np.array(var_tmp.get_tensor())).shape
        assert var.name in origin_shape_map, var.name + " MUST in var list."
        origin_shape = origin_shape_map.get(var.name)
        if new_shape != origin_shape:
            raise RuntimeError(
                f"Shape mismatch, program needs a parameter with shape ({origin_shape}), "
                f"but the loaded parameter ('{var.name}') has a shape of ({new_shape})."
            )


def load_from_file(path):
    """
    Load file in binary mode.

    Args:
        path(str): Path of an existed file.

    Returns:
        bytes: Content of file.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> paddle.enable_static()
            >>> path_prefix = "./infer_model"

            # 用户自定义网络，此处用 softmax 回归为例。
            >>> image = paddle.static.data(name='img', shape=[None, 28, 28], dtype='float32')
            >>> label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
            >>> predict = paddle.static.nn.fc(image, 10, activation='softmax')
            >>> loss = paddle.nn.functional.cross_entropy(predict, label)
            >>> exe = paddle.static.Executor(paddle.CPUPlace())
            >>> exe.run(paddle.static.default_startup_program())

            # 序列化参数
            >>> serialized_params = paddle.static.serialize_persistables([image], [predict], exe)

            # 将序列化之后的参数保存到文件
            >>> params_path = path_prefix + ".params"
            >>> paddle.static.save_to_file(params_path, serialized_params)

            # 从文件加载序列化之后的参数
            >>> serialized_params_copy = paddle.static.load_from_file(params_path)
    """
    with open(path, 'rb') as f:
        data = f.read()
    return data


@static_only
def load_inference_model(path_prefix, executor, **kwargs):
    """

    Load inference model from a given path. By this API, you can get the model
    structure(Inference Program) and model parameters.

    Args:
        path_prefix(str | None): One of the following:
          - Directory path to save model + model name without suffix.
          - Set to None when reading the model from memory.
        executor(Executor): The executor to run for loading inference model.
                            See :ref:`api_guide_executor_en` for more details about it.
        kwargs: Supported keys including 'model_filename', 'params_filename'. Attention please, kwargs is used for backward compatibility mainly.

            - model_filename(str): specify model_filename if you don't want to use default name.

            - params_filename(str): specify params_filename if you don't want to use default name.

    Returns:
        list: The return of this API is a list with three elements:
        (program, feed_target_names, fetch_targets). The `program` is a
        ``Program`` (refer to :ref:`api_guide_Program_en`), which is used for inference.
        The `feed_target_names` is a list of ``str``, which contains names of variables
        that need to feed data in the inference program. The `fetch_targets` is a list of
        ``Variable`` (refer to :ref:`api_guide_Program_en`). It contains variables from which
        we can get inference results.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import numpy as np

            >>> paddle.enable_static()

            # Build the model
            >>> startup_prog = paddle.static.default_startup_program()
            >>> main_prog = paddle.static.default_main_program()
            >>> with paddle.static.program_guard(main_prog, startup_prog):
            ...     image = paddle.static.data(name="img", shape=[64, 784])
            ...     w = paddle.create_parameter(shape=[784, 200], dtype='float32')
            ...     b = paddle.create_parameter(shape=[200], dtype='float32')
            ...     hidden_w = paddle.matmul(x=image, y=w)
            ...     hidden_b = paddle.add(hidden_w, b)
            >>> exe = paddle.static.Executor(paddle.CPUPlace())
            >>> exe.run(startup_prog)

            # Save the inference model
            >>> path_prefix = "./infer_model"
            >>> paddle.static.save_inference_model(path_prefix, [image], [hidden_b], exe)

            >>> [inference_program, feed_target_names, fetch_targets] = (
            ...     paddle.static.load_inference_model(path_prefix, exe))
            >>> tensor_img = np.array(np.random.random((64, 784)), dtype=np.float32)
            >>> results = exe.run(inference_program,
            ...               feed={feed_target_names[0]: tensor_img},
            ...               fetch_list=fetch_targets)

            # In this example, the inference program was saved in file
            # "./infer_model.pdmodel" and parameters were saved in file
            # " ./infer_model.pdiparams".
            # By the inference program, feed_target_names and
            # fetch_targets, we can use an executor to run the inference
            # program to get the inference result.
    """
    if in_pir_mode():
        return load_pir_inference_model(path_prefix, executor, **kwargs)
    # check kwargs
    supported_args = ('model_filename', 'params_filename')
    deprecated_args = ('pserver_endpoints',)
    caller = inspect.currentframe().f_code.co_name
    _check_args(caller, kwargs, supported_args, deprecated_args)

    # load from memory
    if path_prefix is None:
        _logger.warning(
            "Load inference model from memory is deprecated. Please specify path_prefix."
        )
        model_filename = kwargs.get('model_filename', None)
        params_filename = kwargs.get('params_filename', None)
        if params_filename is None:
            raise ValueError(
                "params_filename cannot be None when path_prefix is None."
            )
        program_bytes = model_filename
        # deserialize bytes to program
        program = deserialize_program(program_bytes)

        # do type promotion
        program = process_type_promotion(program)

        vars = list(filter(is_persistable, program.list_vars()))
        if len(vars) > 0:
            load_vars(
                executor,
                # load from memory, dirname is None
                dirname=None,
                main_program=program,
                predicate=is_persistable,
                filename=params_filename,
            )
    # load from file
    else:
        # check and norm path_prefix
        path_prefix = _normalize_path_prefix(path_prefix)
        dir_path = os.path.dirname(path_prefix)
        if not os.path.isdir(dir_path):
            raise ValueError(f"There is no directory named {dir_path}")
        # set model_path and params_path in new way,
        # path_prefix represents a file path without suffix in this case.
        if not kwargs:
            model_path = path_prefix + ".pdmodel"
            params_path = path_prefix + ".pdiparams"
        # set model_path and params_path in old way for compatible,
        # path_prefix represents a directory path.
        else:
            model_filename = kwargs.get('model_filename', None)
            params_filename = kwargs.get('params_filename', None)
            # set model_path
            if model_filename is None:
                model_path = os.path.join(path_prefix, "__model__")
            else:
                model_path = os.path.join(
                    path_prefix, model_filename + ".pdmodel"
                )
                if not os.path.exists(model_path):
                    model_path = os.path.join(path_prefix, model_filename)
            # set params_path
            if params_filename is None:
                params_path = os.path.join(path_prefix, "")
            else:
                params_path = os.path.join(
                    path_prefix, params_filename + ".pdiparams"
                )
                if not os.path.exists(params_path):
                    params_path = os.path.join(path_prefix, params_filename)
            _logger.warning(
                "The old way to load inference model is deprecated. Please specify path_prefix."
                f" model path: {model_path}, params path: {params_path}"
            )

        program_bytes = load_from_file(model_path)

        # deserialize bytes to program
        program = deserialize_program(program_bytes)

        # do type promotion
        program = process_type_promotion(program)

        vars = list(filter(is_persistable, program.list_vars()))
        if len(vars) > 0:
            load_dirname = os.path.dirname(params_path)
            params_filename = os.path.basename(params_path)

            load_vars(
                executor,
                dirname=load_dirname,
                main_program=program,
                predicate=is_persistable,
                filename=params_filename,
            )

    feed_target_names = program.desc.get_feed_target_names()
    fetch_target_names = program.desc.get_fetch_target_names()
    fetch_targets = [
        program.global_block().var(name) for name in fetch_target_names
    ]

    return [program, feed_target_names, fetch_targets]


@dygraph_not_support
def save_vars(
    executor,
    dirname,
    main_program=None,
    vars=None,
    predicate=None,
    filename=None,
):
    """
    Save specific variables in the `Program` to files.

    There are two ways to specify the variables to be saved: set variables in
    a list and assign it to the `vars`, or use the `predicate` function to select
    variables that make `predicate(variable) == True`. The first way has a higher priority.

    The `dirname` is used to specify the folder where to save variables.
    If you prefer to save variables in separate files in the `dirname` folder,
    do not set `filename`. If you prefer to save all variables in a single file,
    use `filename` to specify it.

    Args:
        executor(Executor): The executor to run for saving variables.
        dirname(str, optional): The folder where to save variables.
                            When you need to save the parameter to the memory, set it to None.
        main_program(Program, optional): The program whose variables will be saved.
                                    If it is None, the default main program will
                                    be used automatically.
                                    Default: None
        vars(list[Variable], optional): The list contains all variables to be saved.
                                        Default: None
        predicate(function, optional): The function selects the variables that make
                                       `predicate(variable) == True`.
                                       Default: None
        filename(str, optional): If you prefer to save all variables in a single file,
                                 use `filename` to specify it. Otherwise, let `filename` be None.
                                 Default: None

    Returns:
        str: When saving parameters to a file, returns None.
             When saving parameters to memory, returns a binary string containing parameters.

    Raises:
        TypeError: If `main_program` is not an instance of Program nor None.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.static as static

            >>> paddle.enable_static()
            >>> main_prog = static.Program()
            >>> startup_prog = static.Program()
            >>> with static.program_guard(main_prog, startup_prog):
            ...     data = paddle.static.data(name="img", shape=[64, 784])
            ...     w = paddle.create_parameter(shape=[784, 200], dtype='float32', name='fc_w')
            ...     b = paddle.create_parameter(shape=[200], dtype='float32', name='fc_b')
            ...     hidden_w = paddle.matmul(x=data, y=w)
            ...     hidden_b = paddle.add(hidden_w, b)
            >>> place = static.CPUPlace()
            >>> exe = static.Executor(place)
            >>> exe.run(startup_prog)

            # The first usage: use `vars` to set the saved variables.
            >>> var_list = [w, b]
            >>> path = "./my_paddle_vars"

            # w and b will be save in a file named "var_file".
            >>> paddle.static.io.save_vars(executor=exe, dirname=path, vars=var_list,
            ...                 filename="vars_file")

            # The second usage: use `predicate` to select the saved variable.
            >>> def name_has_fc(var):
            ...     res = "fc" in var.name
            ...     return res
            >>> param_path = "./my_paddle_model"

            # all variables whose names contain "fc " are saved.
            >>> paddle.static.io.save_vars(executor=exe, dirname=param_path, main_program=main_prog, vars=None, predicate = name_has_fc)


    """
    if in_pir_mode():
        return save_vars_pir(dirname, main_program, vars, filename)

    save_to_memory = False
    if dirname is None and filename is None:
        save_to_memory = True

    main_program = _get_valid_program(main_program)

    if vars is None:
        return save_vars(
            executor,
            main_program=main_program,
            dirname=dirname,
            vars=list(filter(predicate, main_program.list_vars())),
            filename=filename,
        )
    else:
        params_var_name = "saved_params"
        # give warning when there is no var in model
        if len(list(vars)) == 0:
            warnings.warn(
                "no variable in your model, please ensure there are any variables in your model to save"
            )
            return None

        save_program = Program()
        save_block = save_program.global_block()

        save_var_map = {}
        for each_var in vars:
            # NOTE: don't save the variable which type is RAW
            if each_var.type == core.VarDesc.VarType.RAW:
                continue
            new_var = _clone_var_in_block_(save_block, each_var)
            if filename is None and save_to_memory is False:
                save_file_path = os.path.join(
                    os.path.normpath(dirname), new_var.name
                )
                save_block.append_op(
                    type='save',
                    inputs={'X': [new_var]},
                    outputs={},
                    attrs={'file_path': os.path.normpath(save_file_path)},
                )
            else:
                save_var_map[new_var.name] = new_var

        if filename is not None or save_to_memory:
            save_var_list = []
            for name in sorted(save_var_map.keys()):
                save_var_list.append(save_var_map[name])

            save_path = ''
            if save_to_memory is False:
                save_path = os.path.join(os.path.normpath(dirname), filename)

            saved_params = save_block.create_var(
                type=core.VarDesc.VarType.RAW, name=params_var_name
            )
            saved_params.desc.set_persistable(True)
            save_block.append_op(
                type='save_combine',
                inputs={'X': save_var_list},
                outputs={'Y': saved_params},
                attrs={
                    'file_path': save_path,
                    'save_to_memory': save_to_memory,
                },
            )

        # NOTE(zhiqiu): save op will add variable kLookupTablePath in save_program.desc,
        # which leads to diff on save_program and its desc. Call _sync_with_cpp
        # to keep consistency.
        save_program._sync_with_cpp()
        # flush to root_scope
        executor.flush()
        executor.run(save_program)
        if save_to_memory:
            return global_scope().find_var(params_var_name).get_bytes()


def load_vars(
    executor,
    dirname,
    main_program=None,
    vars=None,
    predicate=None,
    filename=None,
):
    """
    :api_attr: Static Graph

    This API loads variables from files by executor.

    There are two ways to specify the variables to be loaded: the first way, set
    variables in a list and assign it to the `vars`; the second way, use the
    `predicate` function to select variables that make `predicate(variable) == True`.
    The first way has a higher priority.

    The `dirname` is used to specify the folder where to load variables.
    If variables were saved in separate files in the folder `dirname`,
    set `filename` None. If all variables were saved in a single file,
    use `filename` to specify it.

    Args:
        executor(Executor): The executor to run for loading variables.
        dirname(str): The folder where to load the variables.
        main_program(Program, optional): The program whose variables will be loaded.
                                    If it is None, the default main program will
                                    be used automatically.
                                    Default: None
        vars(list[Variable], optional): The list that contains all variables to be loaded.
                                   Default: None
        predicate(function, optional): The function selects variables that make
                                        `predicate(variable) == True`.
                                        Default: None
        filename(str, optional): The file which saved all required variables. If variables
                                were saved in separate files, set it to be None.
                                Default: None

    Returns:
        None

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.static as static

            >>> paddle.enable_static()
            >>> main_prog = static.Program()
            >>> startup_prog = static.Program()
            >>> with static.program_guard(main_prog, startup_prog):
            ...     data = paddle.static.data(name="img", shape=[64, 784])
            ...     w = paddle.create_parameter(shape=[784, 200], dtype='float32', name='fc_w')
            ...     b = paddle.create_parameter(shape=[200], dtype='float32', name='fc_b')
            ...     hidden_w = paddle.matmul(x=data, y=w)
            ...     hidden_b = paddle.add(hidden_w, b)
            >>> place = paddle.CPUPlace()
            >>> exe = static.Executor(place)
            >>> exe.run(startup_prog)

            # The first usage: using `vars` to specify the variables.
            >>> path = "./my_paddle_vars"
            >>> var_list = [w, b]
            >>> paddle.static.io.save_vars(executor=exe, dirname=path, vars=var_list,
            ...                    filename="vars_file")
            >>> paddle.static.io.load_vars(executor=exe, dirname=path, vars=var_list,
            ...                    filename="vars_file")

            # w and b will be loaded, and they are supposed to
            # be saved in the same file named 'var_file' in the path "./my_paddle_vars".

            # The second usage: using the `predicate` function to select variables
            >>> param_path = "./my_paddle_model"
            >>> def name_has_fc(var):
            ...     res = "fc" in var.name
            ...     return res
            >>> paddle.static.io.save_vars(executor=exe, dirname=param_path, main_program=main_prog,
            ...                    vars=None, predicate=name_has_fc)
            >>> paddle.static.io.load_vars(executor=exe, dirname=param_path, main_program=main_prog,
            ...                    vars=None, predicate=name_has_fc)

            # Load All variables in the `main_program` whose name includes "fc".
            # And all the variables are supposed to be saved in separate files.

    """
    if in_pir_mode():
        return load_vars_pir(executor, dirname, main_program, vars, filename)

    vars_from_memory = False
    if dirname is not None:
        dirname = os.path.normpath(dirname)
    else:
        vars_from_memory = True

    if filename == '':
        filename = None

    if vars is None:
        if main_program is None:
            main_program = default_main_program()
        if not isinstance(main_program, Program):
            raise TypeError(
                "The type of input main_program is invalid, expected type is base.Program, but received %s"
                % type(main_program)
            )

        load_vars(
            executor,
            dirname=dirname,
            main_program=main_program,
            vars=list(filter(predicate, main_program.list_vars())),
            filename=filename,
        )
    else:
        load_prog = Program()
        load_block = load_prog.global_block()

        if main_program is None:
            main_program = default_main_program()

        if not isinstance(main_program, Program):
            raise TypeError(
                "The type of input main_program is invalid, expected type is base.Program, but received %s"
                % type(main_program)
            )

        # save origin param shape
        orig_para_shape = {}
        load_var_map = {}

        check_vars = []
        sparse_vars = []

        for each_var in vars:
            assert isinstance(each_var, Variable)

            if each_var.type == core.VarDesc.VarType.RAW:
                continue

            if isinstance(each_var, Parameter):
                orig_para_shape[each_var.name] = tuple(
                    each_var.desc.get_shape()
                )

            if each_var.type == core.VarDesc.VarType.SELECTED_ROWS:
                sparse_vars.append(each_var)
                continue

            new_var = _clone_var_in_block_(load_block, each_var)
            check_vars.append(each_var)

            if filename is None:
                if dirname is None:
                    raise ValueError(
                        "The directory path and params cannot be None at the same time."
                    )
                load_block.append_op(
                    type='load',
                    inputs={},
                    outputs={'Out': [new_var]},
                    attrs={'file_path': os.path.join(dirname, new_var.name)},
                )
            else:
                load_var_map[new_var.name] = new_var

        for each_var in sparse_vars:
            assert isinstance(each_var, Variable)

            if filename is not None:
                raise ValueError(
                    "SelectedRows can not be load with load_combine"
                )

            new_var = _clone_var_in_block_(load_block, each_var)

            var_path = os.path.join(dirname, new_var.name)
            if not os.path.exists(var_path):
                raise ValueError(
                    f"SelectedRows var {new_var.name} can not find at {var_path}"
                )

            if os.path.isfile(var_path):
                load_block.append_op(
                    type='load',
                    inputs={},
                    outputs={'Out': [new_var]},
                    attrs={'file_path': os.path.join(dirname, new_var.name)},
                )
            else:
                blocks = []
                block_paths = os.listdir(var_path)

                for block in block_paths:
                    if block.startswith(new_var.name):
                        blocks.append(block)

                slices = []
                for block in blocks:
                    slice = load_block.create_var(
                        name=block,
                        type=new_var.type,
                        shape=new_var.shape,
                        dtype=new_var.dtype,
                        persistable=False,
                    )
                    slices.append(slice)

                    file_path = os.path.join(var_path, block, "Param")
                    load_block.append_op(
                        type='load',
                        inputs={},
                        outputs={'Out': [slice]},
                        attrs={'file_path': file_path},
                    )

                load_block.append_op(
                    type='lookup_sparse_table_merge',
                    inputs={'X': slices},
                    outputs={'Out': new_var},
                    attrs={},
                )

        if filename is not None:
            load_var_list = []
            for name in sorted(load_var_map.keys()):
                load_var_list.append(load_var_map[name])

            if vars_from_memory is False:
                filename = os.path.join(dirname, filename)

            load_block.append_op(
                type='load_combine',
                inputs={},
                outputs={"Out": load_var_list},
                attrs={
                    'file_path': filename,
                    'model_from_memory': vars_from_memory,
                },
            )
        executor.run(load_prog)

        # check var shape
        for each_var in check_vars:
            if not isinstance(each_var, Parameter):
                continue
            var_temp = paddle.base.global_scope().find_var(each_var.name)
            assert var_temp is not None, "can't not find var: " + each_var.name
            new_shape = (np.array(var_temp.get_tensor())).shape
            assert each_var.name in orig_para_shape, (
                each_var.name + "MUST in var list"
            )
            orig_shape = orig_para_shape.get(each_var.name)
            if new_shape != orig_shape:
                raise RuntimeError(
                    f"Variable's shape does not match, the Program requires a parameter with the shape of ({orig_shape}), "
                    f"while the loaded parameter (namely [ {each_var.name} ]) has a shape of  ({new_shape})."
                )


@static_only
def save(program, model_path, protocol=4, **configs):
    """

    This function save parameters, optimizer information and network description to model_path.

    The parameters contains all the trainable Tensor, will save to a file with suffix ".pdparams".
    The optimizer information contains all the Tensor used by optimizer. For Adam optimizer, contains beta1, beta2, momentum etc. All the information will save to a file with suffix ".pdopt". (If the optimizer have no Tensor need to save (like SGD), the fill will not generated).
    The network description is the description of the program. It's only used for deployment. The description  will save to a file with a suffix ".pdmodel".

    Args:
        program(Program) : The program to saved.
        model_path(str): the file prefix to save the program. The format is "dirname/file_prefix". If file_prefix is empty str. A exception will be raised
        protocol(int, optional): The protocol version of pickle module must be greater than 1 and less than 5.
                                 Default: 4
        configs(dict, optional) : optional keyword arguments.

    Returns:
        None

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.static as static

            >>> paddle.enable_static()

            >>> x = static.data(name="x", shape=[10, 10], dtype='float32')
            >>> y = static.nn.fc(x, 10)
            >>> z = static.nn.fc(y, 10)

            >>> place = paddle.CPUPlace()
            >>> exe = static.Executor(place)
            >>> exe.run(static.default_startup_program())
            >>> prog = static.default_main_program()

            >>> static.save(prog, "./temp")
    """
    if in_pir_mode():
        return save_pir(program, model_path, protocol, **configs)

    base_name = os.path.basename(model_path)
    assert (
        base_name != ""
    ), "The input model_path MUST be format of dirname/filename [dirname\\filename in Windows system], but received model_path is empty string."
    if 'pickle_protocol' in configs:
        protocol = configs['pickle_protocol']
        warnings.warn(
            "'pickle_protocol' is a deprecated argument. Please use 'protocol' instead."
        )

    if not isinstance(protocol, int):
        raise ValueError(
            f"The 'protocol' MUST be `int`, but received {type(protocol)}"
        )

    if protocol < 2 or protocol > 4:
        raise ValueError(
            f"Expected 1<'protocol'<5, but received protocol={protocol}"
        )

    dir_name = os.path.dirname(model_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)

    def get_tensor(var):
        t = global_scope().find_var(var.name).get_tensor()
        return np.array(t)

    parameter_list = list(filter(is_parameter, program.list_vars()))
    param_dict = {p.name: get_tensor(p) for p in parameter_list}

    param_dict = _unpack_saved_dict(param_dict, protocol)

    # When value of dict is lager than 4GB ,there is a Bug on 'MAC python3'
    if sys.platform == 'darwin' and sys.version_info.major == 3:
        pickle_bytes = pickle.dumps(param_dict, protocol=protocol)
        with open(model_path + ".pdparams", 'wb') as f:
            max_bytes = 2**30
            for i in range(0, len(pickle_bytes), max_bytes):
                f.write(pickle_bytes[i : i + max_bytes])
    else:
        with open(model_path + ".pdparams", 'wb') as f:
            pickle.dump(param_dict, f, protocol=protocol)

    optimizer_var_list = list(
        filter(is_belong_to_optimizer, program.list_vars())
    )

    opt_dict = {p.name: get_tensor(p) for p in optimizer_var_list}
    with open(model_path + ".pdopt", 'wb') as f:
        pickle.dump(opt_dict, f, protocol=protocol)

    main_program = program.clone()
    program.desc.flush()

    with open(model_path + ".pdmodel", "wb") as f:
        f.write(program.desc.serialize_to_string())


@static_only
def load(program, model_path, executor=None, var_list=None):
    """
    :api_attr: Static Graph

    This function get parameters and optimizer information from program, and then get corresponding value from file.
    An exception will throw if shape or dtype of the parameters is not match.

    This function can also load model file saved with [ save_params, save_persistables, save_vars ].
    var_list can not be None  when load single model file
    ( filename is not None When save_params, save_persistables or save_vars is called ).

    Args:
        program(Program): The program will be loaded
        model_path(str): The file prefix store the program
        executor(Executor, optional): The executor used for initialize the parameter
                                      When startup program is not run.
        var_list(list|tuple, optional): The Tensor list/tuple to load single model file saved with
                                  [ save_params, save_persistables, save_vars ].
                                  Default: None

    Returns:
        None

     Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.static as static

            >>> paddle.enable_static()

            >>> x = static.data(name="x", shape=[10, 10], dtype='float32')
            >>> y = static.nn.fc(x, 10)
            >>> z = static.nn.fc(y, 10)

            >>> place = paddle.CPUPlace()
            >>> exe = static.Executor(place)
            >>> exe.run(static.default_startup_program())
            >>> prog = static.default_main_program()

            >>> static.save(prog, "./temp")
            >>> static.load(prog, "./temp")
    """
    if in_pir_mode():
        return load_pir(program, model_path, executor, var_list)

    assert executor is None or isinstance(executor, Executor)

    model_prefix = model_path
    if model_prefix.endswith(".pdparams"):
        model_prefix = model_prefix[:-9]
    elif model_prefix.endswith(".pdopt"):
        model_prefix = model_prefix[:-6]
    elif model_prefix.endswith(".pdmodel"):
        model_prefix = model_prefix[:-8]

    parameter_file_name = model_prefix + ".pdparams"

    if not os.path.exists(parameter_file_name):
        # model file save by base.save not found, try to load model file saved with
        # [save_vars, save_params, save_persistables]
        _logger.debug(
            f"{parameter_file_name} not found, try to load model file saved with [ save_params, save_persistables, save_vars ]"
        )
        if executor is None:
            raise ValueError(
                "executor is required when loading model file saved with [ save_params, save_persistables, save_vars ]"
            )

        if var_list is not None:
            var_list_names = [var.name for var in var_list]
        else:
            var_list_names = None

        if os.path.isdir(model_path):
            binary_file_set = set()
            for root, dirs, files in os.walk(model_path, topdown=False):
                for f in files:
                    binary_file_set.add(
                        os.path.join(root, f).replace("\\", "/")
                    )
            program_var_list = list(program.list_vars())
            loaded_var_list = []
            for var in program_var_list:
                var_path = os.path.join(model_path, var.name).replace("\\", "/")
                load_condition = (
                    var_list_names is None or var.name in var_list_names
                )
                if var_path in binary_file_set and load_condition:
                    loaded_var_list.append(var)
                    binary_file_set.remove(var_path)
            if len(binary_file_set) > 0:
                unused_var_list = " ".join(list(binary_file_set))
                _logger.warning(
                    "variable file [ %s ] not used"
                    % (" ".join(list(binary_file_set)))
                )
            try:
                load_vars(
                    executor=executor, dirname=model_path, vars=loaded_var_list
                )
            except RuntimeError as e:
                _logger.error(e)
                raise e
            except:
                raise RuntimeError(
                    "Failed to load model file, please make sure model file is saved with the "
                    "following APIs: save_params, save_persistables, save_vars"
                )

            return
        elif os.path.isfile(model_path):
            if var_list is None:
                raise ValueError(
                    "var_list is required when loading model file saved with [ save_params, save_persistables, save_vars ]"
                )
            program_var_list = program.list_vars()
            program_var_name_set = {var.name for var in program_var_list}

            # check all the variable included in program
            for var in var_list:
                if var.name not in program_var_name_set:
                    raise LookupError(
                        "loaded var [{}] is not in program variable list"
                    )

            dir_name, file_name = os.path.split(model_path)
            try:
                load_vars(
                    executor=executor,
                    dirname=dir_name,
                    vars=var_list,
                    filename=file_name,
                )
            except RuntimeError as e:
                _logger.error(e)
                raise e
            except:
                raise RuntimeError(
                    "Failed to load model file , please make sure model file is saved with the "
                    "the following APIs: [ save_params, save_persistables, save_vars ]. "
                    "When these API called, filename CANNOT be None"
                )

            return

    def set_var(var, ndarray):
        t = global_scope().find_var(var.name).get_tensor()
        p = t._place()
        if p.is_cpu_place():
            place = paddle.base.CPUPlace()
        elif p.is_cuda_pinned_place():
            place = paddle.base.CUDAPinnedPlace()
        elif p.is_xpu_place():
            p = paddle.base.core.Place()
            p.set_place(t._place())
            place = paddle.base.XPUPlace(p.xpu_device_id())
        elif p.is_custom_place():
            p = paddle.base.core.Place()
            p.set_place(t._place())
            place = paddle.base.CustomPlace(
                paddle.device.get_device().split(':')[0], p.custom_device_id()
            )
        else:
            p = paddle.base.core.Place()
            p.set_place(t._place())
            place = paddle.base.CUDAPlace(p.gpu_device_id())

        t.set(ndarray, place)

    parameter_list = list(filter(is_parameter, program.list_vars()))

    if executor:
        paddle.base.core._create_loaded_parameter(
            parameter_list, global_scope(), executor._default_executor
        )
    with open(parameter_file_name, 'rb') as f:
        # When value of dict is lager than 4GB ,there is a Bug on 'MAC python3'
        if sys.platform == 'darwin' and sys.version_info.major == 3:
            load_dict = _pickle_loads_mac(parameter_file_name, f)
        else:
            load_dict = _safe_load_pickle(f, encoding='latin1')
        load_dict = _pack_loaded_dict(load_dict)
    for v in parameter_list:
        assert (
            v.name in load_dict
        ), f"Can not find [{v.name}] in model file [{parameter_file_name}]"
        set_var(v, load_dict[v.name])

    optimizer_var_list = list(
        filter(is_belong_to_optimizer, program.list_vars())
    )

    if len(optimizer_var_list) > 0:
        opt_file_name = model_prefix + ".pdopt"
        assert os.path.exists(
            opt_file_name
        ), f"Optimizer file [{opt_file_name}] not exits"

        if executor:
            paddle.base.core._create_loaded_parameter(
                optimizer_var_list, global_scope(), executor._default_executor
            )

        with open(opt_file_name, 'rb') as f:
            load_dict = _safe_load_pickle(f, encoding='latin1')
        for v in optimizer_var_list:
            assert (
                v.name in load_dict
            ), f"Can not find [{v.name}] in model file [{opt_file_name}]"
            set_var(v, load_dict[v.name])


@static_only
def set_program_state(program, state_dict):
    """
    Set program parameter from state_dict

    An exception will throw if shape or dtype of the parameters is not match.

    NOTICE: This function MUST called after run start_up_program

    Args:
        program(Program): The program to be set
        state_dict(dict): the dict store Parameter and optimizer information
    Returns:
        None

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.static as static

            >>> paddle.enable_static()

            >>> x = static.data(name="x", shape=[10, 10], dtype='float32')
            >>> y = static.nn.fc(x, 10)
            >>> z = static.nn.fc(y, 10)

            >>> place = paddle.CPUPlace()
            >>> exe = static.Executor(place)
            >>> exe.run(static.default_startup_program())
            >>> prog = static.default_main_program()

            >>> static.save(prog, "./temp")
            >>> program_state = static.load_program_state("./temp")

            >>> static.set_program_state(prog, program_state)
    """
    state_dict = _pack_loaded_dict(state_dict)
    if in_pir_mode():
        params, opts = get_pir_parameters(program)
        parameter_list = params + opts
        parameter_list = [var for var in parameter_list if var.persistable]
    else:
        parameter_list = list(filter(is_persistable, program.list_vars()))

    used_para_list = {}
    for para in parameter_list:
        var_temp = paddle.base.global_scope().find_var(para.name)
        assert (
            var_temp is not None
        ), f"Variable [ {para.name} ] Not found, Please make sure run startup program"
        if para.name in state_dict:
            # set value from state dict
            orig_para_np = np.array(var_temp.get_tensor())
            new_para_np = state_dict[para.name]
            assert orig_para_np.shape == new_para_np.shape, (
                f"Parameter's shape does not match, the Program requires a parameter with the shape of ({orig_para_np.shape}), "
                f"while the loaded parameter (namely [ {para.name} ]) has a shape of  ({new_para_np.shape})."
            )
            assert orig_para_np.dtype == new_para_np.dtype, (
                f"Parameter's data type does not match, the Program requires a parameter with a dtype of ({orig_para_np.dtype}), "
                f"while the loaded parameter (namely [ {para.name} ]) has a dtype of  ({new_para_np.dtype})."
            )

            ten = var_temp.get_tensor()
            ten_place = ten._place()

            # assert ten_place.is_gpu_place() or ten_place.is_cpu_place(), \
            #    "Place not support, only support CPUPlace and GPUPlace, now is {}".format(str(ten_place))
            py_place = paddle.base.CPUPlace()
            if ten_place.is_cuda_pinned_place():
                place = paddle.base.CUDAPinnedPlace()
            elif ten_place.is_gpu_place():
                p = paddle.base.core.Place()
                p.set_place(ten_place)
                py_place = paddle.base.CUDAPlace(p.gpu_device_id())
            elif ten_place.is_xpu_place():
                p = paddle.base.core.Place()
                p.set_place(ten_place)
                py_place = paddle.base.XPUPlace(p.xpu_device_id())

            ten.set(new_para_np, py_place)

            used_para_list[para.name] = 1

    unused_para_list = []
    for k, v in state_dict.items():
        if k not in used_para_list:
            unused_para_list.append(k)
    if len(unused_para_list) > 0:
        warnings.warn(
            "This list is not set, Because of Parameter not found in program. There are: {}".format(
                " ".join(unused_para_list)
            )
        )


@dygraph_not_support
def get_program_persistable_vars(program):
    """
    Get all the persistable vars from Program.
    Args:
        var(Program): The Program to get persistable vars
    Returns:
        list: The list contains all persistable vars in the program
    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.static.io as io
            >>> paddle.enable_static()
            >>> data = paddle.static.data(name="img", shape=[64, 784])
            >>> w = paddle.create_parameter(shape=[784, 200], dtype='float32', name='fc_w')
            >>> b = paddle.create_parameter(shape=[200], dtype='float32', name='fc_b')
            >>> list_para  = io.get_program_persistable_vars(  paddle.static.default_main_program() )
    """
    return list(filter(is_persistable, program.list_vars()))


def load_program_state(model_path, var_list=None):
    """

    Load program state from local file

    Args:
        model_path(str): The file prefix store the program
        var_list(list|tuple, optional): The Tensor list/tuple to load saved with
                                  [ save_params, save_persistables, save_vars ].
                                  Default: None.
                                  The var_list is only used to get name,
                                  will not be modified.
    Returns:
        state_dict(dict): the dict store Parameter and optimizer information

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> import paddle.static as static

            >>> paddle.enable_static()

            >>> x = static.data(name="x", shape=[10, 10], dtype='float32')
            >>> y = static.nn.fc(x, 10)
            >>> z = static.nn.fc(y, 10)

            >>> place = paddle.CPUPlace()
            >>> exe = static.Executor(place)
            >>> exe.run(static.default_startup_program())
            >>> prog = static.default_main_program()

            >>> static.save(prog, "./temp")
            >>> program_state = static.load_program_state("./temp")
    """
    model_prefix = model_path
    if model_prefix.endswith(".pdparams"):
        model_prefix = model_prefix[:-9]
    elif model_prefix.endswith(".pdopt"):
        model_prefix = model_prefix[:-6]
    elif model_prefix.endswith(".pdmodel"):
        model_prefix = model_prefix[:-8]

    parameter_file_name = model_prefix + ".pdparams"
    if not os.path.exists(parameter_file_name):
        # model file saved with base.save is not found, try to load model file saved with
        # [save_vars, save_params, save_persistables]
        _logger.debug(
            f"{parameter_file_name} not found, try to load model file saved with [ save_params, save_persistables, save_vars ]"
        )

        var_name_list = []
        if var_list is None and os.path.isfile(model_path):
            raise ValueError(
                "var_list can not be None when model_path is a file type"
            )

        for root, dirs, files in os.walk(model_path, topdown=False):
            for f in files:
                file_path = os.path.join(root, f)
                var_temp_name = os.path.relpath(file_path, model_path)
                var_temp_name = var_temp_name.replace("\\", "/")
                var_name_list.append(var_temp_name)

        with _load_program_scope():
            load_prog = Program()
            load_block = load_prog.global_block()

            def clone_var_to_block(block, var):
                if not isinstance(var, Variable):
                    raise TypeError("value in var_list must be variable")
                return block.create_var(
                    name=var.name,
                    shape=var.shape,
                    dtype=var.dtype,
                    type=var.type,
                    lod_level=var.lod_level
                    if var.desc.type() == core.VarDesc.VarType.LOD_TENSOR
                    else None,
                    persistable=True,
                )

            def _load_vars_with_try_catch(
                exe, dirname, vars, filename, raise_error=True
            ):
                try:
                    load_vars(
                        executor=exe,
                        dirname=dirname,
                        vars=vars,
                        filename=filename,
                    )
                    return True
                except:
                    error_str = (
                        "Failed to load model/variables `%s`, please make sure "
                        "model/variables file is saved with the following APIs: "
                        "save_params, save_persistables, save_vars."
                    )
                    filenames = (
                        [var.name for var in vars]
                        if filename is None
                        else filename
                    )
                    if raise_error:
                        raise RuntimeError(error_str % filenames)
                    else:
                        warnings.warn(error_str % filenames, RuntimeWarning)
                return False

            place = paddle.base.CPUPlace()
            exe = paddle.base.Executor(place)

            loaded_var_list = []

            if os.path.isfile(model_path):
                # when model_path is file, var_list cannot be None
                dir_name, file_name = os.path.split(model_path)
                for var in var_list:
                    loaded_var_list.append(clone_var_to_block(load_block, var))
                _load_vars_with_try_catch(
                    exe, dir_name, loaded_var_list, file_name
                )
            else:
                # var_list can be None or not None
                if var_list is not None:
                    for var in var_list:
                        loaded_var_list.append(
                            clone_var_to_block(load_block, var)
                        )
                    _load_vars_with_try_catch(
                        exe, model_path, loaded_var_list, None
                    )
                else:
                    for var_name in var_name_list:
                        # NOTE(chenweihang): If identify which files the user wants
                        # to load from the disk, we load these variables one by one.
                        # If a file does not exist, we only warn the user that the
                        # file may be an irrelevant file, but does not throw an error
                        # to ensure that other legal variables can be loaded.
                        temp_var = load_block.create_var(
                            name=var_name, persistable=True
                        )
                        if _load_vars_with_try_catch(
                            exe, model_path, [temp_var], None, False
                        ):
                            loaded_var_list.append(temp_var)

            res_dict = {}
            for var in loaded_var_list:
                res_dict[var.name] = np.asarray(
                    paddle.base.global_scope().find_var(var.name).get_tensor()
                )

            return res_dict

    assert os.path.exists(
        parameter_file_name
    ), f"Parameter file [{parameter_file_name}] not exits"

    with open(parameter_file_name, 'rb') as f:
        # When value of dict is lager than 4GB ,there is a Bug on 'MAC python3'
        if sys.platform == 'darwin' and sys.version_info.major == 3:
            para_dict = _pickle_loads_mac(parameter_file_name, f)
        else:
            para_dict = _safe_load_pickle(f, encoding='latin1')
    para_dict = _pack_loaded_dict(para_dict)

    opt_file_name = model_prefix + ".pdopt"
    if os.path.exists(opt_file_name):
        with open(opt_file_name, 'rb') as f:
            opti_dict = _safe_load_pickle(f, encoding='latin1')

        para_dict.update(opti_dict)

    return para_dict
