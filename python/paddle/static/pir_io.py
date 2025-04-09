#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import pir
from paddle.autograd.backward_utils import (
    ValueSet,
    get_real_op_inputs,
    get_real_op_outputs,
    some_in_set,
)
from paddle.base import (
    core,
    default_main_program,
)
from paddle.base.executor import Executor, global_scope
from paddle.base.framework import (
    dygraph_not_support,
    static_only,
)
from paddle.base.log_helper import get_logger
from paddle.framework.io_utils import (
    _pack_loaded_dict,
    _pickle_loads_mac,
    _unpack_saved_dict,
)

from .io_utils import (
    _check_args,
    _check_vars,
    _get_valid_program,
    _normalize_path_prefix,
    _safe_load_pickle,
)

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)


def get_pir_parameters(program):
    """
    Get parameters and optimizer variables from program.
        Args:
            program(Program): The program to get parameters and optimizer variables.
    """
    params = []
    opts = []
    for var in program.list_vars():
        if (
            var.is_parameter
            or var.get_defining_op().name() == "builtin.parameter"
        ):
            params.append(var)
        elif var.persistable and var.get_defining_op().name() == "pd_op.data":
            opts.append(var)
    return params, opts


def get_pir_feed_and_fetch(program):
    feed_name_list = []
    fetch_targets = []
    for op in program.global_block().ops:
        if op.name() == "pd_op.data" or op.name() == "pd_op.feed":
            feed_name_list.append(op.attrs()["name"])
        if op.name() == "pd_op.fetch":
            fetch_targets.extend(op.operands_source())
    return feed_name_list, fetch_targets


def set_var(name, ndarray):
    t = global_scope().find_var(name).get_tensor()
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


def append_pir_feed_ops(program, feed_vars):
    """
    Append feed ops to the program.
    Args:
        program(Program): Specify a program you want to append fetch op.
        feed_vars(Value | list[Value]): Values should be feed.
    Returns:
        modify program
    """
    for i, var in enumerate(feed_vars):
        orig_op = var.get_defining_op()
        if orig_op.name() != 'pd_op.feed' and orig_op.name() != 'pd_op.data':
            value = paddle._pir_ops.data(
                "feed_name_" + str(i),
                var.shape,
                var.dtype,
                paddle.base.core.Place(),
            )
            var.replace_all_uses_with(value)
            value.get_defining_op().move_before(orig_op)

    for i, var in enumerate(feed_vars):
        orig_op = var.get_defining_op()
        if orig_op.name() != 'pd_op.feed' and orig_op.name() != 'pd_op.data':
            orig_op.get_parent_block().remove_op(orig_op)


def append_pir_fetch_ops(program, fetch_name_var_maps):
    """
    Append fetch ops to the program.
    Args:
        program(Program): Specify a program you want to append fetch op.
        fetch_vars(Tensor | list[Tensor]): Values returned by inference.
    Returns:
        modify program
    """
    for i, (var, name) in enumerate(fetch_name_var_maps):
        out = paddle._pir_ops.fetch(var, name, i)
        out.persistable = True


def pir_prune_with_input(program, feed_vars, target_vars):
    """
    Prune a program according to feed_vars and target_vars.
    Args:
        program(Program): Specify a program you want to prune.
        feed_vars(Tensor | list[Tensor]): Values needed by inference.
        target_vars(Tensor | list[Tensor]): Values returned by inference.
    Returns
        modify program
    """
    if not isinstance(program, paddle.static.Program):
        raise TypeError(
            f"program type must be `paddle.static.Program`, but received `{type(program)}`"
        )

    total_ops = program.global_block().ops
    intersection_op_flags = [True] * len(total_ops)
    skip_prune_ops = ["builtin.parameter"]

    # from output to input
    target_vars_ = ValueSet(target_vars)
    for i, op in reversed(list(enumerate(total_ops))):
        if (
            some_in_set(get_real_op_outputs(op), target_vars_)
            or op.name() in skip_prune_ops
        ):
            for operand in get_real_op_inputs(op):
                target_vars_.add(operand)
        else:
            intersection_op_flags[i] = False

    for i, op in reversed(list(enumerate(total_ops))):
        if not intersection_op_flags[i]:
            if some_in_set(get_real_op_outputs(op), ValueSet(feed_vars)):
                raise ValueError(
                    f"The feed_var create by: '{op.name()}' is not involved in the target_vars calculation"
                    f"Please remove it from feed_vars ."
                )
            program.global_block().remove_op(op)


def _inference_optimize(program, prune_read_op=True):
    """
    This method will create a new program and do following adjustments on it:
    1. Remove all reader variables and their creator ops if exist.

    2. Remove the :code:`read_op` if exists.

    3. change the :code:`is_test`
    attribute of operators to :code:`True`. All the :code:`Parameter`
    information will be lost.

    Args:
        prune_read_op(bool): remove the read ops that are added by py_reader
                             for cpp inference library

    Notes: This API is a very low level API. Use
    :code:`Program.clone(for_test=True)` instead.

    Returns:
        Program: The new program.
    """

    # remove all readers and the read_op if exist
    if prune_read_op:
        pass

    # change all `is_test` attributes to True
    for block in program.blocks:
        for op in block.ops:
            if op.has_attr("is_test"):
                op.set_bool_attr("is_test", True)
            if op.name() == "pd_op.batch_norm":
                # Remove the output ReserveSpace of batch_norm if exists.
                pass


def normalize_pir_program(program, feed_vars, fetch_vars, **kwargs):
    """

    Normalize/Optimize a program according to feed_vars and fetch_vars.

    Args:
        program(Program): Specify a program you want to optimize.
        feed_vars(Tensor | list[Tensor]): Values needed by inference.
        fetch_vars(Tensor | list[Tensor]): Values returned by inference.
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
    if not isinstance(program, paddle.static.Program):
        raise TypeError(
            f"program type must be `paddle.static.Program`, but received `{type(program)}`"
        )
    if not isinstance(feed_vars, list):
        feed_vars = [feed_vars]
    if not all(isinstance(v, pir.Value) for v in feed_vars):
        raise TypeError("feed_vars type must be a Value or a list of Value.")
    if not isinstance(fetch_vars, list):
        fetch_vars = [fetch_vars]
    if not all(isinstance(v, pir.Value) for v in fetch_vars):
        raise TypeError("fetch_vars type must be a Value or a list of Value.")

    if len(program.global_block().ops) == 0:
        raise ValueError(
            "program must not be empty. at least one operator is required!"
        )

    # remind users to set auc_states to 0 if auc op were found.
    for op in program.global_block().ops:
        if op.name() == 'pd_op.auc':
            warnings.warn(
                "Be sure that you have set auc states to 0 before saving inference model."
            )
            break

    # serialize program
    value_map = paddle.pir.IrMapping()
    copy_program = program.clone(value_map)
    global_block = copy_program.global_block()
    clone_feed_vars = [value_map.look_up(v) for v in feed_vars]
    clone_fetch_vars = [value_map.look_up(v) for v in fetch_vars]

    for op in global_block.ops:
        # can not delete feed op because it's output used by other op.
        if op.name() == "pd_op.fetch":
            global_block.remove_op(op)

    skip_prune_program = kwargs.get('skip_prune_program', False)
    # if feed var is not connect with target_vars, it will be delete.
    if not skip_prune_program:
        pir_prune_with_input(copy_program, clone_feed_vars, clone_fetch_vars)
    _inference_optimize(copy_program, prune_read_op=True)

    fetch_vars_tuple = []
    for i, var in enumerate(clone_fetch_vars):
        scale_op = var.get_defining_op()
        orig_var = var
        if scale_op.name() == "pd_op.scale":
            full_op = scale_op.operand_source(1).get_defining_op()
            if full_op.has_attr("value") and full_op.attrs()['value'] == 1.0:
                orig_var = scale_op.operand_source(0)
        if orig_var.has_name:
            fetch_vars_tuple.append((orig_var, orig_var.name))
        else:
            fetch_vars_tuple.append((var, "fetch_name_" + str(i)))
    with paddle.static.program_guard(copy_program):
        append_pir_feed_ops(copy_program, clone_feed_vars)
        append_pir_fetch_ops(copy_program, fetch_vars_tuple)

    return copy_program


@dygraph_not_support
def save_vars_pir(
    dirname,
    main_program=None,
    vars=None,
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
        dirname(str, optional): The folder to save variables.
                            When you need to save the parameter to the memory, set it to None.
        main_program(Program, optional): The program whose variables will be saved.
                                    If it is None, the default main program will
                                    be used automatically.
                                    Default: None
        vars(list[Variable], optional): The list contains all variables to be saved.
                                        Default: None
        filename(str, optional): If you prefer to save all variables in a single file,
                                 use `filename` to specify it. Otherwise, let `filename` be None.
                                 Default: None

    Returns:
        str: When saving parameters to a file, returns None.
             When saving parameters to memory, returns a binary string containing parameters.
    """

    save_to_memory = False
    if dirname is None and filename is None:
        save_to_memory = True

    main_program = _get_valid_program(main_program)

    if vars is None:
        param, opt = get_pir_parameters(main_program)
        vars_list = param + opt
        return save_vars_pir(
            main_program=main_program,
            dirname=dirname,
            vars=[var for var in vars_list if var.persistable],
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

        save_var_map = {}
        for v in vars:
            var = global_scope().find_var(v.name)
            # TODO(chenzhiyang): deal with RAW type and sparse
            if filename is None and save_to_memory is False:
                save_file_path = os.path.join(os.path.normpath(dirname), v.name)
                core.save_func(
                    var.get_tensor(), v.name, save_file_path, True, False
                )
            else:
                save_var_map[v.name] = var.get_tensor()

        if filename is not None or save_to_memory:
            save_var_list = []
            save_var_names = []
            for name in sorted(save_var_map.keys()):
                save_var_list.append(save_var_map[name])
                save_var_names.append(name)

            save_path = ''
            if save_to_memory is False:
                save_path = os.path.join(os.path.normpath(dirname), filename)

            core.save_combine_func(
                save_var_list,
                save_var_names,
                save_path,
                True,
                False,
                save_to_memory,
            )

        if save_to_memory:
            return global_scope().find_var(params_var_name).get_bytes()


def load_vars_pir(
    executor,
    dirname,
    main_program=None,
    vars=None,
    filename=None,
):
    """
    :api_attr: PIR Static Graph

    This API loads variables from files by C++ function.

    There are two ways to specify the variables to be loaded: the first way, set
    variables in a list and assign it to the `vars`; the second way, use the
    `predicate` function to select variables that make `predicate(variable) == True`.
    The first way has a higher priority.

    The `dirname` is used to specify the folder where to load variables.
    If variables were saved in separate files in the folder `dirname`,
    set `filename` None. If all variables were saved in a single file,
    use `filename` to specify it.

    Args:
        executor(Executor): The executor to create variables in scope.
        dirname(str): The folder where to load the variables.
        main_program(Program, optional): The program whose variables will be loaded.
                                    If it is None, the default main program will
                                    be used automatically.
                                    Default: None
        vars(list[Variable], optional): The list that contains all variables to be loaded.
                                   Default: None
        filename(str, optional): The file which saved all required variables. If variables
                                were saved in separate files, set it to be None.
                                Default: None

    Returns:
        None
    """
    assert executor is None or isinstance(executor, Executor)

    vars_from_memory = False
    if dirname is not None:
        dirname = os.path.normpath(dirname)
    # TODO(chenzhiyang): vars_from_memory

    if filename == '':
        filename = None

    if vars is None:
        if main_program is None:
            main_program = default_main_program()

        if not isinstance(main_program, paddle.static.Program):
            raise TypeError(
                f"The type of input main_program is invalid, expected type is paddle.static.Program, but received {type(main_program)}"
            )
        param, opt = get_pir_parameters(main_program)
        vars = param + opt
        paddle.base.libpaddle.pir.create_loaded_parameter(
            vars, global_scope(), executor._default_executor
        )
        load_vars_pir(
            executor,
            dirname=dirname,
            main_program=main_program,
            vars=[var for var in vars if var.persistable],
            filename=filename,
        )
    else:
        if main_program is None:
            main_program = default_main_program()

        if not isinstance(main_program, paddle.static.Program):
            raise TypeError(
                f"The type of input main_program is invalid, expected type is paddle.static.Program, but received {type(main_program)}"
            )

        # TODO(chenzhiyang):save origin param shape, check vars
        load_var_map = {}

        for v in vars:
            var = global_scope().find_var(v.name)
            assert isinstance(var, paddle.base.libpaddle.Variable)
            if filename is None:
                if dirname is None:
                    raise ValueError(
                        "The directory path and params cannot be None at the same time."
                    )
                file_path = os.path.join(dirname, v.name)
                core.load_func(
                    file_path,
                    -1,
                    [],
                    False,
                    var.get_tensor(),
                    executor._default_executor.get_place(),
                )
            else:
                load_var_map[v.name] = var

        if filename is not None:
            load_var_list = []
            load_var_names = []
            for name in sorted(load_var_map.keys()):
                load_var_list.append(load_var_map[name].get_tensor())
                load_var_names.append(name)

            if vars_from_memory is False:
                filename = os.path.join(dirname, filename)

            core.load_combine_func(
                filename,
                load_var_names,
                load_var_list,
                False,
                executor._default_executor.get_place(),
            )
            for name, var in zip(load_var_names, load_var_list):
                set_var(name, np.array(var))


@static_only
def save_pir(program, model_path, protocol=4, **configs):
    """
    This function saves parameters, optimizer information and network description to model_path.

    The parameters contain all the trainable Tensor, and save to a file with suffix ".pdparams".
    The optimizer information contains all the Tensor used by optimizer. For Adam optimizer, contains beta1, beta2, momentum etc. All the information will be saved to a file with suffix ".pdopt". (If the optimizer has no Tensor to save (like SGD), the file will not be generated).
    The network description is the description of the program. It's only used for deployment. The description will be saved to a file with a suffix ".pdmodel".

    Args:
        program(Program) : The program to be saved.
        model_path(str): The file prefix to save the program. The format is "dirname/file_prefix". If file_prefix is an empty str, an exception will be raised.
        protocol(int, optional): The protocol version of pickle module must be greater than 1 and less than 5.
                                 Default: 4
        configs(dict, optional) : Optional keyword arguments.

    Returns:
        None
    """

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

    # get parameters and optimizer variables
    parameter_list, optimizer_param_list = get_pir_parameters(program)
    param_dict = {
        var.name: get_tensor(var) for var in parameter_list if var.persistable
    }
    opt_dict = {
        var.name: get_tensor(var)
        for var in optimizer_param_list
        if var.persistable
    }

    # save parameters
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

    # save optimizer parameters
    with open(model_path + ".pdopt", 'wb') as f:
        pickle.dump(opt_dict, f, protocol=protocol)

    # save program
    paddle.core.serialize_pir_program(
        program, model_path + ".json", 1, True, False, True
    )


@static_only
def load_pir(program, model_prefix, executor=None, var_list=None):
    """
    :api_attr: PIR Static Graph

    This function gets parameters and optimizer information from program, and then gets corresponding value from file.
    An exception will be thrown if shape or dtype of the parameters does not match.

    This function can also load model file saved with [ save_params, save_persistables, save_vars ].
    var_list can not be None when loading a single model file
    ( filename is not None when save_params, save_persistables or save_vars is called ).

    Args:
        program(Program): The program to be loaded
        model_prefix(str): The file prefix to store the program
        executor(Executor, optional): The executor used for initializing the parameter
                                      when startup program is not run.
        var_list(list|tuple, optional): The Tensor list/tuple to load a single model file saved with
                                  [ save_params, save_persistables, save_vars ].
                                  Default: None

    Returns:
        None
    """

    assert executor is None or isinstance(executor, Executor)

    parameter_file_name = model_prefix + ".pdparams"

    # TODO(chenzhiyang):if not os.path.exists(parameter_file_name): load_vars

    parameter_list, optimizer_param_list = get_pir_parameters(program)

    with open(parameter_file_name, 'rb') as f:
        # When value of dict is lager than 4GB ,there is a Bug on 'MAC python3'
        if sys.platform == 'darwin' and sys.version_info.major == 3:
            load_dict = _pickle_loads_mac(parameter_file_name, f)
        else:
            load_dict = _safe_load_pickle(f, encoding='latin1')
        load_dict = _pack_loaded_dict(load_dict)
    for var in parameter_list:
        if var.persistable:
            assert (
                var.name in load_dict
            ), f"Can not find [{var.name}] in model file [{parameter_file_name}]"
            set_var(var.name, load_dict[var.name])

    if len(optimizer_param_list) > 0:
        opt_file_name = model_prefix + ".pdopt"
        assert os.path.exists(
            opt_file_name
        ), f"Optimizer file [{opt_file_name}] not exits"

        if executor:
            paddle.base.libpaddle.pir.create_loaded_parameter(
                optimizer_param_list, global_scope(), executor._default_executor
            )

        with open(opt_file_name, 'rb') as f:
            load_dict = _safe_load_pickle(f, encoding='latin1')
        for var in optimizer_param_list:
            if var.persistable:
                assert (
                    var.name in load_dict
                ), f"Can not find [{var.name}] in model file [{opt_file_name}]"
                set_var(var.name, load_dict[var.name])


@static_only
def save_inference_model_pir(
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
    """
    # check path_prefix, set model_path and params_path
    path_prefix = _normalize_path_prefix(path_prefix)
    try:
        # mkdir may conflict if pserver and trainer are running on the same machine
        dirname = os.path.dirname(path_prefix)
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    model_path = path_prefix + ".json"
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
    # serialize and save program
    program = normalize_pir_program(
        program,
        feed_vars,
        fetch_vars,
        skip_prune_program=kwargs.get('skip_prune_program', False),
    )

    readable = kwargs.get('readable', False)
    trainable = kwargs.get('trainable', True)
    paddle.core.serialize_pir_program(
        program,
        (
            os.path.join(os.path.dirname(model_path), "__model__.json")
            if kwargs.get('separate_parameters', False)
            else model_path
        ),
        1,
        True,
        readable,
        trainable,
    )

    # serialize and save params
    save_dirname = os.path.dirname(params_path)
    params_filename = os.path.basename(params_path)
    save_vars_pir(
        dirname=save_dirname,
        main_program=program,
        filename=(
            None
            if kwargs.get('separate_parameters', False)
            else params_filename
        ),
    )


@static_only
def load_inference_model_pir(path_prefix, executor, **kwargs):
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
            >>> tensor_img = np.array(np.random.random((64, 784)), dtype=np.float32) # type: ignore[var-annotated]
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

        # deserialize bytes to program
        program = paddle.static.Program()
        paddle.base.core.deserialize_pir_program(model_filename, program, 1)

        params, opts = get_pir_parameters(program)
        vars = params + opts
        vars = [var for var in vars if var.persistable]
        if len(vars) > 0:
            load_vars_pir(
                # load from memory, dirname is None
                executor,
                dirname=None,
                main_program=program,
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
            model_path = path_prefix + ".json"
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
                model_path = os.path.join(path_prefix, model_filename + ".json")

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

        # deserialize bytes to program
        program = paddle.static.Program()
        paddle.base.core.deserialize_pir_program(model_path, program, 1)
        # load parameters
        params, opts = get_pir_parameters(program)
        vars = params + opts
        vars = [var for var in vars if var.persistable]
        if len(vars) > 0:
            load_dirname = os.path.dirname(params_path)
            params_filename = os.path.basename(params_path)

            load_vars_pir(
                executor,
                dirname=load_dirname,
                main_program=program,
                filename=params_filename,
            )

    feed_names, fetch_targets = get_pir_feed_and_fetch(program)
    return [program, feed_names, fetch_targets]
