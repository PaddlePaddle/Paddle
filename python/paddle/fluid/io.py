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

from __future__ import print_function

import os
import errno
import warnings
import time
import shutil
import six
from functools import reduce

from paddle.fluid import layers
from paddle.fluid.executor import Executor
from paddle.fluid.evaluator import Evaluator
from paddle.fluid.framework import Program, Parameter, default_main_program, default_startup_program, Variable, program_guard
from . import core

__all__ = [
    'save_vars', 'save_params', 'save_persistables', 'load_vars', 'load_params',
    'load_persistables', 'save_inference_model', 'load_inference_model'
]


def is_parameter(var):
    """
    Check whether the given variable is an instance of Parameter.

    Args:
        var(Variable): The variable to be checked.

    Returns:
        bool: True if the given `var` is an instance of Parameter,
        False if not.

    Examples:
        .. code-block:: python

            param = fluid.default_main_program().global_block().var('fc.w')
            res = fluid.io.is_parameter(param)
    """
    return isinstance(var, Parameter)


def is_persistable(var):
    """
    Check whether the given variable is persistable.

    Args:
        var(Variable): The variable to be checked.

    Returns:
        bool: True if the given `var` is persistable
        False if not.

    Examples:
        .. code-block:: python

            param = fluid.default_main_program().global_block().var('fc.b')
            res = fluid.io.is_persistable(param)
    """
    if var.desc.type() == core.VarDesc.VarType.FEED_MINIBATCH or \
            var.desc.type() == core.VarDesc.VarType.FETCH_LIST or \
            var.desc.type() == core.VarDesc.VarType.READER:
        return False
    return var.persistable


def _clone_var_in_block_(block, var):
    assert isinstance(var, Variable)
    return block.create_var(
        name=var.name,
        shape=var.shape,
        dtype=var.dtype,
        type=var.type,
        lod_level=var.lod_level,
        persistable=True)


def save_vars(executor,
              dirname,
              main_program=None,
              vars=None,
              predicate=None,
              filename=None):
    """
    Save variables to the given directory by executor.

    There are two ways to specify variables to be saved: The first way, list
    variables in a list and assign it to the `vars`. The second way, assign the
    `main_program` with an existing program, then all variables in the program
    will be saved. The first way has a higher priority. In other words, if `vars`
    are assigned, the `main_program` and the `predicate` will be ignored.

    The `dirname` are used to specify the folder where to save variables.
    If you prefer to save variables in separate files in the folder `dirname`,
    set `filename` None; if you prefer to save all variables in a single file,
    use `filename` to specify it.

    Args:
        executor(Executor): The executor to run for saving variables.
        dirname(str): The directory path.
        main_program(Program|None): The program whose variables will be saved.
                                    If it is None, the default main program will
                                    be used automatically.
                                    Default: None
        vars(list[Variable]|None): The list that contains all variables to save.
                                   It has a higher priority than the `main_program`.
                                   Default: None
        predicate(function|None): If it is not None, only variables in the
                                  `main_program` that makes predicate(variable)==True
                                  will be saved. It only works when we are using the
                                  `main_program` to specify variables (In other words
                                  `vars` is None).
                                  Default: None
        filename(str|None): The file which to save all variables. If you prefer to save
                            variables separately, set it to None.
                            Default: None

    Returns:
        None

    Raises:
        TypeError: If `main_program` is not an instance of Program nor None.

    Examples:
        .. code-block:: python

            exe = fluid.Executor(fluid.CPUPlace())
            param_path = "./my_paddle_model"

            # The first usage: using `main_program` to specify variables
            def name_has_fc(var):
                res = "fc" in var.name
                return res

            prog = fluid.default_main_program()
            fluid.io.save_vars(executor=exe, dirname=path, main_program=prog,
                               vars=None, predicate = name_has_fc)
            # All variables in `main_program` whose name includes "fc" will be saved.
            # And variables are going to be saved separately.


            # The second usage: using `vars` to specify variables
            var_list = [var_a, var_b, var_c]
            fluid.io.save_vars(executor=exe, dirname=path, vars=var_list,
                               filename="vars_file")
            # var_a, var_b and var_c will be saved. And they are going to be
            # saved in the same file named 'var_file' in the path "./my_paddle_model".
    """
    if vars is None:
        if main_program is None:
            main_program = default_main_program()
        if not isinstance(main_program, Program):
            raise TypeError("program should be as Program type or None")

        save_vars(
            executor,
            main_program=main_program,
            dirname=dirname,
            vars=list(filter(predicate, main_program.list_vars())),
            filename=filename)
    else:
        save_program = Program()
        save_block = save_program.global_block()

        if main_program is None:
            main_program = default_main_program()
        if not isinstance(main_program, Program):
            raise TypeError("program should be as Program type or None")

        save_var_map = {}
        for each_var in vars:
            # NOTE: don't save the variable which type is RAW
            if each_var.type == core.VarDesc.VarType.RAW:
                continue
            new_var = _clone_var_in_block_(save_block, each_var)
            if filename is None:
                save_block.append_op(
                    type='save',
                    inputs={'X': [new_var]},
                    outputs={},
                    attrs={'file_path': os.path.join(dirname, new_var.name)})
            else:
                save_var_map[new_var.name] = new_var

        if filename is not None:
            save_var_list = []
            for name in sorted(save_var_map.keys()):
                save_var_list.append(save_var_map[name])

            save_block.append_op(
                type='save_combine',
                inputs={'X': save_var_list},
                outputs={},
                attrs={'file_path': os.path.join(dirname, filename)})

        executor.run(save_program)


def save_params(executor, dirname, main_program=None, filename=None):
    """
    This function filters out all parameters from the give `main_program`
    and then save them to the folder `dirname` or the file `filename`.

    Use the `dirname` to specify the saving folder. If you would like to
    save parameters in separate files, set `filename` None; if you would
    like to save all parameters in a single file, use `filename` to specify
    the file name.

    NOTICE: Some variables are not Parameter while they are necessary for
    training. So you can NOT save and continue your training just by
    `save_params()` and `load_params()`. Please use `save_persistables()`
    and `load_persistables()` instead.

    Args:
        executor(Executor): The executor to run for saving parameters.
        dirname(str): The saving directory path.
        main_program(Program|None): The program whose parameters will be
                                    saved. If it is None, the default
                                    main program will be used automatically.
                                    Default: None
        filename(str|None): The file to save all parameters. If you prefer
                            to save parameters in differnet files, set it
                            to None.
                            Default: None

    Returns:
        None

    Examples:
        .. code-block:: python

            exe = fluid.Executor(fluid.CPUPlace())
            param_path = "./my_paddle_model"
            prog = fluid.default_main_program()
            fluid.io.save_params(executor=exe, dirname=param_path,
                                 main_program=None)
    """
    save_vars(
        executor,
        dirname=dirname,
        main_program=main_program,
        vars=None,
        predicate=is_parameter,
        filename=filename)


def _save_distributed_persistables(executor, dirname, main_program):
    """
    save_persistables for distributed training.
    the method will do things listed below:
    1.save part of persistable variables on trainer.
    2.receive "remote prefetch variables" from parameter servers and merge them.
    3.save "distributed lookup table" on parameter servers.
    4.receive "optimizer variables" from parameter servers and merge them.

    Args:
        executor(Executor): The executor to run for saving parameters.
        dirname(str): The saving directory path.
        main_program(Program): The program whose parameters will be
                            saved. the main_program must be the trainer_program
                            get after transpiler.

    Returns:
        None

    Examples:
        .. code-block:: python

            exe = fluid.Executor(fluid.CPUPlace())
            param_path = "./my_paddle_model"
            t = distribute_transpiler.DistributeTranspiler()
            t.transpile(...)
            train_program = t.get_trainer_program()
            _save_distributed_persistables(executor=exe, dirname=param_path, main_program=train_program)
    """

    def __save_remote_params(executor, dirname, remote_params_map):
        """
        recive params on pserver through rpc.
        if the params are be sliced, will concat them to one, then save it.
        """
        if not remote_params_map:
            return

        prog = Program()
        block = prog.global_block()

        # recv optimize vars from pserver
        for name, remote_params in remote_params_map.items():
            origin_var = None
            is_slice = False
            slice_vars = [0] * len(remote_params)
            slice_var_names = [""] * len(remote_params)
            endpoints = [""] * len(remote_params)

            for idx, optimizer in enumerate(remote_params):
                origin = optimizer.origin
                slice = optimizer.slice
                is_slice = optimizer.is_slice
                block_id = optimizer.block_id
                endpoint = optimizer.endpoint

                if idx == 0:
                    origin_var = block.create_var(
                        name=origin.name,
                        type=origin.type,
                        shape=origin.shape,
                        dtype=origin.dtype,
                        persistable=True)

                slice_var = block.create_var(
                    name="{}.slice.{}".format(slice.name, idx),
                    type=slice.type,
                    shape=slice.shape,
                    dtype=slice.dtype,
                    persistable=True)

                index = block_id if is_slice else idx
                slice_vars[index] = slice_var
                slice_var_names[index] = slice.name
                endpoints[index] = endpoint

            if is_slice:
                block.append_op(
                    type='recv',
                    inputs={"X": []},
                    outputs={"Out": slice_vars},
                    attrs={
                        "epmap": endpoints,
                        "with_barrier": False,
                        "varnames": slice_var_names,
                        "sync_mode": True
                    })
                block.append_op(
                    type='concat',
                    inputs={'X': slice_vars},
                    outputs={'Out': origin_var},
                    attrs={})
            else:
                block.append_op(
                    type='recv',
                    inputs={"X": []},
                    outputs={"Out": [origin_var]},
                    attrs={
                        "epmap": endpoints[:1],
                        "with_barrier": False,
                        "varnames": slice_var_names,
                        "sync_mode": True
                    })
            block.append_op(
                type='save',
                inputs={'X': [origin_var]},
                outputs={},
                attrs={'file_path': os.path.join(dirname, origin_var.name)})
            block.append_op(type='delete_var', inputs={'X': slice_vars})
        executor.run(prog)

    def __save_distributed_lookup_tables(executor, dirname,
                                         distributed_lookup_table, endpoints):
        """
        because the distributed lookup table may too huge to merge and save at one place,
        it will be saved at parameter server independent respectively.

        the save directory is dirname/"__lookup_table__".

        """
        prog = Program()
        block = prog.global_block()

        # if there is lookup table, the trainer 0 will notify all pserver to save.
        lookup_table_filename = os.path.join(dirname, "__lookup_table__")
        attrs = {}
        attrs['epmap'] = endpoints
        attrs['dir'] = lookup_table_filename
        attrs['lookup_table'] = distributed_lookup_table
        block.append_op(
            type='checkpoint_notify', inputs={}, outputs={}, attrs=attrs)
        executor.run(prog)

    def __exclude_vars(exclude_var_names=[]):
        def is_valid(var):
            if var.name in exclude_var_names:
                return False
            if var.desc.type() == core.VarDesc.VarType.FEED_MINIBATCH or \
                        var.desc.type() == core.VarDesc.VarType.FETCH_LIST or \
                        var.desc.type() == core.VarDesc.VarType.READER:
                return False
            return var.persistable

        return is_valid

    if not isinstance(main_program, Program):
        raise ValueError("'main_program' should be an instance of Program.")

    if not main_program._is_distributed:
        raise ValueError(
            "'_save_distributed_persistables' just be designed for distributed training."
        )

    remote_params_map = main_program._parameters_on_pservers.get_distributed_vars_by_vtypes(
        ["Optimizer", "RemotePrefetch"], groupby=True)

    exclude_var_names = []
    if remote_params_map:
        exclude_var_names.extend(remote_params_map.keys())

    if main_program._distributed_lookup_table:
        if isinstance(main_program._distributed_lookup_table, list):
            exclude_var_names.extend(main_program._distributed_lookup_table)
        else:
            exclude_var_names.append(main_program._distributed_lookup_table)

    local_vars = list(
        filter(__exclude_vars(exclude_var_names), main_program.list_vars()))
    save_vars(
        executor, main_program=main_program, dirname=dirname, vars=local_vars)

    if main_program._is_chief:
        if remote_params_map:
            __save_remote_params(executor, dirname, remote_params_map)
        if main_program._distributed_lookup_table:
            __save_distributed_lookup_tables(
                executor, dirname, main_program._distributed_lookup_table,
                main_program._endpoints)


def save_persistables(executor, dirname, main_program=None, filename=None):
    """
    This function filters out all variables with `persistable==True` from the
    give `main_program` and then saves these variables to the folder `dirname`
    or file `filename`.

    The `dirname` is used to specify the folder where persistable variables
    are going to be saved. If you would like to save variables in separate
    files, set `filename` None; if you would like to save all variables in a
    single file, use `filename` to specify the file name.

    Args:
        executor(Executor): The executor to run for saving persistable variables.
        dirname(str): The directory path.
        main_program(Program|None): The program whose persistbale variables will
                                    be saved. If it is None, the default main
                                    program will be used automatically.
                                    Default: None
        filename(str|None): The file to saved all variables. If you prefer to
                            save variables in differnet files, set it to None.
                            Default: None

    Returns:
        None

    Examples:
        .. code-block:: python

            exe = fluid.Executor(fluid.CPUPlace())
            param_path = "./my_paddle_model"
            # `prog` can be a program defined by the user
            prog = fluid.default_main_program()
            fluid.io.save_persistables(executor=exe, dirname=param_path,
                                       main_program=prog)
    """

    if main_program and main_program._is_distributed:
        _save_distributed_persistables(
            executor, dirname=dirname, main_program=main_program)

    else:
        save_vars(
            executor,
            dirname=dirname,
            main_program=main_program,
            vars=None,
            predicate=is_persistable,
            filename=filename)


def load_vars(executor,
              dirname,
              main_program=None,
              vars=None,
              predicate=None,
              filename=None):
    """
    Load variables from the given directory by executor.

    There are two ways to specify variables to be loaded: The first way, list
    variables in a list and assign it to the `vars`. The second way, assign the
    `main_program` with an existing program, then all variables in the program
    will be loaded. The first way has a higher priority. In other words if `vars`
    are assigned, the `main_program` and the `predicate` will be ignored.

    The `dirname` are used to specify the folder where to load variables.
    If variables were saved in separate files in the folder `dirname`,
    set `filename` None; if all variables were saved in a single file,
    use `filename` to specify it.

    Args:
        executor(Executor): The executor to run for loading variables.
        dirname(str): The directory path.
        main_program(Program|None): The program whose variables will be loaded.
                                    If it is None, the default main program will
                                    be used automatically.
                                    Default: None
        vars(list[Variable]|None): The list that contains all variables to load.
                                   It has a higher priority than the `main_program`.
                                   Default: None
        predicate(function|None): If it is not None, only variables in the
                                  `main_program` that makes predicate(variable)==True
                                  will be loaded. It only works when we are using the
                                  `main_program` to specify variables (In other words
                                  `vars` is None).
                                  Default: None
        filename(str|None): The file which saved all required variables. If variables
                            were saved in differnet files, set it to None.
                            Default: None

    Returns:
        None

    Raises:
        TypeError: If `main_program` is not an instance of Program nor None.

    Examples:
        .. code-block:: python

            exe = fluid.Executor(fluid.CPUPlace())
            param_path = "./my_paddle_model"

            # The first usage: using `main_program` to specify variables
            def name_has_fc(var):
                res = "fc" in var.name
                return res

            prog = fluid.default_main_program()
            fluid.io.load_vars(executor=exe, dirname=path, main_program=prog,
                               vars=None, predicate=name_has_fc)
            # All variables in `main_program` whose name includes "fc" will be loaded.
            # And all the variables are supposed to have been saved in differnet files.


            # The second usage: using `vars` to specify variables
            var_list = [var_a, var_b, var_c]
            fluid.io.load_vars(executor=exe, dirname=path, vars=var_list,
                               filename="vars_file")
            # var_a, var_b and var_c will be loaded. And they are supposed to haven
            # been saved in the same file named 'var_file' in the path "./my_paddle_model".
    """
    if vars is None:
        if main_program is None:
            main_program = default_main_program()
        if not isinstance(main_program, Program):
            raise TypeError("program's type should be Program")

        load_vars(
            executor,
            dirname=dirname,
            main_program=main_program,
            vars=list(filter(predicate, main_program.list_vars())),
            filename=filename)
    else:
        load_prog = Program()
        load_block = load_prog.global_block()

        if main_program is None:
            main_program = default_main_program()
        if not isinstance(main_program, Program):
            raise TypeError("program should be as Program type or None")

        load_var_map = {}
        for each_var in vars:
            assert isinstance(each_var, Variable)
            if each_var.type == core.VarDesc.VarType.RAW:
                continue
            new_var = _clone_var_in_block_(load_block, each_var)
            if filename is None:
                load_block.append_op(
                    type='load',
                    inputs={},
                    outputs={'Out': [new_var]},
                    attrs={'file_path': os.path.join(dirname, new_var.name)})
            else:
                load_var_map[new_var.name] = new_var

        if filename is not None:
            load_var_list = []
            for name in sorted(load_var_map.keys()):
                load_var_list.append(load_var_map[name])

            load_block.append_op(
                type='load_combine',
                inputs={},
                outputs={"Out": load_var_list},
                attrs={'file_path': os.path.join(dirname, filename)})
        executor.run(load_prog)


def load_params(executor, dirname, main_program=None, filename=None):
    """
    This function filters out all parameters from the give `main_program`
    and then trys to load these parameters from the folder `dirname` or
    the file `filename`.

    Use the `dirname` to specify the folder where parameters were saved. If
    parameters were saved in separate files in the folder `dirname`, set
    `filename` None; if all parameters were saved in a single file, use
    `filename` to specify the file name.

    NOTICE: Some variables are not Parameter while they are necessary for
    training. So you can NOT save and continue your training just by
    `save_params()` and `load_params()`. Please use `save_persistables()`
    and `load_persistables()` instead.

    Args:
        executor(Executor): The executor to run for loading parameters.
        dirname(str): The directory path.
        main_program(Program|None): The program whose parameters will be
                                    loaded. If it is None, the default
                                    main program will be used automatically.
                                    Default: None
        filename(str|None): The file which saved all parameters. If parameters
                            were saved in differnet files, set it to None.
                            Default: None

    Returns:
        None

    Examples:
        .. code-block:: python

            exe = fluid.Executor(fluid.CPUPlace())
            param_path = "./my_paddle_model"
            prog = fluid.default_main_program()
            fluid.io.load_params(executor=exe, dirname=param_path,
                                main_program=None)
    """
    load_vars(
        executor,
        dirname=dirname,
        main_program=main_program,
        predicate=is_parameter,
        filename=filename)


def load_persistables(executor, dirname, main_program=None, filename=None):
    """
    This function filters out all variables with `persistable==True` from the
    give `main_program` and then trys to load these variables from the folder
    `dirname` or the file `filename`.

    Use the `dirname` to specify the folder where persistable variables were
    saved. If variables were saved in separate files, set `filename` None;
    if all variables were saved in a single file, use `filename` to specify
    the file name.

    Args:
        executor(Executor): The executor to run for loading persistable variables.
        dirname(str): The directory path.
        main_program(Program|None): The program whose persistbale variables will
                                    be loaded. If it is None, the default main
                                    program will be used automatically.
                                    Default: None
        filename(str|None): The file which saved all variables. If variables were
                            saved in differnet files, set it to None.
                            Default: None

    Returns:
        None

    Examples:
        .. code-block:: python

            exe = fluid.Executor(fluid.CPUPlace())
            param_path = "./my_paddle_model"
            prog = fluid.default_main_program()
            fluid.io.load_persistables(executor=exe, dirname=param_path,
                                       main_program=None)
    """

    if main_program and main_program._is_distributed:
        _load_distributed_persistables(
            executor, dirname=dirname, main_program=main_program)
    else:
        load_vars(
            executor,
            dirname=dirname,
            main_program=main_program,
            predicate=is_persistable,
            filename=filename)


def _load_distributed_persistables(executor, dirname, main_program=None):
    """
    customized load_persistables for distributed training.
    it should be used on parameter server,

    Args:
        executor(Executor): The executor to run for saving parameters.
        dirname(str): The load directory path.
        main_program(Program): The program whose parameters will be
                            loaded. the main_program must be the pserver_program
                            get after transpiler.

    Returns:
        None

    Examples:
        .. code-block:: python

            exe = fluid.Executor(fluid.CPUPlace())
            param_path = "./my_paddle_model"
            t = distribute_transpiler.DistributeTranspiler()
            t.transpile(...)
            pserver_prog = t.get_pserver_program(...)
            _load_distributed_persistables(executor=exe, dirname=param_path, main_program=pserver_prog)
    """

    def __is_distributed_part_var(varname):
        trainer_idx = varname.find(".trainer_")
        block_idx = varname.find(".block")
        return trainer_idx or block_idx

    def __load_persistable_vars(executor, dirname, need_load_vars):
        load_prog = Program()
        load_block = load_prog.global_block()
        need_delete_vars = []

        for param in need_load_vars:
            origin_var = param.origin
            slice_var = param.slice
            is_slice = param.is_slice
            offset = param.offset

            if is_slice:
                origin = load_block.create_var(
                    name="{}.load".format(origin_var.name),
                    type=origin_var.type,
                    shape=origin_var.shape,
                    dtype=origin_var.dtype,
                    persistable=True)

                load_block.append_op(
                    type='load',
                    inputs={},
                    outputs={'Out': [origin]},
                    attrs={
                        'file_path': os.path.join(dirname, origin_var.name)
                    })

                slice = load_block.create_var(
                    name=slice_var.name,
                    type=slice_var.type,
                    shape=slice_var.shape,
                    dtype=slice_var.dtype,
                    persistable=True)

                dim1_flatten = 1
                if len(slice.shape) >= 2:
                    dim1_flatten = reduce(lambda x, y: x * y, slice.shape[1:])

                start = int(offset / dim1_flatten)
                end = int(offset / dim1_flatten + slice.shape[0])

                load_block.append_op(
                    type="slice",
                    inputs={'Input': origin},
                    outputs={'Out': slice},
                    attrs={'axes': [0],
                           'starts': [start],
                           'ends': [end]})

                need_delete_vars.append(origin)
            else:
                origin = load_block.create_var(
                    name="{}".format(origin_var.name),
                    type=origin_var.type,
                    shape=origin_var.shape,
                    dtype=origin_var.dtype,
                    persistable=True)
                load_block.append_op(
                    type='load',
                    inputs={},
                    outputs={'Out': [origin]},
                    attrs={
                        'file_path': os.path.join(dirname, origin_var.name)
                    })

        load_block.append_op(
            type='delete_var',
            inputs={'X': need_delete_vars}, )

        executor.run(load_prog)

    if not isinstance(main_program, Program):
        raise ValueError("'main_program' should be an instance of Program.")

    if not main_program._is_distributed:
        raise ValueError(
            "'_load_distributed_persistables' just be designed for distributed training."
        )

    if not main_program._ps_endpoint:
        raise ValueError(
            "'_load_distributed_persistables' need current_endpoint set in DistributeTranspiler.transpile"
        )

    need_load_vars = main_program._parameters_on_pservers.get_distributed_vars_by_ep(
        main_program._ps_endpoint)
    __load_persistable_vars(executor, dirname, need_load_vars)


def prepend_feed_ops(inference_program,
                     feed_target_names,
                     feed_holder_name='feed'):
    if len(feed_target_names) == 0:
        return

    global_block = inference_program.global_block()
    feed_var = global_block.create_var(
        name=feed_holder_name,
        type=core.VarDesc.VarType.FEED_MINIBATCH,
        persistable=True)

    for i, name in enumerate(feed_target_names):
        out = global_block.var(name)
        global_block._prepend_op(
            type='feed',
            inputs={'X': [feed_var]},
            outputs={'Out': [out]},
            attrs={'col': i})


def append_fetch_ops(inference_program,
                     fetch_target_names,
                     fetch_holder_name='fetch'):
    global_block = inference_program.global_block()
    fetch_var = global_block.create_var(
        name=fetch_holder_name,
        type=core.VarDesc.VarType.FETCH_LIST,
        persistable=True)

    for i, name in enumerate(fetch_target_names):
        global_block.append_op(
            type='fetch',
            inputs={'X': [name]},
            outputs={'Out': [fetch_var]},
            attrs={'col': i})


def save_inference_model(dirname,
                         feeded_var_names,
                         target_vars,
                         executor,
                         main_program=None,
                         model_filename=None,
                         params_filename=None,
                         export_for_deployment=True):
    """
    Prune the given `main_program` to build a new program especially for inference,
    and then save it and all related parameters to given `dirname` by the `executor`.

    Args:
        dirname(str): The directory path to save the inference model.
        feeded_var_names(list[str]): Names of variables that need to be feeded data
                                     during inference.
        target_vars(list[Variable]): Variables from which we can get inference
                                     results.
        executor(Executor): The executor that saves the inference model.
        main_program(Program|None): The original program, which will be pruned to
                                    build the inference model. If is setted None,
                                    the default main program will be used.
                                    Default: None.
        model_filename(str|None): The name of file to save the inference program
                                  itself. If is setted None, a default filename
                                  `__model__` will be used.
        params_filename(str|None): The name of file to save all related parameters.
                                   If it is setted None, parameters will be saved
                                   in separate files .
        export_for_deployment(bool): If True, programs are modified to only support
                                     direct inference deployment. Otherwise,
                                     more information will be stored for flexible
                                     optimization and re-training. Currently, only
                                     True is supported.

    Returns:
        target_var_name_list(list): The fetch variables' name list

    Raises:
        ValueError: If `feed_var_names` is not a list of basestring.
        ValueError: If `target_vars` is not a list of Variable.

    Examples:
        .. code-block:: python

            exe = fluid.Executor(fluid.CPUPlace())
            path = "./infer_model"
            fluid.io.save_inference_model(dirname=path, feeded_var_names=['img'],
                         target_vars=[predict_var], executor=exe)

            # In this exsample, the function will prune the default main program
            # to make it suitable for infering the `predict_var`. The pruned
            # inference program is going to be saved in the "./infer_model/__model__"
            # and parameters are going to be saved in separate files under folder
            # "./infer_model".

    """
    if isinstance(feeded_var_names, six.string_types):
        feeded_var_names = [feeded_var_names]
    elif export_for_deployment:
        if len(feeded_var_names) > 0:
            # TODO(paddle-dev): polish these code blocks
            if not (bool(feeded_var_names) and all(
                    isinstance(name, six.string_types)
                    for name in feeded_var_names)):
                raise ValueError("'feed_var_names' should be a list of str.")

    if isinstance(target_vars, Variable):
        target_vars = [target_vars]
    elif export_for_deployment:
        if not (bool(target_vars) and
                all(isinstance(var, Variable) for var in target_vars)):
            raise ValueError("'target_vars' should be a list of Variable.")

    if main_program is None:
        main_program = default_main_program()
        if main_program._is_mem_optimized:
            warnings.warn(
                "save_inference_model must put before you call memory_optimize. \
                                            the memory_optimize will modify the original program, \
                                            is not suitable for saving inference model \
                                            we save the original program as inference model.",
                RuntimeWarning)

    # fix the bug that the activation op's output as target will be pruned.
    # will affect the inference performance.
    # TODO(Superjomn) add an IR pass to remove 1-scale op.
    with program_guard(main_program):
        uniq_target_vars = []
        for i, var in enumerate(target_vars):
            if isinstance(var, Variable):
                var = layers.scale(
                    var, 1., name="save_infer_model/scale_{}".format(i))
            uniq_target_vars.append(var)
        target_vars = uniq_target_vars
    target_var_name_list = [var.name for var in target_vars]

    # when a pserver and a trainer running on the same machine, mkdir may conflict
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    if model_filename is not None:
        model_basename = os.path.basename(model_filename)
    else:
        model_basename = "__model__"
    model_basename = os.path.join(dirname, model_basename)

    # When export_for_deployment is true, we modify the program online so that
    # it can only be loaded for inference directly. If it's false, the whole
    # original program and related meta are saved so that future usage can be
    # more flexible.

    origin_program = main_program.clone()

    if export_for_deployment:
        main_program = main_program.clone()
        global_block = main_program.global_block()
        need_to_remove_op_index = []
        for i, op in enumerate(global_block.ops):
            op.desc.set_is_target(False)
            if op.type == "feed" or op.type == "fetch":
                need_to_remove_op_index.append(i)

        for index in need_to_remove_op_index[::-1]:
            global_block._remove_op(index)

        main_program.desc.flush()

        main_program = main_program._prune(targets=target_vars)
        main_program = main_program._inference_optimize(prune_read_op=True)
        fetch_var_names = [v.name for v in target_vars]

        prepend_feed_ops(main_program, feeded_var_names)
        append_fetch_ops(main_program, fetch_var_names)

        with open(model_basename, "wb") as f:
            f.write(main_program.desc.serialize_to_string())
    else:
        # TODO(panyx0718): Save more information so that it can also be used
        # for training and more flexible post-processing.
        with open(model_basename + ".main_program", "wb") as f:
            f.write(main_program.desc.serialize_to_string())

    main_program._copy_dist_param_info_from(origin_program)

    if params_filename is not None:
        params_filename = os.path.basename(params_filename)

    save_persistables(executor, dirname, main_program, params_filename)
    return target_var_name_list


def load_inference_model(dirname,
                         executor,
                         model_filename=None,
                         params_filename=None,
                         pserver_endpoints=None):
    """
    Load inference model from a directory

    Args:
        dirname(str): The directory path
        executor(Executor): The executor to run for loading inference model.
        model_filename(str|None): The name of file to load inference program.
                                  If it is None, the default filename
                                  '__model__' will be used.
                                  Default: None
        params_filename(str|None): The name of file to load all parameters.
                                   It is only used for the case that all
                                   parameters were saved in a single binary
                                   file. If parameters were saved in separate
                                   files, set it as 'None'.
        pserver_endpoints(list|None): This only need by distributed inference.
                                    When use distributed look up table in training,
                                    We also need it in inference.The parameter is
                                    a list of pserver endpoints.

    Returns:
        tuple: The return of this function is a tuple with three elements:
        (program, feed_target_names, fetch_targets). The `program` is a
        Program, it's the program for inference. The `feed_target_names` is
        a list of str, it contains Names of variables that need to feed
        data in the inference program. The `fetch_targets` is a list of
        Variable. It contains variables from which we can get inference
        results.

    Raises:
        ValueError: If `dirname` is not a existing directory.

    Examples:
        .. code-block:: python

            exe = fluid.Executor(fluid.CPUPlace())
            path = "./infer_model"
            endpoints = ["127.0.0.1:2023","127.0.0.1:2024"]
            [inference_program, feed_target_names, fetch_targets] =
                fluid.io.load_inference_model(dirname=path, executor=exe)
            results = exe.run(inference_program,
                          feed={feed_target_names[0]: tensor_img},
                          fetch_list=fetch_targets)

            # if we need lookup table, we will use:
            fluid.io.load_inference_model(dirname=path, executor=exe, pserver_endpoints=endpoints)

            # In this exsample, the inference program was saved in the
            # "./infer_model/__model__" and parameters were saved in
            # separate files in ""./infer_model".
            # After getting inference program, feed target names and
            # fetch targets, we can use an Executor to run the inference
            # program to get the inference result.

    """
    if not os.path.isdir(dirname):
        raise ValueError("There is no directory named '%s'", dirname)

    if model_filename is not None:
        model_filename = os.path.basename(model_filename)
    else:
        model_filename = "__model__"
    model_filename = os.path.join(dirname, model_filename)

    if params_filename is not None:
        params_filename = os.path.basename(params_filename)

    with open(model_filename, "rb") as f:
        program_desc_str = f.read()

    program = Program.parse_from_string(program_desc_str)
    if not core._is_program_version_supported(program._version()):
        raise ValueError("Unsupported program version: %d\n" %
                         program._version())
    # Binary data also need versioning.
    load_persistables(executor, dirname, program, params_filename)

    if pserver_endpoints:
        program = _endpoints_replacement(program, pserver_endpoints)

    feed_target_names = program.desc.get_feed_target_names()
    fetch_target_names = program.desc.get_fetch_target_names()
    fetch_targets = [
        program.global_block().var(name) for name in fetch_target_names
    ]

    return [program, feed_target_names, fetch_targets]


def _endpoints_replacement(program, endpoints):
    ENDPOINT_MAP = "epmap"
    for op in program.global_block().ops:
        if op.has_attr(ENDPOINT_MAP):
            op.set_attr(ENDPOINT_MAP, endpoints)
    program._sync_with_cpp()
    return program


def get_parameter_value(para, executor):
    """
    Get the LoDTensor value of the given parameter.

    Args:
        para(Parameter): The parameter to get value from.
        executor(Executor): The executor to run for retrieving the value.

    Returns:
        numpy.array: The given parameter's values.

    Raises:
        AssertionError: If the `para` is not an instance of Parameter.

    Examples:
        .. code-block:: python

            exe = fluid.Executor(fluid.CPUPlace())
            param = fluid.default_main_program().global_block().var('fc.w')
            p = fluid.io.get_parameter_value(param, exe)

    """
    assert is_parameter(para)

    get_program = Program()
    block = get_program.global_block()
    new_var = _clone_var_in_block_(block, para)
    return executor.run(get_program, feed={}, fetch_list=[new_var])[0]


def get_parameter_value_by_name(name, executor, program=None):
    """
    Get the LoDTensor value of a certain parameter by its name.

    Args:
        name(str): The parameter's name.
        executor(Executor): The executor to run for retrieving the value.
        program(Program | None): The program where to find the parameter.
                               If it's set to be None, the function will
                               try to find the parameter in the default
                               main program.

    Returns:
        numpy.array: The parameter's values.

    Raises:
        TypeError: If given `name` is not an instance of basestring.
        TypeError: If the parameter with the given name doesn't exist.
        AssertionError: If there is a varibale named `name` in the
                        given program but it is not a Parameter.

    Examples:
        .. code-block:: python

            exe = fluid.Executor(fluid.CPUPlace())
            p = fluid.io.get_parameter_value('fc.w', exe)
    """
    if program is None:
        program = default_main_program()
    var = program.global_block().var(name)
    return get_parameter_value(var, executor)
