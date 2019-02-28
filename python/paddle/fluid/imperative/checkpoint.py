# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.wrapped_decorator import signature_safe_contextmanager

import os
import contextlib
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.framework import Variable
from paddle.fluid.executor import Executor


def save_persistables(var_list, dirname, filename=None, place=None):
    """
    This function filters out all variables in layer.parameters from the
    give `layer` and then trys to load these variables from the folder
    `dirname` or the file `filename`.

    Use the `dirname` to specify the folder where persistable variables were
    saved. If variables were saved in separate files, set `filename` None;
    if all variables were saved in a single file, use `filename` to specify
    the file name.

    Args:
        executor(Executor): The executor to run for loading persistable variables.
        dirname(str): The directory path.
        filename(str|None): The file which saved all variables. If variables were
                            saved in differnet files, set it to None.
                            Default: None
        var_list(list of Parameters): The parameters will
                                    be saved. If it is None, nothing
                                    will be deal.


    Returns:

    Examples:
        .. code-block:: python
            ptb_model = PtbModel(
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_steps=num_steps,
                init_scale=init_scale)

            x_data = np.arange(12).reshape(4, 3).astype('int64')
            y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
            x_data = x_data.reshape((-1, num_steps, 1))
            y_data = y_data.reshape((-1, 1))
            init_hidden_data = np.zeros(
                (num_layers, batch_size, hidden_size), dtype='float32')
            init_cell_data = np.zeros(
                (num_layers, batch_size, hidden_size), dtype='float32')
            x = to_variable(x_data)
            y = to_variable(y_data)
            init_hidden = to_variable(init_hidden_data)
            init_cell = to_variable(init_cell_data)
            dy_loss, last_hidden, last_cell = ptb_model(x, y, init_hidden,
                                                        init_cell)
            param_path = "./my_paddle_model"
            fluid.imperative.checkpoint.save_persistables(ptb_model.parameters, dirname=param_path,
                                       layer=ptb_model, place=fluid.CPUPlace())
    """
    if var_list:
        _save_var_to_file(var_list, place, dirname, filename)


def load_persistables(var_list, dirname, filename=None, place=None):
    """
    This function trys to load persistable variables from the folder
    `dirname` or the file `filename`.

    Use the `dirname` to specify the folder where persistable variables were
    saved. If variables were saved in separate files, set `filename` None;
    if all variables were saved in a single file, use `filename` to specify
    the file name.

    Args:
        var_list(list of Parameters): The parameters will be loaded.
        dirname(str): The directory path.
        filename(str|None): The file which saved all variables, this file path should be end with '.npz'. If variables were
                            saved in differnet files, set it to None.
                            Default: None
        place():

    Returns:
        dict: The parameter-dict resumed from file

    Examples:
        .. code-block:: python
            param_path = "./my_paddle_model"

            param_dict = fluid.imperative.checkpoint.load_persistables(param_path, place=fluid.CPUPlace())
            param_1 = param_dict['PtbModel_0.w_1']

            or:
            param_path = "./my_paddle_model"
            filename = "model.file"
            param_dict = fluid.imperative.checkpoint.load_persistables(param_path, filename=filename,
                                                                       place=fluid.CPUPlace())
            param_1 = param_dict['PtbModel_0.w_1']

        """
    if var_list:
        return _load_var_from_file(var_list, place, dirname, filename)

    return {}


def _save_var_to_file(var_list, place, file_dir, file_name):
    with guard(place) as train_pro:
        with new_program_scope(
                main=train_pro[0], startup=train_pro[1]) as (prog, scope):
            exe = fluid.Executor(place)
            save_block = prog.global_block()
            save_var_map = {}
            for each_var in var_list:
                new_var = _clone_var_in_block_(save_block, each_var)
                save_var_map[new_var.name] = new_var
                if file_name is None:
                    save_block.append_op(
                        type='save',
                        inputs={'X': [new_var]},
                        outputs={},
                        attrs={
                            'file_path': os.path.join(file_dir, new_var.name)
                        })

            if file_name is not None:
                save_var_list = []
                for name in sorted(save_var_map.keys()):
                    save_var_list.append(save_var_map[name])

                save_block.append_op(
                    type='save_combine',
                    inputs={'X': save_var_list},
                    outputs={},
                    attrs={'file_path': os.path.join(file_dir, file_name)})
            exe.run(prog, feed=save_var_map)


def _load_var_from_file(var_list, place, file_dir, file_name):
    with guard(place) as (train_pro, startup_pro):
        with new_program_scope(
                main=train_pro, startup=startup_pro) as (prog, scope):
            exe = fluid.Executor(place)
            load_block = prog.global_block()
            load_var_map = {}
            load_var_list_fetch = []
            for each_var in var_list:
                assert isinstance(each_var, Variable)
                if each_var.type == core.VarDesc.VarType.RAW:
                    continue
                new_var = _clone_var_in_block_(load_block, each_var)
                load_var_list_fetch.append(new_var.name)
                if file_name is None:
                    load_block.append_op(
                        type='load',
                        inputs={},
                        outputs={'Out': [new_var]},
                        attrs={
                            'file_path': os.path.join(file_dir, new_var.name)
                        })
                else:
                    load_var_map[new_var.name] = new_var

            if file_name is not None:
                load_var_list = []
                for name in sorted(load_var_map.keys()):
                    load_var_list.append(load_var_map[name])

                load_block.append_op(
                    type='load_combine',
                    inputs={},
                    outputs={"Out": load_var_list},
                    attrs={'file_path': os.path.join(file_dir, file_name)})
            exe.run(prog)
            var_dict = {}
            for var_name in load_var_list_fetch:
                var_dict[var_name] = scope.find_var(each_var.name)
            return var_dict


@contextlib.contextmanager
def new_program_scope(main=None, startup=None, scope=None):
    # TODO fix me after supporting no-kernal op
    prog = main if main else fluid.Program()
    startup_prog = startup if startup else fluid.Program()
    scope = scope if scope else fluid.core.Scope()
    with fluid.scope_guard(scope):
        with fluid.program_guard(prog, startup_prog):
            with fluid.unique_name.guard():
                yield prog, scope


def _clone_var_in_block_(block, var):
    assert isinstance(var, Variable)
    return block.create_var(
        name=var.name,
        shape=var.shape,
        dtype=var.dtype,
        type=var.type,
        lod_level=var.lod_level,
        persistable=True)


@signature_safe_contextmanager
def guard(place=None):
    """
    fix me with supported no-kernal op
    :param place:
    :return:
    """
    train = fluid.framework.Program()
    startup = fluid.framework.Program()
    tracer = None

    if place is None:
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()

    with fluid.framework.program_guard(train, startup):
        with fluid.framework.unique_name.guard():
            with fluid.framework._imperative_guard(tracer):
                with fluid.framework._imperative_place_guard(place):
                    yield train, startup
