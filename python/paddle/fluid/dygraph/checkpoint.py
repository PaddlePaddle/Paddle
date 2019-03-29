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

import os
import collections
from .. import core
from ..framework import Variable, default_main_program

__all__ = ['save_persistables', 'load_persistables']


def save_persistables(vardict, dirname, filename=None):
    """
    This function filters out all variables in layer.parameters from the
    give `layer` and then trys to load these variables from the folder
    `dirname` or the file `filename`.

    Use the `dirname` to specify the folder where persistable variables were
    saved. If variables were saved in separate files, set `filename` None;
    if all variables were saved in a single file, use `filename` to specify
    the file name.

    Args:
        vardict(dict of Parameters): The parameters will
                                    be saved. If it is None, nothing
                                    will be deal.
        dirname(str): The directory path.
        filename(str|None): The file which saved all variables. If variables were
                            saved in differnet files, set it to None.
                            Default: None

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
            fluid.dygraph.save_persistables(ptb_model.state_dict(), dirname=param_path,
                                       layer=ptb_model)
    """
    if isinstance(vardict, collections.OrderedDict):
        _save_var_to_file(vardict, dirname, filename)


def load_persistables(vardict, dirname, filename=None):
    """
    This function trys to load persistable variables from the folder
    `dirname` or the file `filename`.

    Use the `dirname` to specify the folder where persistable variables were
    saved. If variables were saved in separate files, set `filename` None;
    if all variables were saved in a single file, use `filename` to specify
    the file name.

    Args:
        vardict(dict of Parameters): The parameters will be loaded.
        dirname(str): The directory path.
        filename(str|None): The file which saved all variables, this file path should be end with '.npz'. If variables were
                            saved in differnet files, set it to None.
                            Default: None

    Returns:
        dict: The parameter-dict resumed from file

    Examples:
        .. code-block:: python
            my_layer = layer(fluid.Layer)
            param_path = "./my_paddle_model"

            param_dict = fluid.dygraph.load_persistables(my_layer.parameters(), param_path)
            param_1 = param_dict['PtbModel_0.w_1']

        """
    if isinstance(vardict, collections.OrderedDict):
        return _load_var_from_file(vardict, dirname, filename)

    return {}


def _save_var_to_file(stat_dict, file_dir, file_name):
    save_block = default_main_program().global_block()
    save_var_map = {}
    for each_var in stat_dict.items():
        save_var_map[each_var.name] = each_var
        if file_name is None:
            save_block.append_op(
                type='save',
                inputs={'X': [each_var]},
                outputs={},
                attrs={'file_path': os.path.join(file_dir, each_var.name)})

    if file_name is not None:
        save_var_list = []
        for name in sorted(save_var_map.keys()):
            save_var_list.append(save_var_map[name])

        save_block.append_op(
            type='save_combine',
            inputs={'X': save_var_list},
            outputs={},
            attrs={'file_path': os.path.join(file_dir, file_name)})


def _load_var_from_file(stat_dict, file_dir, file_name):
    load_block = default_main_program().global_block()
    load_var_map = {}

    for each_var in stat_dict.items():
        assert isinstance(each_var, Variable)
        if each_var.type == core.VarDesc.VarType.RAW:
            continue
        new_var = _clone_var_in_block_(load_block, each_var)
        if file_name is None:
            load_block.append_op(
                type='load',
                inputs={},
                outputs={'Out': [new_var]},
                attrs={'file_path': os.path.join(file_dir, each_var.name)})

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
        for res_var in load_var_list:
            load_var_map[res_var.name] = res_var

    return load_var_map


def _clone_var_in_block_(block, var):
    assert isinstance(var, Variable)
    return block.create_var(
        name=var.name,
        shape=var.shape,
        dtype=var.dtype,
        type=var.type,
        lod_level=var.lod_level,
        persistable=True)
