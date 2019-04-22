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
import pickle
import warnings

__all__ = ['save_persistables', 'load_persistables']


def save_persistables(model_dict, optimizer, dirname, filename=None):
    """
    This function filters out all variables in layer.parameters from the
    give `layer` and then trys to load these variables from the folder
    `dirname` or the file `filename`.

    Use the `dirname` to specify the folder where persistable variables were
    saved. If variables were saved in separate files, set `filename` None;
    if all variables were saved in a single file, use `filename` to specify
    the file name.

    Args:
        model_dict(dict of Parameters): The parameters will
                                    be saved. If it is None, nothing
                                    will be deal.
        dirname(str): The directory path.
        filename(str|None): The file which saved all variables. If variables were
                            saved in different files, set it to None.
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
    if isinstance(model_dict, collections.OrderedDict):
        _save_var_to_file(model_dict, optimizer, dirname, filename)


def load_persistables(dirname):
    """
    This function trys to load persistable variables from the folder
    `dirname` or the file `filename`.

    Use the `dirname` to specify the folder where persistable variables were
    saved. If variables were saved in separate files, set `filename` None;
    if all variables were saved in a single file, use `filename` to specify
    the file name.

    Args:
        dirname(str): The directory path.
        optimizer(Optimizer): Optimizer to be save

    Returns:
        dict: The parameter-dict resumed from file

    Examples:
        .. code-block:: python
            my_layer = layer(fluid.Layer)
            param_path = "./my_paddle_model"

            param_dict = fluid.dygraph.load_persistables(my_layer.parameters(), param_path)
            param_1 = param_dict['PtbModel_0.w_1']

        """
    return _load_var_from_file(dirname)


def _save_var_to_file(stat_dict, optimizers, file_dir, file_name):
    save_block = default_main_program().global_block()
    save_var_map = {}
    for var_key, each_var in stat_dict.items():
        save_var_map[each_var.name] = each_var
        if file_name is None:
            save_block.append_op(
                type='save',
                inputs={'X': [each_var]},
                outputs={},
                attrs={
                    'file_path': os.path.join(file_dir,
                                              os.path.normpath(each_var.name))
                })
    if isinstance(optimizers, (list, tuple)):
        optimizers = optimizers
    else:
        optimizers = [optimizers]
    if os.path.exists(os.path.join(file_dir, os.path.normpath("optimizers"))):
        pass
    else:
        os.mkdir(os.path.join(file_dir, os.path.normpath("optimizers")))
    import learning_rate_scheduler
    for optimizer in optimizers:
        if isinstance(optimizer._learning_rate,
                      learning_rate_scheduler.LearningRateDecay):
            f = open(
                os.path.join(file_dir, "optimizers",
                             os.path.normpath(str(optimizer._name))), "wb")
            pickle.dump(optimizer._learning_rate, f, 2)
            f.close()
        else:
            warnings.warn(
                "Optimizer not saved, Only optimizer with 'LearningRateDecay' under DyGraph mode need to be saved"
            )

    if file_name is not None:
        save_var_list = []
        for name in sorted(save_var_map.keys()):
            save_var_list.append(save_var_map[name])

        save_block.append_op(
            type='save_combine',
            inputs={'X': save_var_list},
            outputs={},
            attrs={
                'file_path': os.path.join(file_dir, os.path.normpath(file_name))
            })


def _load_var_from_file(file_dir):
    def walk_filename(file_dir):
        base_path = os.path.join(file_dir)
        var_name_list = []
        if os.path.exists(base_path):
            for dirpath, dirnames, filenames in os.walk(base_path):
                if "optimizers" in dirpath:
                    continue
                pt = dirpath.replace(base_path, "", 1)
                if pt.startswith("/") or pt.startswith("\\"):
                    pt = pt[1:]
                for fth_name in filenames:
                    if fth_name[0] != '.':
                        name_path = os.path.join(pt, fth_name)
                        if "\\" in name_path:
                            name_path = name_path.replace("\\", "/")
                        var_name_list.append(name_path)

        return var_name_list

    load_block = default_main_program().global_block()
    load_var_map = {}
    load_optimizer_map = {}
    file_var_list = walk_filename(file_dir)
    for var_name in file_var_list:
        new_var = Variable(block=load_block, name=var_name)
        load_block.append_op(
            type='load',
            inputs={},
            outputs={'Out': [new_var]},
            attrs={
                'file_path': os.path.join(file_dir,
                                          os.path.normpath(new_var.name))
            })

        load_var_map[new_var.name] = new_var
    opt_path = os.path.join(file_dir, "optimizers")
    for _, _, optimizers in os.walk(opt_path):
        for optimizer in optimizers:
            f = open(os.path.join(opt_path, optimizer), "rb")
            load_optimizer_map[optimizer] = pickle.load(f)
            f.close()
    if len(load_optimizer_map) == 0:
        warnings.warn("No optimizer loaded")

    return load_var_map, load_optimizer_map


def _clone_var_in_block_(block, var):
    assert isinstance(var, Variable)
    return block.create_var(
        name=var.name,
        shape=var.shape,
        dtype=var.dtype,
        type=var.type,
        lod_level=0,
        persistable=True)
