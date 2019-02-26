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
import numpy
from .layers import Layer


def save_persistables(dirname, filename=None, layer=None):
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
        layer(Layer|PyLayer|None): The layer whose parameters will
                                    be loaded. If it is None, nothing
                                    will be deal.
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
            fluid.imperative.save_persistables(dirname=param_path,
                                       layer=ptb_model)
    """
    # Todo: PyLayer
    if isinstance(layer, Layer):
        layers_parameter_dict = {}
        for param in layer.parameters():
            layers_parameter_dict[param.name] = param._numpy()
        _save_var_to_file(layers_parameter_dict, dirname, filename)


def load_persistables(dirname, filename=None):
    """
    This function trys to load persistable variables from the folder
    `dirname` or the file `filename`.

    Use the `dirname` to specify the folder where persistable variables were
    saved. If variables were saved in separate files, set `filename` None;
    if all variables were saved in a single file, use `filename` to specify
    the file name.

    Args:
        dirname(str): The directory path.
        filename(str|None): The file which saved all variables, this file path should be end with '.npz'. If variables were
                            saved in differnet files, set it to None.
                            Default: None

    Returns:
        dict: The parameter-dict resumed from file

    Examples:
        .. code-block:: python

            param_path = "./my_paddle_model"
            param_dict = fluid.imperative.load_persistables(param_path)
            param_1 = param_dict['PtbModel_0.w_1']

            or:
            param_path = "./my_paddle_model"
            filename = "model.npz"
            param_dict = fluid.imperative.load_persistables(param_path, filename=filename)
            param_1 = param_dict['PtbModel_0.w_1']

        """
    parameter_dict = {}
    if filename is not None:
        # todo lujun: if remove np.save/load, fix this
        parameter_dict = numpy.load(os.path.join(dirname, filename))
    else:
        for pwd, dirs, files in os.walk(dirname):
            for parameter_file in files:
                # todo lujun: if remove np.save/load, fix this
                if os.path.splitext(parameter_file)[-1] == '.npy':
                    parameter_dict[parameter_file[:-4]] = numpy.load(
                        os.path.join(pwd, parameter_file))
    return parameter_dict


def _save_var_to_file(var_dict, file_dir, file_name):
    # ToDo change to C++ do
    if file_name is not None:
        # save to np.savez
        numpy.savez(os.path.join(file_dir, file_name), **var_dict)
    else:
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        for var_name in var_dict.keys():
            numpy.save(os.path.join(file_dir, var_name), var_dict[var_name])
