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
from ..framework import Variable, default_main_program
import pickle
from . import learning_rate_scheduler
import warnings

__all__ = [
    'save_persistables', 'load_persistables', 'save_parameter',
    'load_parameter', 'save_optimizer', 'load_optimizer'
]


def save_persistables(model_dict, dirname='save_dir', optimizers=None):
    """
    This function filters out all variables in layer.parameters from the give `layer`, and optimizer's learning rate decay.
    And then trys to save these variables to the folder `dirname`.

    Use the `dirname` to specify the folder where persistable variables were
    saved.

    Args:
        model_dict(dict of Parameters): The parameters will
                                    be saved. If it is None, nothing
                                    will be deal.
        dirname(str): The directory path.
        optimizers(fluid.Optimizer|list(fluid.Optimizer)|None): The optimizers to be saved

    Returns:
        None

    Examples:

        .. code-block:: python

          ptb_model = PtbModel(
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_steps=num_steps,
                init_scale=init_scale)
          sgd = fluid.optimizer.SGD(learning_rate=0.01)
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
          dy_loss.backward()
          sgd.minimize(dy_loss)
          ptb_model.clear_gradient()
          param_path = "./my_paddle_model"
          fluid.dygraph.save_persistables(ptb_model.state_dict(), dirname=param_path, sgd)
    """
    if isinstance(model_dict, collections.OrderedDict):
        _save_var_to_file(model_dict, optimizers, dirname, None)


def load_persistables(dirname='save_dir'):
    """
    This function trys to load persistable variables and optimizer's learning rate decay from the folder `dirname`.
    And return the restored values in a dictionary way, respectively.

    Use the `dirname` to specify the folder where persistable variables were
    saved.

    Args:
        dirname(str): The directory path. default is save_dir

    Returns:
        layer_dict: The parameter-dict resumed from file
        optimizer: The optimizer

    Examples:

         .. code-block:: python

           my_layer = layer(fluid.Layer)
           param_path = "./my_paddle_model"
           sgd = SGDOptimizer(learning_rate=1e-3)
           param_dict, optimizer_dict = fluid.dygraph.load_persistables(my_layer.parameters(), param_path)
           param_1 = param_dict['PtbModel_0.w_1']
           sgd.load(optimizer_dict)

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

    if optimizers is not None:
        if isinstance(optimizers, (list, tuple)):
            optimizers = optimizers
        else:
            optimizers = [optimizers]
        if os.path.exists(
                os.path.join(file_dir, os.path.normpath("optimizers"))):
            pass
        else:
            os.mkdir(os.path.join(file_dir, os.path.normpath("optimizers")))
        for optimizer in optimizers:
            if isinstance(optimizer._learning_rate,
                          learning_rate_scheduler.LearningRateDecay):
                try:
                    f = open(
                        os.path.join(file_dir, "optimizers",
                                     os.path.normpath(str(optimizer._name))),
                        "wb")
                    pickle.dump(optimizer._learning_rate, f, 2)
                    f.close()
                except ():
                    raise IOError("Can't load %s",
                                  os.path.join(
                                      file_dir, "optimizers",
                                      os.path.normpath(str(optimizer._name))))
            else:
                warnings.warn(
                    "Optimizer not saved, Only optimizer with 'LearningRateDecay' under DyGraph mode need to be saved"
                )
    else:
        pass

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
    if not os.path.exists(file_dir):
        raise IOError("{} not exist".format(file_dir))

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
            try:
                f = open(os.path.join(opt_path, optimizer), "rb")
                load_optimizer_map[optimizer] = pickle.load(f)
                f.close()
            except IOError:
                raise IOError("Can't load %s",
                              os.path.join(
                                  file_dir, "optimizers",
                                  os.path.normpath(str(optimizer._name))))
    if len(load_optimizer_map) == 0:
        print(
            "No optimizer loaded. If you didn't save optimizer, please ignore this. The program can still work with new optimizer. "
        )
        pass

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


def save_parameter(para_dict, save_dir):
    """
    This function saves model parameter dict to the folder `save_dir`.

    Args:
        para_dict(dict of Parameter): specify the model state dict. The key type is str, the value type is Parameter
        save_dir(str): the folder to save the parameters.

    Returns:
        None.

    Examples:

         .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import numpy as np
            from paddle.fluid.dygraph.nn import FC
            from paddle.fluid.optimizer import SGDOptimizer
            from paddle.fluid.dygraph.base import to_variable


            class MyLayer(fluid.dygraph.Layer):
                def __init__(self, name_scope):
                    super(MyLayer, self).__init__(name_scope)

                def forward(self, inputs):
                    x = fluid.layers.relu(inputs)
                    self._x_for_debug = x
                    x = fluid.layers.elementwise_mul(x, x)
                    x = fluid.layers.reduce_sum(x)
                    return [x]


            class MLP(fluid.Layer):
                def __init__(self, name_scope):
                    super(MLP, self).__init__(name_scope)
                    self._fc1 = FC(self.full_name(), 10)
                    self._fc2 = FC(self.full_name(), 10)

                def forward(self, inputs):
                    y = self._fc1(inputs)
                    y = self._fc2(y)
                    return y


            if __name__ == '__main__':
                    train_reader = paddle.batch(
                        paddle.dataset.mnist.train(), batch_size=128, drop_last=True)

                    with fluid.dygraph.guard():
                        mlp = MLP("mlp")

                        opt = SGDOptimizer(learning_rate=fluid.layers.natural_exp_decay(
                            learning_rate=0.1,
                            decay_steps=1,
                            decay_rate=0.5,
                            staircase=True))
                        for batch_id, data in enumerate(train_reader()):
                            dy_x_data = np.array(
                                [x[0].reshape(1, 28, 28) for x in data]).astype('float32')
                            y_data = np.array([x[1] for x in data]).astype('int64').reshape(
                                128, 1)

                            img = to_variable(dy_x_data)
                            label = to_variable(y_data)
                            label._stop_gradient = True

                            cost = mlp(img)
                            avg_loss = fluid.layers.reduce_mean(cost)
                            avg_loss.backward()

                            opt.minimize(avg_loss)
                        
                            if batch_id == 1:
                                break


                        with fluid.unique_name.guard():
                            fluid.dygraph.save_parameter(mlp.state_dict(), save_dir="dirname")
                            para_dict = fluid.dygraph.load_parameter(load_dir="dirname")
                            another_mlp = MLP("mlp")
                            another_mlp.set_dict(para_dict)

                        opt_dict = opt.state_dict()
                        fluid.dygraph.save_optimizer(opt_dict, "dirname")
                        opt.set_dict(fluid.dygraph.load_optimizer("dirname"))

    """

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_block = default_main_program().global_block()
    save_var_list = list()
    for name in sorted(para_dict.keys()):
        save_var_list.append(para_dict[name])
    save_block.append_op(
        type='save_combine',
        inputs={'X': save_var_list},
        outputs={},
        attrs={'file_path': os.path.join(save_dir, "model.pdparams")})

    var_names_path = os.path.join(save_dir, "var_names.pdvar")
    var_names_str = "\n".join(sorted(para_dict.keys()))
    with open(var_names_path, "w") as f:
        f.write(var_names_str)
        f.close()


def load_parameter(load_dir):
    """
    This function loads model parameter dict from the folder `load_dir`.

    Args:
        para_dict(dict of Parameter): the model state dict. The key type is str, the value type is Parameter
        load_dir(str): the folder to load from.

    Returns:
        para_dict(dict of Parameter): the dict of model parameters.

    Examples:

         .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import numpy as np
            from paddle.fluid.dygraph.nn import FC
            from paddle.fluid.optimizer import SGDOptimizer
            from paddle.fluid.dygraph.base import to_variable


            class MyLayer(fluid.dygraph.Layer):
                def __init__(self, name_scope):
                    super(MyLayer, self).__init__(name_scope)

                def forward(self, inputs):
                    x = fluid.layers.relu(inputs)
                    self._x_for_debug = x
                    x = fluid.layers.elementwise_mul(x, x)
                    x = fluid.layers.reduce_sum(x)
                    return [x]


            class MLP(fluid.Layer):
                def __init__(self, name_scope):
                    super(MLP, self).__init__(name_scope)
                    self._fc1 = FC(self.full_name(), 10)
                    self._fc2 = FC(self.full_name(), 10)

                def forward(self, inputs):
                    y = self._fc1(inputs)
                    y = self._fc2(y)
                    return y


            if __name__ == '__main__':
                    train_reader = paddle.batch(
                        paddle.dataset.mnist.train(), batch_size=128, drop_last=True)

                    with fluid.dygraph.guard():
                        mlp = MLP("mlp")

                        opt = SGDOptimizer(learning_rate=fluid.layers.natural_exp_decay(
                            learning_rate=0.1,
                            decay_steps=1,
                            decay_rate=0.5,
                            staircase=True))
                        for batch_id, data in enumerate(train_reader()):
                            dy_x_data = np.array(
                                [x[0].reshape(1, 28, 28) for x in data]).astype('float32')
                            y_data = np.array([x[1] for x in data]).astype('int64').reshape(
                                128, 1)

                            img = to_variable(dy_x_data)
                            label = to_variable(y_data)
                            label._stop_gradient = True

                            cost = mlp(img)
                            avg_loss = fluid.layers.reduce_mean(cost)
                            avg_loss.backward()

                            opt.minimize(avg_loss)
                        
                            if batch_id == 1:
                                break


                        with fluid.unique_name.guard():
                            fluid.dygraph.save_parameter(mlp.state_dict(), save_dir="dirname")
                            para_dict = fluid.dygraph.load_parameter(load_dir="dirname")
                            another_mlp = MLP("mlp")
                            another_mlp.set_dict(para_dict)

                        opt_dict = opt.state_dict()
                        fluid.dygraph.save_optimizer(opt_dict, "dirname")
                        opt.set_dict(fluid.dygraph.load_optimizer("dirname"))
    """

    var_names_path = os.path.join(load_dir, "var_names.pdvar")
    with open(var_names_path, "r") as f:
        var_names = [line.strip() for line in f.readlines()]
        f.close()

    load_block = default_main_program().global_block()
    load_var_list = list()
    for var_name in var_names:
        new_var = Variable(block=load_block, name=var_name)
        load_var_list.append(new_var)
    para_path = os.path.join(load_dir, "model.pdparams")
    load_block.append_op(
        type='load_combine',
        inputs={},
        outputs={"Out": load_var_list},
        attrs={'file_path': para_path})
    para_dict = {var.name: var for var in load_var_list}
    return para_dict


def save_optimizer(opt_dict, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    """
    This function saves optimizer state dict to the folder `save_dir`.

    Args:
        opt_dict(dict of numpy array): the optimizer state dict. The key type is str, the value type is numpy array.
        save_dir(str): the folder to save the optimizer state.

    Returns:
        None.

    Examples:

         .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import numpy as np
            from paddle.fluid.dygraph.nn import FC
            from paddle.fluid.optimizer import SGDOptimizer
            from paddle.fluid.dygraph.base import to_variable


            class MyLayer(fluid.dygraph.Layer):
                def __init__(self, name_scope):
                    super(MyLayer, self).__init__(name_scope)

                def forward(self, inputs):
                    x = fluid.layers.relu(inputs)
                    self._x_for_debug = x
                    x = fluid.layers.elementwise_mul(x, x)
                    x = fluid.layers.reduce_sum(x)
                    return [x]


            class MLP(fluid.Layer):
                def __init__(self, name_scope):
                    super(MLP, self).__init__(name_scope)
                    self._fc1 = FC(self.full_name(), 10)
                    self._fc2 = FC(self.full_name(), 10)

                def forward(self, inputs):
                    y = self._fc1(inputs)
                    y = self._fc2(y)
                    return y


            if __name__ == '__main__':
                    train_reader = paddle.batch(
                        paddle.dataset.mnist.train(), batch_size=128, drop_last=True)

                    with fluid.dygraph.guard():
                        mlp = MLP("mlp")

                        opt = SGDOptimizer(learning_rate=fluid.layers.natural_exp_decay(
                            learning_rate=0.1,
                            decay_steps=1,
                            decay_rate=0.5,
                            staircase=True))
                        for batch_id, data in enumerate(train_reader()):
                            dy_x_data = np.array(
                                [x[0].reshape(1, 28, 28) for x in data]).astype('float32')
                            y_data = np.array([x[1] for x in data]).astype('int64').reshape(
                                128, 1)

                            img = to_variable(dy_x_data)
                            label = to_variable(y_data)
                            label._stop_gradient = True

                            cost = mlp(img)
                            avg_loss = fluid.layers.reduce_mean(cost)
                            avg_loss.backward()

                            opt.minimize(avg_loss)
                        
                            if batch_id == 1:
                                break


                        with fluid.unique_name.guard():
                            fluid.dygraph.save_parameter(mlp.state_dict(), save_dir="dirname")
                            para_dict = fluid.dygraph.load_parameter(load_dir="dirname")
                            another_mlp = MLP("mlp")
                            another_mlp.set_dict(para_dict)

                        opt_dict = opt.state_dict()
                        fluid.dygraph.save_optimizer(opt_dict, "dirname")
                        opt.set_dict(fluid.dygraph.load_optimizer("dirname"))


    """
    opt_path = os.path.join(save_dir, "model.pdopt")
    with open(opt_path, "wb") as f:
        pickle.dump(opt_dict, f, 2)
        f.close()


def load_optimizer(load_dir):
    """
    This function loads optimizer state parameter dict from the folder `load_dir`.

    Args:
        opt_dict(dict of numpy array): the optimizer state dict. The key type is str, the value type is numpy array.
        load_dir(str): the folder to load from.

    Returns:
        opt_dict(dict of numpy array): the dict of optimizer state.

    Examples:

         .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import numpy as np
            from paddle.fluid.dygraph.nn import FC
            from paddle.fluid.optimizer import SGDOptimizer
            from paddle.fluid.dygraph.base import to_variable


            class MyLayer(fluid.dygraph.Layer):
                def __init__(self, name_scope):
                    super(MyLayer, self).__init__(name_scope)

                def forward(self, inputs):
                    x = fluid.layers.relu(inputs)
                    self._x_for_debug = x
                    x = fluid.layers.elementwise_mul(x, x)
                    x = fluid.layers.reduce_sum(x)
                    return [x]


            class MLP(fluid.Layer):
                def __init__(self, name_scope):
                    super(MLP, self).__init__(name_scope)
                    self._fc1 = FC(self.full_name(), 10)
                    self._fc2 = FC(self.full_name(), 10)

                def forward(self, inputs):
                    y = self._fc1(inputs)
                    y = self._fc2(y)
                    return y


            if __name__ == '__main__':
                    train_reader = paddle.batch(
                        paddle.dataset.mnist.train(), batch_size=128, drop_last=True)

                    with fluid.dygraph.guard():
                        mlp = MLP("mlp")

                        opt = SGDOptimizer(learning_rate=fluid.layers.natural_exp_decay(
                            learning_rate=0.1,
                            decay_steps=1,
                            decay_rate=0.5,
                            staircase=True))
                        for batch_id, data in enumerate(train_reader()):
                            dy_x_data = np.array(
                                [x[0].reshape(1, 28, 28) for x in data]).astype('float32')
                            y_data = np.array([x[1] for x in data]).astype('int64').reshape(
                                128, 1)

                            img = to_variable(dy_x_data)
                            label = to_variable(y_data)
                            label._stop_gradient = True

                            cost = mlp(img)
                            avg_loss = fluid.layers.reduce_mean(cost)
                            avg_loss.backward()

                            opt.minimize(avg_loss)
                        
                            if batch_id == 1:
                                break


                        with fluid.unique_name.guard():
                            fluid.dygraph.save_parameter(mlp.state_dict(), save_dir="dirname")
                            para_dict = fluid.dygraph.load_parameter(load_dir="dirname")
                            another_mlp = MLP("mlp")
                            another_mlp.set_dict(para_dict)

                        opt_dict = opt.state_dict()
                        fluid.dygraph.save_optimizer(opt_dict, "dirname")
                        opt.set_dict(fluid.dygraph.load_optimizer("dirname"))


    """

    opt_path = os.path.join(load_dir, "model.pdopt")
    with open(opt_path, "rb") as f:
        opt_dict = pickle.load(f)
        f.close()
    return opt_dict
