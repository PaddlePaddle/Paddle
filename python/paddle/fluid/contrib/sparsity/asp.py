# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2021 NVIDIA Corporation.  All rights reserved.
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
"""
Functions for Auto SParsity (ASP) training and inference.
"""

import os
import copy
import numpy as np
import paddle
from paddle.fluid.framework import dygraph_only
from paddle.fluid import global_scope, program_guard, layers
from paddle.fluid.initializer import ConstantInitializer
from paddle.fluid.contrib import sparsity
from paddle.fluid import core
from paddle.fluid.contrib.sparsity.supported_layer_list import supported_layers_and_prune_func_map
from paddle.fluid.contrib.sparsity.supported_layer_list import _default_pruning

OpRole = core.op_proto_and_checker_maker.OpRole
OP_ROLE_KEY = core.op_proto_and_checker_maker.kOpRoleAttrName()

__all__ = [
    'decorate', 'prune_model', 'set_excluded_layers', 'reset_excluded_layers'
]


def set_excluded_layers(param_names, main_program=None):
    r"""
    Set parameter name of layers which would not be pruned as sparse weights.

    Args:
        param_names (list of string): A list contains names of parameters.
        main_program (Program, optional): Program with model definition and its parameters.
                                          If None is given, then it would be set as `paddle.static.default_main_program().
                                          Default is None.
    Examples:
        1. Usage of Dynamic Graph

            .. code-block:: python

                import paddle

                class MyLayer(paddle.nn.Layer):
                    def __init__(self):
                        super(MyLayer, self).__init__()
                        self.conv1 = paddle.nn.Conv2D(
                            in_channels=3, out_channels=4, kernel_size=3, padding=2)
                        self.linear1 = paddle.nn.Linear(4624, 100)

                    def forward(self, img):
                        hidden = self.conv1(img)
                        hidden = paddle.flatten(hidden, start_axis=1)
                        prediction = self.linear1(hidden)
                        return prediction

                my_layer = MyLayer()
                optimizer = paddle.optimizer.SGD(
                    learning_rate=0.01, parameters=my_layer.parameters())

                # Need to set excluded layers before calling decorate
                paddle.incubate.asp.set_excluded_layers([my_layer.linear1.full_name()])

                optimizer = paddle.incubate.asp.decorate(optimizer)

        2. Usage of Static Graph

            .. code-block:: python

                import paddle

                paddle.enable_static()

                class MyLayer(paddle.nn.Layer):
                    def __init__(self):
                        super(MyLayer, self).__init__()
                        self.conv1 = paddle.nn.Conv2D(
                            in_channels=3, out_channels=4, kernel_size=3, padding=2)
                        self.linear1 = paddle.nn.Linear(4624, 100)

                    def forward(self, img):
                        hidden = self.conv1(img)
                        hidden = paddle.flatten(hidden, start_axis=1)
                        prediction = self.linear1(hidden)
                        return prediction

                main_program = paddle.static.Program()
                startup_program = paddle.static.Program()

                with paddle.static.program_guard(main_program, startup_program):
                    input_data = paddle.static.data(name='data', shape=[None, 3, 224, 224])
                    label = paddle.static.data(name='label', shape=[None, 100])
                    my_layer = MyLayer()
                    prob = my_layer(input_data)
                    loss = paddle.mean(paddle.nn.functional.square_error_cost(prob, label))

                    # Setup exluded layers out from ASP workflow.
                    # Please note, excluded_layers must be set before calling optimizer.minimize().
                    paddle.incubate.asp.set_excluded_layers([my_layer.linear1.full_name()], main_program)

                    optimizer = paddle.optimizer.SGD(learning_rate=0.1)
                    optimizer = paddle.static.amp.decorate(optimizer )
                    # Calling paddle.incubate.asp.decorate() to wrap minimize() in optimizer, which
                    # will insert necessary masking operations for ASP workflow.
                    optimizer = paddle.incubate.asp.decorate(optimizer)
                    optimizer.minimize(loss, startup_program)
    """
    if main_program is None:
        main_program = paddle.static.default_main_program()
    ASPHelper.set_excluded_layers(param_names=param_names,
                                  main_program=main_program)


def reset_excluded_layers(main_program=None):
    r"""
    Reset exculded layers setting corresponding to :attr:`main_program`. If :attr:`main_program`
    is None, then all configurations of excluded_layers would be cleaned.

    Args:
        main_program (Program, optional): Program with model definition and its parameters.
                                          If None is given, then this function would reset all excluded_layers.
                                          Default is None.
    Examples:
        1. Usage of Dynamic Graph

            .. code-block:: python

                import paddle

                class MyLayer(paddle.nn.Layer):
                    def __init__(self):
                        super(MyLayer, self).__init__()
                        self.conv1 = paddle.nn.Conv2D(
                            in_channels=3, out_channels=4, kernel_size=3, padding=2)
                        self.linear1 = paddle.nn.Linear(4624, 100)

                    def forward(self, img):
                        hidden = self.conv1(img)
                        hidden = paddle.flatten(hidden, start_axis=1)
                        prediction = self.linear1(hidden)
                        return prediction

                my_layer = MyLayer()
                optimizer = paddle.optimizer.SGD(
                    learning_rate=0.01, parameters=my_layer.parameters())

                # Need to set excluded layers before calling decorate
                paddle.incubate.asp.set_excluded_layers([my_layer.linear1.full_name()])
                # Reset excluded_layers, all supported layers would be included into Automatic SParsity's workflow.
                # Please note, reset_excluded_layers also must be called before calling sparsity.decorate().
                paddle.incubate.asp.reset_excluded_layers()

                optimizer = paddle.incubate.asp.decorate(optimizer)

        2. Usage of Static Graph

            .. code-block:: python

                import paddle

                paddle.enable_static()

                class MyLayer(paddle.nn.Layer):
                    def __init__(self):
                        super(MyLayer, self).__init__()
                        self.conv1 = paddle.nn.Conv2D(
                            in_channels=3, out_channels=4, kernel_size=3, padding=2)
                        self.linear1 = paddle.nn.Linear(4624, 100)

                    def forward(self, img):
                        hidden = self.conv1(img)
                        hidden = paddle.flatten(hidden, start_axis=1)
                        prediction = self.linear1(hidden)
                        return prediction

                main_program = paddle.static.Program()
                startup_program = paddle.static.Program()

                with paddle.static.program_guard(main_program, startup_program):
                    input_data = paddle.static.data(name='data', shape=[None, 3, 224, 224])
                    label = paddle.static.data(name='label', shape=[None, 100])
                    my_layer = MyLayer()
                    prob = my_layer(input_data)
                    loss = paddle.mean(paddle.nn.functional.square_error_cost(prob, label))

                    # Setup exluded layers out from ASP workflow.
                    # Please note, excluded_layers must be set before calling optimizer.minimize().
                    paddle.incubate.asp.set_excluded_layers([my_layer.linear1.full_name()], main_program)
                    # Reset excluded_layers, all supported layers would be included into Automatic SParsity's workflow.
                    # Please note, reset_excluded_layers also must be called before calling optimizer.minimize().
                    paddle.incubate.asp.reset_excluded_layers(main_program)

                    optimizer = paddle.optimizer.SGD(learning_rate=0.1)
                    optimizer = paddle.static.amp.decorate(optimizer )
                    # Calling paddle.incubate.asp.decorate() to wrap minimize() in optimizer, which
                    # will insert necessary masking operations for ASP workflow.
                    optimizer = paddle.incubate.asp.decorate(optimizer)
                    optimizer.minimize(loss, startup_program)
    """
    ASPHelper.reset_excluded_layers(main_program=main_program)


def decorate(optimizer):
    r"""
    Wrap the given optimizer as a OptimizerWithSparsityGuarantee,
    If runnig with dynamic graph mode. ASP would creates mask variables for supported parameters.
    Else if in static graph mode, ASP would creates mask variables and inserts necessary ops
    when calling minimize()

    Args:
        optimizer (Optimizer): A Optimizer used for training.
    Returns:
        OptimizerWithSparsityGuarantee: A wrapper for ASP to decorate `minimize` function of the given optimizer.
    Examples:
        1. Usage of Dynamic Graph

            .. code-block:: python

                import paddle

                class MyLayer(paddle.nn.Layer):
                    def __init__(self):
                        super(MyLayer, self).__init__()
                        self.conv1 = paddle.nn.Conv2D(
                            in_channels=3, out_channels=4, kernel_size=3, padding=2)
                        self.linear1 = paddle.nn.Linear(4624, 32)
                        self.linear2 = paddle.nn.Linear(32, 32)
                        self.linear3 = paddle.nn.Linear(32, 10)

                    def forward(self, img):
                        hidden = self.conv1(img)
                        hidden = paddle.flatten(hidden, start_axis=1)
                        hidden = self.linear1(hidden)
                        hidden = self.linear2(hidden)
                        prediction = self.linear3(hidden)
                        return prediction

                my_layer = MyLayer()
                optimizer = paddle.optimizer.SGD(
                    learning_rate=0.01, parameters=my_layer.parameters())

                # Calling paddle.incubate.asp.decorate() to wrap step() in optimizer, which
                # will apply necessary masking operations for ASP workflow.
                # In dynamic graph mode, ASP would create related mask variables during decoration.
                optimizer = paddle.incubate.asp.decorate(optimizer)

        2. Usage of Static Graph

            .. code-block:: python

                import paddle

                paddle.enable_static()

                class MyLayer(paddle.nn.Layer):
                    def __init__(self):
                        super(MyLayer, self).__init__()
                        self.conv1 = paddle.nn.Conv2D(
                            in_channels=3, out_channels=4, kernel_size=3, padding=2)
                        self.linear1 = paddle.nn.Linear(4624, 100)

                    def forward(self, img):
                        hidden = self.conv1(img)
                        hidden = paddle.flatten(hidden, start_axis=1)
                        prediction = self.linear1(hidden)
                        return prediction

                main_program = paddle.static.Program()
                startup_program = paddle.static.Program()

                with paddle.static.program_guard(main_program, startup_program):
                    input_data = paddle.static.data(name='data', shape=[None, 3, 224, 224])
                    label = paddle.static.data(name='label', shape=[None, 100])
                    my_layer = MyLayer()
                    prob = my_layer(input_data)
                    loss = paddle.mean(paddle.nn.functional.square_error_cost(prob, label))

                    optimizer = paddle.optimizer.SGD(learning_rate=0.1)
                    # Calling paddle.incubate.asp.decorate() to wrap minimize() in optimizer, which
                    # will insert necessary masking operations for ASP workflow.
                    # In static graph mode, ASP creates related mask variables
                    # during minimize().
                    optimizer = paddle.incubate.asp.decorate(optimizer)
                    optimizer.minimize(loss, startup_program)
    """
    return ASPHelper.decorate(optimizer)


def prune_model(model, n=2, m=4, mask_algo='mask_1d', with_mask=True):
    r"""
    Pruning parameters of supported layers in :attr:`model` via
    specified mask generation function given by :attr:`mask_algo`. This
    function supports both training and inference controlled by :attr:`with_mask`.
    If :attr:`with_mask` is True, it would also prune parameter related ASP mask Variables,
    else only prunes parameters.

    *Note*: (Static graph mode) If calling this function with :attr:`with_mask`, it should call `OptimizerWithSparsityGuarantee.minimize`
    and initialization (`exe.run(startup_program`)) before (For successfully obtain mask Variable).
    Typically set `with_mask` as true for training (have called `OptimizerWithSparsityGuarantee.minimize`) and false for
    inference only. To obtain OptimizerWithSparsityGuarantee, please see `paddle.incubate.asp.decoreate()`.

    Args:
        model (Program|nn.Layer): Program with model definition and its parameters, or a object of `paddle.nn.Layer`.
        n (int, optional): n of `n:m` sparse pattern. Default is 2.
        m (int, optional): m of `n:m` sparse pattern. Default is 4.
        mask_algo (string, optional): The function name to generate spase mask. Default is `mask_1d`.
                                      The vaild inputs should be one of 'mask_1d', 'mask_2d_greedy' and 'mask_2d_best'.
        with_mask (bool, optional): To prune mask Variables related to parameters or not. Ture is purning also, False is not. Default is True.
    Returns:
        dictionary: A dictionary with key: `parameter name` (string) and value: its corresponding mask Variable.
    Examples:
        1. Usage of Dynamic Graph

            .. code-block:: python

                import paddle
                import numpy as np

                class MyLayer(paddle.nn.Layer):
                    def __init__(self):
                        super(MyLayer, self).__init__()
                        self.conv1 = paddle.nn.Conv2D(
                            in_channels=3, out_channels=4, kernel_size=3, padding=2)
                        self.linear1 = paddle.nn.Linear(4624, 32)
                        self.linear2 = paddle.nn.Linear(32, 32)
                        self.linear3 = paddle.nn.Linear(32, 10)

                    def forward(self, img):
                        hidden = self.conv1(img)
                        hidden = paddle.flatten(hidden, start_axis=1)
                        hidden = self.linear1(hidden)
                        hidden = self.linear2(hidden)
                        prediction = self.linear3(hidden)
                        return prediction

                my_layer = MyLayer()
                loss_fn = paddle.nn.MSELoss(reduction='mean')

                optimizer = paddle.optimizer.SGD(
                    learning_rate=0.01, parameters=my_layer.parameters())

                # Calling paddle.incubate.asp.decorate() to wrap step() in optimizer, which
                # will apply necessary masking operations for ASP workflow.
                # In dynamic graph mode, ASP would create related mask variables during decoration.
                optimizer = paddle.incubate.asp.decorate(optimizer)

                # Must call paddle.incubate.asp.decorate() first before calling paddle.incubate.asp.prune_model()
                paddle.incubate.asp.prune_model(my_layer, mask_algo='mask_2d_best')

                for i in range(10):
                    imgs = paddle.to_tensor(
                        np.random.randn(64, 3, 32, 32),
                        dtype='float32', stop_gradient=False)
                    labels = paddle.to_tensor(
                        np.random.randint(10, size=(64, 1)),
                        dtype='float32', stop_gradient=False)
                    output = my_layer(imgs)
                    loss = loss_fn(output, labels)
                    loss.backward()
                    optimizer.step()
                    optimizer.clear_grad()

        2. Usage of Static Graph

            .. code-block:: python

                import paddle
                import numpy as np

                paddle.enable_static()

                class MyLayer(paddle.nn.Layer):
                    def __init__(self):
                        super(MyLayer, self).__init__()
                        self.conv1 = paddle.nn.Conv2D(
                            in_channels=3, out_channels=4, kernel_size=3, padding=2)
                        self.linear1 = paddle.nn.Linear(4624, 32)
                        self.linear2 = paddle.nn.Linear(32, 32)
                        self.linear3 = paddle.nn.Linear(32, 10)

                    def forward(self, img):
                        hidden = self.conv1(img)
                        hidden = paddle.flatten(hidden, start_axis=1)
                        hidden = self.linear1(hidden)
                        hidden = self.linear2(hidden)
                        prediction = self.linear3(hidden)
                        return prediction

                main_program = paddle.static.Program()
                startup_program = paddle.static.Program()

                with paddle.static.program_guard(main_program, startup_program):
                    input_data = paddle.static.data(name='data', shape=[None, 3, 32, 32])
                    label = paddle.static.data(name='label', shape=[None, 1])
                    my_layer = MyLayer()
                    prob = my_layer(input_data)
                    loss = paddle.mean(paddle.nn.functional.square_error_cost(prob, label))

                    optimizer = paddle.optimizer.SGD(learning_rate=0.1)
                    # Calling paddle.incubate.asp.decorate() to wrap minimize() in optimizer, which
                    # will insert necessary masking operations for ASP workflow.
                    # In static graph mode, ASP creates related mask variables
                    # during minimize().
                    optimizer = paddle.incubate.asp.decorate(optimizer)
                    optimizer.minimize(loss, startup_program)

                device = paddle.device.get_device()
                place = paddle.set_device(device)

                exe = paddle.static.Executor(place)
                exe.run(startup_program)

                # Must call exe.run(startup_program) first before calling paddle.asp.prune_model()
                paddle.incubate.asp.prune_model(my_layer, mask_algo='mask_2d_best')
                # it also be accepted to call
                # paddle.incubate.asp.prune_model(main_program, mask_algo='mask_2d_best')

                for i in range(10):
                    imgs = np.random.randn(64, 3, 32, 32).astype('float32')
                    labels = np.random.randint(10, size=(64, 1)).astype('float32')
                    exe.run(main_program, feed={'data':imgs, 'label':labels})
    """
    device = paddle.device.get_device()
    place = paddle.set_device(device)

    MaskAlgo_mapping = {
        'mask_1d': sparsity.MaskAlgo.MASK_1D,
        'mask_2d_greedy': sparsity.MaskAlgo.MASK_2D_GREEDY,
        'mask_2d_best': sparsity.MaskAlgo.MASK_2D_BEST
    }
    assert (mask_algo in MaskAlgo_mapping), \
        'The "mask_algo" should be one of ["mask_1d", "mask_2d_greedy", "mask_2d_best"]'

    prune_func = None
    if isinstance(model, paddle.nn.Layer):
        prune_func = ASPHelper.prune_model_by_layer
    elif isinstance(model, paddle.static.Program):
        prune_func = ASPHelper.prune_model_by_program
        if hasattr(model, "distributed_info_") and \
           model.distributed_info_["sharding_degree"] > 1 and \
           paddle.fluid.is_compiled_with_cuda():
            gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
            place = paddle.CUDAPlace(gpu_id)
    else:
        raise TypeError(
            "model should be paddle.nn.Layer or paddle.static.Program, but got {}"
            .format(type(model)))

    return prune_func(place,
                      model,
                      n=n,
                      m=m,
                      mask_algo=MaskAlgo_mapping[mask_algo],
                      with_mask=with_mask)


class ProgramASPInfo(object):
    r"""
    ProgramASPInfo is a container to keep ASP relevant information of Pragrom. It contains three inner-variables:
    1. __mask_vars (Dictionary): Key is parameter's name and vaule is its corresponding sparse mask Variable object, which is created by `ASPHelper.create_mask_variables`.
    2. __masks (Dictionary): Key is parameter's name and vaule is its corressponding sparse mask Numpy array, which is created by `ASPHelper.prune_model`.
    3. __excluded_layers (List): It stores name of layers which should not involve into ASP workflow.
    """

    def __init__(self):
        self.__mask_vars = {}
        self.__masks = {}
        self.__excluded_layers = []

    def update_mask_vars(self, param_name, var):
        self.__mask_vars[param_name] = var

    def update_masks(self, param_name, var):
        self.__masks[param_name] = var

    def update_excluded_layers(self, param_names):
        self.__excluded_layers.extend(copy.deepcopy(param_names))

    def reset_excluded_layers(self):
        self.__excluded_layers = []

    @property
    def mask_vars(self):
        return self.__mask_vars

    @property
    def masks(self):
        return self.__masks

    @property
    def excluded_layers(self):
        return self.__excluded_layers


class ASPHelper(object):
    r"""
    ASPHelper is a collection of Auto SParsity (ASP) functions to enable

    1. training models with weights in 2:4 sparse pattern on FP16 or 1:2 sparse pattern on FP32 from scratch.
    2. pruning well-trained models into 2:4 sparse pattern on FP16 or 1:2 sparse pattern on FP32 for fine-tuning.
    """

    MASK_APPENDDED_NAME = 'asp_mask'
    PADDLE_WEIGHT_SUFFIX = "w_"

    __asp_info = {}

    @classmethod
    def set_excluded_layers(cls, param_names, main_program):
        r"""
        This is the implementation of `sparsity.set_excluded_layers`, for details please see explanation in `sparsity.set_excluded_layers`.
        """
        asp_info = cls._get_program_asp_info(main_program)
        asp_info.update_excluded_layers(param_names)

    @classmethod
    def reset_excluded_layers(cls, main_program=None):
        r"""
        This is the implementation of `sparsity.reset_excluded_layers`, for details please see explanation in `sparsity.reset_excluded_layers`.
        """
        if main_program is None:
            for prog in cls.__asp_info:
                cls.__asp_info[prog].reset_excluded_layers()
        else:
            cls._get_program_asp_info(main_program).reset_excluded_layers()

    @staticmethod
    def decorate(optimizer):
        r"""
        This is the implementation of `sparsity.decorate`, for details please see explanation in `sparsity.decorate`.
        """
        if paddle.in_dynamic_mode():
            # main_prog and startup_prog would be used with paddle.static.program_guard
            # to create ASP masks. Moreover, main_prog is a key to map paddle.static.Program
            # to its own ASP informantion, like ASP mask variables. For dynamic graph, we use
            # default_main_program as the key.
            main_prog = paddle.static.default_main_program()
            startup_prog = paddle.static.default_startup_program()
            ASPHelper._create_mask_variables(main_prog, startup_prog,
                                             optimizer._parameter_list)
        return OptimizerWithSparsityGuarantee(optimizer)

    @classmethod
    def prune_model_by_program(cls,
                               place,
                               main_program=None,
                               n=2,
                               m=4,
                               mask_algo=sparsity.MaskAlgo.MASK_1D,
                               with_mask=True):
        r"""
        This is the implementation of `sparsity.prune_model`, for details please see explanation in `sparsity.prune_model`.
        """

        if main_program is None:
            main_program = paddle.static.default_main_program()

        asp_info = cls._get_program_asp_info(main_program)
        for param in main_program.global_block().all_parameters():
            if ASPHelper._is_supported_layer(main_program, param.name):
                weight_tensor = global_scope().find_var(param.name).get_tensor()
                weight_nparray = np.array(weight_tensor)

                prune_func = ASPHelper._get_prune_func_by_name(param.name)

                weight_pruned_nparray, weight_sparse_mask = \
                    prune_func(weight_nparray, m, n, mask_algo, param.name)
                weight_pruned_nparray = weight_pruned_nparray.astype(
                    weight_nparray.dtype)
                weight_tensor.set(weight_pruned_nparray, place)

                if with_mask:
                    weight_mask_param = global_scope().find_var(
                        ASPHelper._get_mask_name(param.name))
                    assert weight_mask_param is not None, \
                        'Cannot find {} variable, please call optimizer.minimize (' \
                        'paddle.sparsity.decorate(optimizer).minimize(loss)' \
                        ' and initialization (exe.run(startup_program)) first!'.format(ASPHelper._get_mask_name(param.name))
                    weight_mask_tensor = weight_mask_param.get_tensor()
                    weight_sparse_mask = weight_sparse_mask.astype(
                        np.array(weight_mask_tensor).dtype)
                    weight_mask_tensor.set(weight_sparse_mask, place)
                asp_info.update_masks(param.name, weight_sparse_mask)
        return asp_info.masks.copy()

    @classmethod
    def prune_model_by_layer(cls,
                             place,
                             layer,
                             n=2,
                             m=4,
                             mask_algo=sparsity.MaskAlgo.MASK_1D,
                             with_mask=True):
        r"""
        This is the implementation of `sparsity.prune_model`, for details please see explanation in `sparsity.prune_model`.
        """
        if paddle.in_dynamic_mode():
            main_program = paddle.static.default_main_program()
            asp_info = cls._get_program_asp_info(main_program)

            for param in layer.parameters():
                if ASPHelper._is_supported_layer(main_program, param.name):
                    weight_nparray = param.numpy()

                    prune_func = ASPHelper._get_prune_func_by_name(param.name)

                    weight_pruned_nparray, weight_sparse_mask = \
                        prune_func(weight_nparray, m, n, mask_algo, param.name)

                    weight_pruned_nparray = weight_pruned_nparray.astype(
                        weight_nparray.dtype)
                    param.set_value(weight_pruned_nparray)

                    if with_mask:
                        weight_mask_param = asp_info.mask_vars.get(
                            param.name, None)
                        assert weight_mask_param is not None, \
                            'Cannot find {} variable, please call sparsity.decorate() to' \
                            ' decorate your optimizer first!'.format(ASPHelper._get_mask_name(param.name))
                        weight_mask_param.set_value(weight_sparse_mask)

                    asp_info.update_masks(param.name, weight_sparse_mask)

            return asp_info.masks.copy()
        else:
            # This for loop is only used to obtain Block and Program from
            # first parameters.
            target_program = None
            for param in layer.parameters():
                target_program = param.block.program
            assert target_program is not None, \
                    'Cannot get paddle.static.Program from Paddle.nn.Layer.'
            return ASPHelper.prune_model_by_program(place,
                                                    target_program,
                                                    n=n,
                                                    m=m,
                                                    mask_algo=mask_algo,
                                                    with_mask=with_mask)

    @staticmethod
    def _get_mask_name(param_name):
        r"""
        Return mask name by given parameter name :attr:`param_name`.

        Args:
            param_name (string): The name of parameter.
        Returns:
            string: The mask name of :attr:`param_name`.
        """
        return param_name + "." + ASPHelper.MASK_APPENDDED_NAME

    @staticmethod
    def _get_not_ASP_relevant_vars(main_program):
        r"""
        Get all parameters's Variables in :attr:`main_program` but excluded ASP mask Variables.

        Args:
            main_program (Program): Program with model definition and its parameters.
        Returns:
            list: A list of parameter Variables in :attr:`main_program` (excluded ASP mask Variables).
        """
        var_list = []
        for param in main_program.global_block().all_parameters():
            param_name_list = param.name.split('.')

            if ASPHelper.MASK_APPENDDED_NAME not in param_name_list:
                var_list.append(param)
        return var_list

    @classmethod
    def _get_program_asp_info(cls, main_program):
        if main_program not in cls.__asp_info:
            cls.__asp_info[main_program] = ProgramASPInfo()
        return cls.__asp_info[main_program]

    @classmethod
    def _is_supported_layer(cls, main_program, param_name):
        r"""
        Verify if given :attr:`param_name` is supported by ASP.

        Args:
            param_name (string): The name of parameter.
        Returns:
            bool: True if it is supported, else False.
        Examples:
            .. code-block:: python

              from paddle.static.sparsity.asp import ASPHelper

              main_program = paddle.static.Program()
              startup_program = paddle.static.Program()

              with paddle.static.program_guard(main_program, startup_program):
                  input_data = paddle.static.data(name='data', shape=[None, 128])
                  fc = paddle.static.nn.fc(x=input_data, num_flatten_dims=-1, size=32, activation=None)

              for param in main_program.global_block().all_parameters():
                  ASPHelper._is_supported_layer(main_program, param.name)
              # fc_0.w_0 -> True
              # fc_0.b_0 -> False
        """
        param_name_list = param_name.split('.')

        if ASPHelper.MASK_APPENDDED_NAME in param_name_list:
            return False

        for layer in cls._get_program_asp_info(main_program).excluded_layers:
            if layer in param_name:
                return False

        if param_name in supported_layers_and_prune_func_map:
            return True

        # The parameter's name is neither in *.* format nor added to supported_layers_and_prune_func_map, return False.
        if len(param_name_list) == 1:
            return False

        param_name_no_weight_suffix = param_name_list[0]
        param_type_suffix = param_name_list[1]
        layer_name = param_name_no_weight_suffix[:param_name_no_weight_suffix.
                                                 rfind('_')]
        if ASPHelper.PADDLE_WEIGHT_SUFFIX not in param_type_suffix:
            return False

        if param_name_no_weight_suffix in supported_layers_and_prune_func_map or \
            layer_name in supported_layers_and_prune_func_map:
            return True

        return False

    @classmethod
    def _get_prune_func_by_name(cls, param_name):
        func = supported_layers_and_prune_func_map.get(param_name, None)
        param_name_no_weight_suffix = param_name.split('.')[0]
        if func is None:
            func = supported_layers_and_prune_func_map.get(
                param_name_no_weight_suffix, None)
        if func is None:
            layer_name = param_name_no_weight_suffix[:
                                                     param_name_no_weight_suffix
                                                     .rfind('_')]
            func = supported_layers_and_prune_func_map.get(
                layer_name, _default_pruning)
        return func

    @classmethod
    def _minimize(cls,
                  optimizer,
                  loss,
                  main_program=None,
                  startup_program=None,
                  parameter_list=None,
                  no_grad_set=None):
        r"""
        This function is a decorator of `minimize` function in `Optimizer`.
        There are three steps:

        1. Call :attr:`optimizer`.minimize(:attr:`loss`)
        2. Create sparse mask Tensors according to supported layers in :attr:`main_program`.
        3. Insert masking ops in the end of parameters update.

        *Note*: Please use `ASP.decorate` instead when applying distributed training with `Fleet`.
        (Due to there is a invisiable graphs optimization in `Fleet.minimize()` which make training graph
        cannot be modified anymore.)

        Args:
            optimizer (Optimizer): A Optimizer used for training.
            loss (Variable): A Variable containing the value to minimize.
            main_program (Program, optional): Program with model definition and its parameters. Default is `loss.block.program`.
            startup_program (Program, optional): Program for initializing parameters in `parameter_list`. Default is `paddle.static.default_startup_program()`.
            parameter_list (Iterable, optional): Iterable of `Variable` or `Variable.name` to update to minimize `loss`. The default value is None, at this time all parameters will be updated.
            no_grad_set (set, optional): Set of `Variable  or `Variable.name` that don't need to be updated. The default value is None.
        Returns:
            list: operators from :attr:`optimizer`.minimize(:attr:`loss`).
            list: pairs of parameters and their gradients.
        """
        if main_program is None:
            main_program = loss.block.program

        if startup_program is None:
            startup_program = paddle.static.default_startup_program()

        optimizer_ops, params_and_grads = optimizer.minimize(
            loss, startup_program, parameter_list, no_grad_set=no_grad_set)

        params_only = [pg[0] for pg in params_and_grads]
        cls._create_mask_variables(main_program, startup_program, params_only)
        cls._insert_sparse_mask_ops(main_program, params_only)
        return optimizer_ops, params_and_grads

    @classmethod
    @dygraph_only
    def _step(cls, optimizer):
        r"""
        This function is a decorator of `step` function in `Optimizer`.
        There are three steps:

        1. Call :attr:`optimizer`.step()
        2. Mask parameters with sparse masks.

        *Note*: Please use `ASP.decorate` instead when applying distributed training with `Fleet`.
        (Due to there is a invisiable graphs optimization in `Fleet.minimize()` which make training graph
        cannot be modified anymore.)

        Args:
            optimizer (Optimizer): A Optimizer used for training.
        """
        optimizer.step()
        main_prog = paddle.static.default_main_program()
        with paddle.fluid.dygraph.no_grad():
            ASPHelper._insert_sparse_mask_ops(main_prog,
                                              optimizer._parameter_list)

    @classmethod
    def _create_mask_variables(cls, main_program, startup_program, params):
        r"""
        Create sparse mask Tensors according to supported layers in :attr:`main_program`.
        This function is called in second step of `ASPHelper._minimize`

        Args:
            main_program (Program): Program with model definition and its parameters.
            startup_program (Program): Program for initializing parameters.
            params (list): Variable parameters.
        """
        asp_info = cls._get_program_asp_info(main_program)
        with program_guard(main_program, startup_program):
            for param in params:
                if ASPHelper._is_supported_layer(main_program, param.name):
                    if param.name not in asp_info.mask_vars:
                        mask_param = layers.create_parameter(
                            name=ASPHelper._get_mask_name(param.name),
                            shape=param.shape,
                            dtype=param.dtype,
                            default_initializer=ConstantInitializer(value=1.0))
                        mask_param.stop_gradient = True
                        mask_param.trainable = False
                        asp_info.update_mask_vars(param.name, mask_param)

    @classmethod
    def _insert_sparse_mask_ops(cls, main_program, params):
        r"""
        Insert masking ops in the end of parameters update.
        This function is called in third step of `ASPHelper._minimize`

        Args:
            main_program (Program): Program with model definition and its parameters.
            params (list): Variable parameters.
        """
        block = main_program.global_block()
        asp_info = cls._get_program_asp_info(main_program)
        for param in params:
            if param.name in asp_info.mask_vars:
                block.append_op(type='elementwise_mul',
                                inputs={
                                    "X": param,
                                    'Y': asp_info.mask_vars[param.name]
                                },
                                outputs={'Out': param},
                                attrs={
                                    'axis': -1,
                                    'use_mkldnn': False,
                                    OP_ROLE_KEY: int(OpRole.Optimize)
                                })


class OptimizerWithSparsityGuarantee(object):
    r"""
    OptimizerWithSparsityGuarantee is a wrapper to decorate `minimize` function of given optimizer by `_minimize` of ASPHelper.
    The decorated `minimize` function would do three things (exactly same as `ASPHelper._minimize`):
    1. Call `minimize` function of given optimizer.
    2. Call `ASPHelper._create_mask_variables` to create mask Variables.
    3. Call `ASPHelper._insert_sparse_mask_ops` to insert weight masking ops in the end of `loss`'s Program.
    """

    def __init__(self, optimizer):
        self._optimizer = optimizer

    def __getattr__(self, item):
        return getattr(self._optimizer, item)

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        r"""
        This function is to call `ASPHelper.minimize()` and return its return

        Args:
            loss (Variable): A Variable containing the value to minimize.
            startup_program (Program, optional): Program for initializing parameters in `parameter_list`. Default is `paddle.static.default_startup_program()`.
            parameter_list (Iterable, optional): Iterable of `Variable` or `Variable.name` to update to minimize `loss`. The default value is None, at this time all parameters will be updated.
            no_grad_set (set, optional): Set of `Variable  or `Variable.name` that don't need to be updated. The default value is None.
        Returns:
            list: operators from :attr:`optimizer`.minimize(:attr:`loss`).
            list: pairs of parameters and their gradients.
        """
        return ASPHelper._minimize(self._optimizer,
                                   loss,
                                   startup_program=startup_program,
                                   parameter_list=parameter_list,
                                   no_grad_set=no_grad_set)

    @dygraph_only
    def step(self):
        r"""
        This function is a decorator of `step` function in `Optimizer`.
        There are three steps:

        1. Call :attr:`optimizer`.step()
        2. Mask parameters with sparse masks.

        *Note*: Please use `ASP.decorate` instead when applying distributed training with `Fleet`.
        (Due to there is a invisiable graphs optimization in `Fleet.minimize()` which make training graph
        cannot be modified anymore.)

        Args:
            optimizer (Optimizer): A Optimizer used for training.
        """
        ASPHelper._step(self._optimizer)

    @dygraph_only
    def state_dict(self):
        r"""
        This function is a decorator of `state_dict` function in `Optimizer`.

        Returns:
            state_dict(dict) : dict contains all the Tensor used by optimizer
        """
        state_dict = self._optimizer.state_dict()
        asp_info = ASPHelper._get_program_asp_info(
            paddle.static.default_main_program())
        for param_name, var in asp_info.mask_vars.items():
            state_dict.update({ASPHelper._get_mask_name(param_name): var})
        return state_dict

    @dygraph_only
    def set_state_dict(self, state_dict):
        r"""
        This function is a decorator of `set_state_dict` function in `Optimizer`.
        Args:
            state_dict(dict) : Dict contains all the Tensor needed by optimizer
        Return:
            None
        """
        asp_info = ASPHelper._get_program_asp_info(
            paddle.static.default_main_program())
        for param_name, var in asp_info.mask_vars.items():
            param_mask_name = ASPHelper._get_mask_name(param_name)
            assert param_mask_name in state_dict, \
                "The {} is not found.".format(param_mask_name)
            var.set_value(state_dict[param_mask_name])
            asp_info.update_masks(param_name, var.numpy())
        return self._optimizer.set_state_dict(state_dict)
