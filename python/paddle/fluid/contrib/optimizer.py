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
from paddle.fluid.optimizer import Optimizer
from paddle.fluid.regularizer import L1DecayRegularizer
from paddle.fluid.regularizer import L2DecayRegularizer
from paddle.fluid.regularizer import append_regularization_ops
from paddle.fluid import framework
from paddle.fluid import core
from paddle.fluid.framework import program_guard
from paddle.fluid.clip import append_gradient_clip_ops

__all__ = ['Momentum']


class Momentum(Optimizer):
    """

    Simple Momentum optimizer with velocity state

    This optimizer has a flag for Nestrov Momentum.

    The update equations are as follows:

    .. math::

        & velocity = mu * velocity + gradient

        & if (use\_nesterov):

        &\quad   param = param - (gradient + mu * velocity) * learning\_rate

        & else:

        &\quad   param = param - learning\_rate * velocity

    Parameters:
        learning_rate (float|Variable): The learning rate used to update parameters. \
            Can be a float value or a Variable with one float value as data element.
        momentum (float): Momentum factor
        parameter_list (Iterable, optional):  Iterable of ``Variable`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
        use_nesterov (bool, optional): Enables Nesterov momentum, default is false.
        regularization (WeightDecayRegularizer, optional): The strategy of regularization. There are two method: \
             :ref:`api_fluid_regularizer_L1Decay` , :ref:`api_fluid_regularizer_L2Decay` . If a parameter has set \
            regularizer using :ref:`api_fluid_ParamAttr` already, the regularization setting here in optimizer will be \
            ignored for this parameter. Otherwise, the regularization setting here in optimizer will take effect.  \
            Default None, meaning there is no regularization.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of 
            some derived class of ``GradientClipBase`` . There are three cliping strategies 
            ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` , 
            :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
        name (str, optional): This parameter is used by developers to print debugging information. \
            For details, please refer to :ref:`api_guide_Name`. Default is None.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import numpy as np

            paddle.enable_static()

            place = fluid.CPUPlace()
            main = fluid.Program()
            with fluid.program_guard(main):
                x = paddle.static.data(name='x', shape=[1, 13], dtype='float32')
                y = paddle.static.data(name='y', shape=[1], dtype='float32')
                linear = paddle.nn.Linear(13, 1)
                y_predict = linear(x)
                cost = paddle.nn.functional.square_error_cost(input=y_predict, label=y)
                avg_cost = paddle.mean(cost)

                moment_optimizer = fluid.contrib.optimizer.Momentum(learning_rate=0.001, momentum=0.9)
                moment_optimizer.minimize(avg_cost)

                fetch_list = [avg_cost]
                train_reader = paddle.batch(
                    paddle.dataset.uci_housing.train(), batch_size=1)
                feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
                exe = fluid.Executor(place)
                exe.run(paddle.static.default_startup_program())
                for data in train_reader():
                    exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)

    """
    _velocity_acc_str = "velocity"

    def __init__(self,
                 learning_rate,
                 momentum,
                 parameter_list=None,
                 use_nesterov=False,
                 regularization=None,
                 grad_clip=None,
                 name=None):
        assert learning_rate is not None
        assert momentum is not None
        predicate = lambda regular: isinstance(regular, L2DecayRegularizer)
        py_regular = None if predicate(regularization) else regularization
        super(Momentum, self).__init__(
            learning_rate=learning_rate,
            parameter_list=parameter_list,
            regularization=py_regular,
            grad_clip=grad_clip,
            name=name)
        self.type = "momentum"
        self._momentum = momentum
        self._use_nesterov = bool(use_nesterov)
        self._regularization_method = ""
        self._regularization_coeff = 0
        if (isinstance(regularization, L2DecayRegularizer)):
            self._regularization_method = "l2_decay"
            self._regularization_coeff = regularization._regularization_coeff

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)

        for p in parameters:
            self._add_accumulator(self._velocity_acc_str, p)

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)

        velocity_acc = self._get_accumulator(self._velocity_acc_str,
                                             param_and_grad[0])
        lr = self._create_param_lr(param_and_grad)

        if framework.in_dygraph_mode():
            _, _ = core.ops.momentum(
                param_and_grad[0], param_and_grad[1], velocity_acc, lr,
                param_and_grad[0], velocity_acc, 'mu', self._momentum,
                'use_nesterov', self._use_nesterov, 'regularization_method',
                self._regularization_method, 'regularization_coeff',
                self._regularization_coeff)
            return None

        attrs = {
            "mu": self._momentum,
            "use_nesterov": self._use_nesterov,
            "regularization_method": self._regularization_method,
            "regularization_coeff": self._regularization_coeff
        }
        inputs = {
            "Param": [param_and_grad[0]],
            "Grad": [param_and_grad[1]],
            "Velocity": [velocity_acc],
            "LearningRate": [lr]
        }

        outputs = {
            "ParamOut": [param_and_grad[0]],
            "VelocityOut": [velocity_acc]
        }
        # create the momentum optimize op
        momentum_op = block.append_op(
            type=self.type,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            stop_gradient=True)

        return momentum_op
