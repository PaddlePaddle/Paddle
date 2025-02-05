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
import warnings

from paddle import _C_ops, _legacy_C_ops, pir
from paddle.base import framework
from paddle.framework import (
    in_dynamic_mode,
    in_pir_mode,
)
from paddle.optimizer import Optimizer


class LarsMomentumOptimizer(Optimizer):
    r"""
    Momentum optimizer with LARS support

    The update equations are as follows:

    .. math::

        & local\_learning\_rate = learning\_rate * lars\_coeff * \\
          \\frac{||param||}{||gradient|| + lars\_weight\_decay * ||param||}

        & velocity = mu * velocity + local\_learning\_rate * (gradient + lars\_weight\_decay * param + epsilon)

        & param = param - velocity

    Parameters:
        learning_rate (float|Variable): The learning rate used to update parameters. \
            Can be a float value or a Variable with one float value as data element. \
            momentum (float): momentum factor
        lars_coeff (float): Defines how much we trust the layer to change its weights.
        lars_weight_decay (float): Weight decay coefficient for decaying using LARS.
        parameter_list (Iterable, optional):  Iterable of ``Variable`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static graph mode, at this time all parameters will be updated.
        regularization (WeightDecayRegularizer, optional): The strategy of regularization. There are two method: \
             :ref:`api_paddle_regularizer_L1Decay` , :ref:`api_paddle_regularizer_L2Decay` . If a parameter has set \
            regularizer using :ref:`api_paddle_ParamAttr` already, the regularization setting here in optimizer will be \
            ignored for this parameter. Otherwise, the regularization setting here in optimizer will take effect.  \
            Default None, meaning there is no regularization.
        grad_clip (GradientClipBase, optional): Gradient clipping strategy, it's an instance of
            some derived class of ``GradientClipBase`` . There are three clipping strategies
            ( :ref:`api_paddle_nn_ClipGradByGlobalNorm` , :ref:`api_paddle_nn_ClipGradByNorm` ,
            :ref:`api_paddle_nn_ClipGradByValue` ). Default None, meaning there is no gradient clipping.
        name (str, optional): This parameter is used by developers to print debugging information. \
            For details, please refer to :ref:`api_guide_Name`. Default is None.
        exclude_from_weight_decay (list[str], optional): Name string of layers which will be exclude from lars weight decay. Default is None.
        epsilon (float, optional): Epsilon to avoid Division by Zero when calculate local lr. Default is 0.
        multi_precision (bool, optional): Whether to use multi-precision during weight updating.
        rescale_grad (float, optional): Multiply the gradient with `rescale_grad` \
            before updating. Often choose to be `1.0/batch_size`.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import numpy as np

            >>> paddle.enable_static()
            >>> np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            >>> inp = paddle.static.data(
            ...     name="inp", shape=[2, 2], dtype='float32')
            >>> out = paddle.static.nn.fc(inp, size=3)
            >>> out = paddle.sum(out)
            >>> optimizer = paddle.incubate.optimizer.LarsMomentumOptimizer(learning_rate=0.001, momentum=0.9)
            >>> optimizer.minimize(out)

            >>> exe = paddle.static.Executor(paddle.CPUPlace())
            >>> exe.run(paddle.static.default_startup_program())
            >>> exe.run(
            ...     feed={"inp": np_inp},
            ...     fetch_list=[out.name])
    """

    _velocity_acc_str = "velocity"

    def __init__(
        self,
        learning_rate,
        momentum,
        lars_coeff=0.001,
        lars_weight_decay=0.0005,
        parameter_list=None,
        regularization=None,
        grad_clip=None,
        name=None,
        exclude_from_weight_decay=None,
        epsilon=0,
        multi_precision=False,
        rescale_grad=1.0,
    ):
        assert learning_rate is not None
        assert momentum is not None
        super().__init__(
            learning_rate=learning_rate,
            parameters=parameter_list,
            weight_decay=regularization,
            grad_clip=grad_clip,
            name=name,
        )
        self.type = "lars_momentum"
        self._momentum = momentum
        self._lars_coeff = float(lars_coeff)
        self._lars_weight_decay = float(lars_weight_decay)
        self._epsilon = float(epsilon)
        if exclude_from_weight_decay is None:
            self._exclude_from_weight_decay = []
        else:
            self._exclude_from_weight_decay = exclude_from_weight_decay
        self._multi_precision = multi_precision
        self._rescale_grad = float(rescale_grad)
        self._master_weights = {}

    def _create_accumulators(self, block, parameters):
        if not isinstance(block, (framework.Block, pir.Block)):
            raise TypeError("block is not instance of Block.")
        for p in parameters:
            if self._multi_precision and self._is_dtype_fp16_or_bf16(p.dtype):
                master_p = self._create_master_weight(p)
                self._add_accumulator(self._velocity_acc_str, master_p)
                continue
            if (
                self._is_dtype_fp16_or_bf16(p.dtype)
                and not self._multi_precision
            ):
                warnings.warn(
                    "Accumulating with FP16/BF16 in optimizer can lead to poor accuracy or slow convergence."
                    "Consider using multi_precision=True option of the Lars optimizer."
                )
            self._add_accumulator(self._velocity_acc_str, p)

    def _append_optimize_op(self, block, param_and_grad):
        if not isinstance(block, (framework.Block, pir.Block)):
            raise TypeError("block is not instance of Block.")
        _lars_weight_decay = self._lars_weight_decay
        param_name = param_and_grad[0].name
        if len(self._exclude_from_weight_decay) > 0:
            for name in self._exclude_from_weight_decay:
                if name in param_name:
                    _lars_weight_decay = 0.0
                    break

        velocity_acc = self._get_accumulator_master(
            self._velocity_acc_str, param_and_grad[0]
        )
        lr = self._create_param_lr(param_and_grad)

        find_master = self._multi_precision and self._is_dtype_fp16_or_bf16(
            param_and_grad[0].dtype
        )
        master_weight = (
            self._master_weights[param_and_grad[0].name]
            if find_master
            else None
        )

        attrs = {
            "mu": self._momentum,
            "lars_coeff": self._lars_coeff,
            "lars_weight_decay": [_lars_weight_decay],
            "multi_precision": find_master,
            "epsilon": self._epsilon,
            "rescale_grad": self._rescale_grad,
        }

        inputs = {
            "Param": param_and_grad[0],
            "Grad": param_and_grad[1],
            "Velocity": velocity_acc,
            "LearningRate": lr,
        }

        outputs = {"ParamOut": param_and_grad[0], "VelocityOut": velocity_acc}

        if find_master:
            inputs["MasterParam"] = master_weight
            outputs["MasterParamOut"] = master_weight

        if in_dynamic_mode():
            tmp, tmp2 = _legacy_C_ops.lars_momentum(
                [param_and_grad[0]],
                [param_and_grad[1]],
                [velocity_acc],
                [lr],
                [param_and_grad[0]],
                [velocity_acc],
                "mu",
                self._momentum,
                "lars_coeff",
                self._lars_coeff,
                "lars_weight_decay",
                [_lars_weight_decay],
                "multi_precision",
                find_master,
                "epsilon",
                self._epsilon,
                "rescale_grad",
                self._rescale_grad,
            )
        elif in_pir_mode():
            if isinstance(master_weight, pir.Value):
                master_weight = [master_weight]
            _, _, _ = _C_ops.lars_momentum_(
                [param_and_grad[0]],
                [param_and_grad[1]],
                [velocity_acc],
                [lr],
                master_weight,
                self._momentum,
                self._lars_coeff,
                [_lars_weight_decay],
                self._epsilon,
                find_master,
                self._rescale_grad,
            )
            return None
        else:
            # create the momentum optimize op
            momentum_op = block.append_op(
                type=self.type,
                inputs=inputs,
                outputs=outputs,
                attrs=attrs,
                stop_gradient=True,
            )

            return momentum_op
