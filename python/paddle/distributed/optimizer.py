# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from paddle import framework
from paddle.framework import core
from paddle.common_ops_import import LayerHelper
from paddle.fluid.clip import GradientClipByNorm, append_gradient_clip_ops
from paddle.optimizer import Optimizer


class DGCMomentumOptimizer(Optimizer):
    r"""
    :api_attr: Static Graph

    DGC (Deep Gradient Compression) Momentum Optimizer. Original paper is https://arxiv.org/abs/1712.01887

    DGC reduces the communication bandwidth by sending only the important gradients (sparse update):\
        only gradients larger than a threshold are transmitted.

    To avoid losing information, DGC accumulates the rest of the gradients locally.

    Eventually, these gradients become large enough to be transmitted.

    Thus, DGC sends the large gradients immediately but eventually sends all of the gradients over time.

    To ensure no loss of accuracy, DGC employs momentum correction and local gradient clipping on top of the gradient sparsification to maintain model performance.

    DGC also uses momentum factor masking and warmup training to overcome the staleness problem caused by reduced communication.

    This optimizer will do two things:

        1. Compress the gradient by get TopK import value from tensor \
            and use it for allreduce to reduce network bandwidth.

        2. Call momentum to optimize the cost.

    Args:
        learning_rate (float|Variable): The learning rate used to update parameters. \
            It can be a float value or a Variable with one float value as a data element.
        momentum (float): Momentum factor.
        rampup_begin_step (int): The beginning step from which gradient compression is implemented.
        rampup_step (int): Time steps used in sparsity warm-up periods. Default is 1.
            For example, if the sparsity is [0.75, 0.9375, 0.984375, 0.996, 0.999], and the rampup_step is 100, \
                it will use 0.75 at 0~19 steps, and 0.9375 at 20~39 steps, and so on. \
                And when reach sparsity array ends, it will use 0.999 then and after.
        sparsity (list[float]): Get top important element from gradient tensor, the ratio is (1 - current sparsity). \
            Default is [0.999]. For example, if the sparsity is [0.99, 0.999], \
                the top [1%, 0.1%] important element will be transmitted.
        parameter_list (Iterable, optional):  Iterable of ``Variable`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
        use_nesterov (bool): Enables Nesterov momentum. True means use Nesterov. Default is False.
        regularization (WeightDecayRegularizer, optional): The strategy of regularization. There are two method: \
             :ref:`api_fluid_regularizer_L1Decay` , :ref:`api_fluid_regularizer_L2Decay` . If a parameter has set \
            regularizer using :ref:`api_fluid_ParamAttr` already, the regularization setting here in optimizer will be \
            ignored for this parameter. Otherwise, the regularization setting here in optimizer will take effect.  \
            Default None, meaning there is no regularization.
        grad_clip (GradientClipByNorm, optional): Gradient cliping strategy. ``DGCMomentumOptimizer`` only support
            :ref:`api_fluid_clip_GradientClipByNorm` , and if not, it will raise TypeError. Default None,
            meaning there is no gradient clipping.
        name (str, optional): This parameter is used by developers to print debugging information. \
            For details, please refer to :ref:`api_guide_Name`. Default is None.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            optimizer = fluid.optimizer.DGCMomentumOptimizer(
                        learning_rate=0.0001,
                        momentum=0.9,
                        rampup_step=1000,
                        rampup_begin_step=1252,
                        sparsity=[0.999, 0.999])

    """
    _u_velocity_acc_str = "_dgc_u_"
    _v_velocity_acc_str = "_dgc_v_"

    def __init__(
        self,
        learning_rate,
        momentum,
        rampup_begin_step,
        rampup_step=1,
        sparsity=[0.999],
        parameter_list=None,
        use_nesterov=False,
        num_trainers=None,
        regularization=None,
        grad_clip=None,
        name=None,
    ):
        if framework._non_static_mode():
            raise Exception("In dygraph, don't support DGCMomentumOptimizer.")

        assert (
            core.is_compiled_with_cuda()
        ), "Paddle is not compiled with CUDA. DGC is only support GPU for now."

        assert learning_rate is not None
        assert momentum is not None
        super().__init__(
            learning_rate=learning_rate,
            parameter_list=parameter_list,
            regularization=regularization,
            grad_clip=grad_clip,
            name=name,
        )
        self.type = "dgc_momentum"
        self._momentum = momentum
        self._use_nesterov = bool(use_nesterov)

        assert rampup_begin_step >= 0, "rampup_begin_step must >= 0"
        self._rampup_begin_step = rampup_begin_step
        self._rampup_step = rampup_step
        self._sparsity = sparsity

        self._rampup_begin_step_var = None
        self._global_step_var = None

        self._dgc_clip_norm = None
        if grad_clip is not None:
            if not isinstance(grad_clip, GradientClipByNorm):
                raise TypeError(
                    "The type of grad_clip should be 'GradientClipByNorm', because DGCMomentumOptimizer only support GradientClipByNorm"
                )
            assert isinstance(num_trainers, int), (
                "The type of num_trainers should be 'int', but received %s"
                % type(num_trainers)
            )
            assert (
                num_trainers > 0
            ), "The value of num_trainers should be greater than 0!"

            self._num_trainers = num_trainers
            self._dgc_clip_norm = grad_clip.clip_norm * (num_trainers**-0.5)

        self.regular_type, self.regular_coeff = self._get_regularization_param(
            self.regularization
        )

    def _get_regularization_param(self, regularization):
        regular_type = 0
        regular_coeff = 0.0

        if regularization is not None:
            regular_coeff = regularization._regularization_coeff
            from .regularizer import L1Decay, L2Decay

            if isinstance(regularization, L1Decay):
                regular_type = 1
            elif isinstance(regularization, L2Decay):
                regular_type = 2
            else:
                assert False, 'regularization must be None|L1Decay|L2Deacy'
        return regular_type, regular_coeff

    def _is_use_dgc(self, param_var, grad_var):
        var_numel = abs(reduce(lambda x, y: x * y, param_var.shape))
        if (
            var_numel < 16384
            or param_var.type == core.VarDesc.VarType.SELECTED_ROWS
            or grad_var.type == core.VarDesc.VarType.SELECTED_ROWS
            or param_var.dtype != core.VarDesc.VarType.FP32
        ):
            return False
        return True

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)
        velocity_acc = self._get_accumulator(
            self._u_velocity_acc_str, param_and_grad[0]
        )
        assert velocity_acc is not None

        inputs = {
            "Param": param_and_grad[0],
            "Grad": param_and_grad[1],
            "Velocity": velocity_acc,
            "LearningRate": self._create_param_lr(param_and_grad),
        }
        outputs = {
            "ParamOut": param_and_grad[0],
            "VelocityOut": velocity_acc,
        }
        attrs = {"mu": self._momentum, "use_nesterov": self._use_nesterov}

        if not self._is_use_dgc(param_and_grad[0], param_and_grad[1]):
            type = "momentum"
        else:
            type = "dgc_momentum"
            inputs.update(
                {
                    "current_step": self._global_step_var,
                    "nranks": self._nranks_var,
                }
            )
            outputs.update({'Grad_out': param_and_grad[1]})
            attrs.update({"rampup_begin_step": float(self._rampup_begin_step)})

        # create the dgc momentum optimize op
        dgc_momentum_op = block.append_op(
            type=type,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            stop_gradient=True,
        )
        return dgc_momentum_op

    def _add_auto_increment_var(self, counter_name, begin, step=1):
        helper = LayerHelper('global_step_counter')
        counter, is_new_var = helper.create_or_get_global_variable(
            name=counter_name, dtype='float32', shape=[1], persistable=True
        )
        if is_new_var:
            helper.set_variable_initializer(
                counter,
                initializer=Constant(value=float(begin - 1), force_cpu=True),
            )
            helper.main_program.global_block()._prepend_op(
                type='increment',
                inputs={'X': [counter]},
                outputs={'Out': [counter]},
                attrs={'step': float(step)},
                stop_gradient=True,
            )
            counter.stop_gradient = True

        return counter

    def _add_nranks_var(self, name, value=-1):
        helper = LayerHelper('global_step_counter')
        counter, is_new_var = helper.create_or_get_global_variable(
            name=name, dtype='float32', shape=[1], persistable=True
        )
        if is_new_var:
            helper.set_variable_initializer(
                counter,
                initializer=Constant(value=float(value), force_cpu=True),
            )
            counter.stop_gradient = True

        return counter

    def _append_dgc_ops(self, param_and_grads):
        main_program = default_main_program()
        main_program._enable_dgc = True

        # step counter
        self._global_step_var = self._add_auto_increment_var(
            counter_name=core.dgc.kDGCCounterName(), begin=0
        )

        self._nranks_var = self._add_nranks_var(
            name=core.dgc.kDGCNRanksName(), value=-1
        )

        # rampup begin step var for all_reduce_op_handle
        self._rampup_begin_step_var = tensor.create_global_var(
            shape=[1],
            dtype=core.VarDesc.VarType.FP32,
            persistable=True,
            name=core.dgc.kDGCRampUpBeginStepName(),
            value=self._rampup_begin_step * 1.0,
            force_cpu=True,
        )

        self.helper = LayerHelper(self.__class__.__name__)

        for param_var, grad_var in param_and_grads:
            # reuse velocity in dgc_op and dgc_momentum_op
            u_var = self._add_accumulator(self._u_velocity_acc_str, param_var)

            if not self._is_use_dgc(param_var, grad_var):
                continue

            v_var = self._add_accumulator(self._v_velocity_acc_str, param_var)

            k_var = tensor.create_global_var(
                shape=[1],
                dtype=param_var.dtype,
                persistable=True,
                name=param_var.name + core.dgc.kDGCKName(),
                value=0.0,
                force_cpu=True,
            )

            encoded_var = tensor.create_global_var(
                shape=[1],
                dtype=param_var.dtype,
                persistable=True,
                name=param_var.name + core.dgc.kDGCEncodedName(),
                value=0.0,
                force_cpu=False,
            )

            gather_var = tensor.create_global_var(
                shape=[1],
                dtype=param_var.dtype,
                persistable=True,
                name=param_var.name + core.dgc.kDGCGatherName(),
                value=0.0,
                force_cpu=False,
            )

            # del back oprolevarname
            op_maker = core.op_proto_and_checker_maker
            backward = core.op_proto_and_checker_maker.OpRole.Backward
            for op in main_program.global_block().ops:
                if not self._is_the_backward_op(op):
                    continue

                var_attr = op.all_attrs()[op_maker.kOpRoleVarAttrName()]
                if param_var.name not in var_attr:
                    continue

                var_attr.remove(param_var.name)
                var_attr.remove(grad_var.name)
                if len(var_attr) > 1:
                    op._set_attr(op_maker.kOpRoleVarAttrName(), var_attr)
                else:
                    op._remove_attr(op_maker.kOpRoleVarAttrName())

            clip_var = grad_var
            if self._dgc_clip_norm is not None:
                clip_var = self._append_clip_norm(grad_var, self._dgc_clip_norm)
            self._dgc_op(
                param_var,
                clip_var,
                grad_var,
                u_var,
                v_var,
                k_var,
                encoded_var,
                gather_var,
            )

    def _is_the_backward_op(self, op):
        op_maker = core.op_proto_and_checker_maker
        backward = core.op_proto_and_checker_maker.OpRole.Backward
        if op_maker.kOpRoleVarAttrName() in op.attr_names and int(
            op.all_attrs()[op_maker.kOpRoleAttrName()]
        ) == int(backward):
            return True
        return False

    def _clip_by_norm(self, x, max_norm, name=None):
        args = {'x': x, 'max_norm': max_norm, 'name': name}

        helper = LayerHelper("dgc_clip_by_norm_op", **args)

        if name is None:
            name = unique_name.generate_with_ignorable_key(
                ".".join([helper.name, 'tmp'])
            )

        out = helper.create_variable(
            type=x.type, name=name, dtype=x.dtype, persistable=False
        )

        helper.append_op(
            type="dgc_clip_by_norm",
            inputs={"X": x, "current_step": self._global_step_var},
            attrs={
                "max_norm": max_norm,
                "rampup_begin_step": float(self._rampup_begin_step),
            },
            outputs={"Out": out},
        )
        return out

    def _append_clip_norm(self, grad_var, clip_norm):
        with grad_var.block.program._backward_role_guard():
            return self._clip_by_norm(
                x=grad_var, max_norm=clip_norm, name=grad_var.name
            )

    def _dgc_op(
        self,
        param_var,
        clip_var,
        grad_var,
        u_var,
        v_var,
        k_var,
        encoded_var,
        gather_var,
    ):
        block = paddle.static.default_main_program().global_block()
        op_maker = core.op_proto_and_checker_maker

        regular_type = self.regular_type
        regular_coeff = self.regular_coeff
        # The regularizer of the Parameters have higher priority
        if param_var.regularizer is not None:
            regular_type, regular_coeff = self._get_regularization_param(
                param_var.regularizer
            )

        dgc_op = block.append_op(
            type="dgc",
            inputs={
                "U": u_var,
                "V": v_var,
                "Grad": clip_var,
                "Param": param_var,
                "current_step": self._global_step_var,
                "nranks": self._nranks_var,
            },
            outputs={
                "U_out": u_var,
                "V_out": v_var,
                "EncodeGrad": encoded_var,
                "k": k_var,
                "Grad_out": grad_var,
                "GatherBuff": gather_var,
            },
            attrs={
                "m": self._momentum,
                "sparsity": self._sparsity,
                "use_nesterov": self._use_nesterov,
                "rampup_begin_step": float(self._rampup_begin_step),
                "rampup_step": float(self._rampup_step),
                "regular_coeff": float(regular_coeff),
                "regular_type": int(regular_type),
            },
            stop_gradient=True,
        )

        backward = op_maker.OpRole.Backward
        dgc_op._set_attr(op_maker.kOpRoleAttrName(), backward)
        dgc_op._set_attr(
            op_maker.kOpRoleVarAttrName(), [param_var.name, grad_var.name]
        )

    @imperative_base.no_grad
    def apply_gradients(self, params_grads):
        # Note: since we can't use all_reduce_op now,
        # dgc_op should be the last op of one grad.
        # Maybe need a grad allreduce pass.
        self._append_dgc_ops(params_grads)

        params_grads = sorted(params_grads, key=lambda x: x[0].name)
        (
            params_grads,
            table_param_and_grad,
            table_optimize_op,
        ) = self._process_distribute_lookuptable(params_grads)

        not_dgc_params_grads = []
        dgc_params_grads = []
        # DGC clip and regularization in optimizer.backward
        for param, grad in params_grads:
            if not self._is_use_dgc(param, grad):
                not_dgc_params_grads.append((param, grad))
            else:
                dgc_params_grads.append((param, grad))

        # 'optimizer(grad_clip)' or 'set_gradient_clip'
        if self._grad_clip is not None:
            not_dgc_params_grads = self._grad_clip(not_dgc_params_grads)
        else:
            not_dgc_params_grads = append_gradient_clip_ops(
                not_dgc_params_grads
            )

        not_dgc_params_grads = self.append_regularization_ops(
            not_dgc_params_grads, self.regularization
        )

        params_grads = not_dgc_params_grads + dgc_params_grads
        params_grads = sorted(params_grads, key=lambda x: x[0].name)

        optimize_ops = self._create_optimization_pass(params_grads)
        if table_optimize_op is not None:
            optimize_ops.append(table_optimize_op)
            params_grads.append(table_param_and_grad)

        return optimize_ops
