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


import paddle
from paddle.base import core
from paddle.base.framework import (
    Variable,
    default_main_program,
    default_startup_program,
    device_guard,
    in_dygraph_mode,
    program_guard,
)

__all__ = []


class GradientMergeOptimizer:
    """
    Gradient Merge, also called as Gradient Accumulation,
    is a training strategy for larger batches. With this strategy,
    the parameter will not be updated until specific steps.

    For each step, the forward network and the backward network
    will run to calculate the gradient of the parameters.

    For every k step, the optimization network will run,
    applying a specific optimization method (such as SGD, Adam)
    to the parameters.

    Args:
        inner_optimizer (Optimizer): The specific optimization (such as SGD, Adam)
            which update the parameters
        k_steps (int): the update period of the parameters
        avg (bool): whether to average the gradients of each mini-batch,
            the default value is `True`

    Examples:
        .. code-block:: python

        >>> import paddle
        >>> import numpy as np
        >>> paddle.enable_static()

        >>> def gen_data(batch_size):
        ...     return {"x": np.random.random(size=(batch_size, 32)).astype('float32'),
        ...             "y": np.random.random(size=(batch_size, 1)).astype('int64')}

        >>> def mlp(input_x, input_y, hid_dim=128, label_dim=2):
        ...     fc_1 = paddle.static.nn.fc(x=input_x, size=hid_dim)
        ...     prediction = paddle.static.nn.fc(x=[fc_1], size=label_dim, activation='softmax')
        ...     cost = paddle.nn.functional.cross_entropy(
        ...         input=prediction, label=input_y,
        ...         reduction='none', use_softmax=False
        ...     )
        ...     sum_cost = paddle.mean(cost)
        ...     return sum_cost, fc_1, prediction

        >>> input_x = paddle.static.data(name="x", shape=[-1,32], dtype='float32')
        >>> input_y = paddle.static.data(name="y", shape=[-1,1], dtype='int64')
        >>> cost, fc_1, pred = mlp(input_x, input_y)
        >>> sgd = paddle.optimizer.Adam(learning_rate=0.01)
        >>> sgd = paddle.incubate.optimizer.GradientMergeOptimizer(sgd, k_steps=4, avg=True)
        >>> sgd.minimize(cost)

        >>> place = paddle.CPUPlace()
        >>> exe = paddle.static.Executor(place)
        >>> exe.run(paddle.static.default_startup_program())

        >>> for i in range(10):
        ...     cost_val = exe.run(feed=gen_data(32),
        ...                program=paddle.static.default_main_program(),
        ...                fetch_list=[cost.name])
        ...     print("step=%d, cost=%f" % (i, cost_val[0]))
    """

    GRAD_MERGE_COND_NAME = "grad_merge_cond_name"

    def __init__(self, inner_optimizer, k_steps=1, avg=True):
        if in_dygraph_mode():
            raise Exception(
                "In dygraph, we don't support GradientMergeOptimizer."
                "You can do Gradient merge by yourself with k-times forward + backward, "
                "and one-time optimizer.minimize()"
            )

        assert inner_optimizer is not None, "inner optimizer can not be None"
        assert (
            isinstance(k_steps, int) and k_steps > 0
        ), "k_steps should be a positive integer"

        self.inner_optimizer = inner_optimizer
        self.k_steps = k_steps
        self.type = "gradient_merge"
        self.avg = avg
        self._optimize_ops = None

    def _set_k_steps(self, k_steps):
        self.k_steps = k_steps

    def _set_avg(self, avg):
        self.avg = avg

    def backward(
        self,
        loss,
        startup_program=None,
        parameter_list=None,
        no_grad_set=None,
        callbacks=None,
    ):
        assert isinstance(loss, Variable), "The loss should be an Variable."
        assert (
            parameter_list is None
        ), "The parameter_list should be None when using GradientMergeOptimizer"
        assert (
            no_grad_set is None
        ), "The no_grad_set should be None when using GradientMergeOptimizer"

        params_grads = self.inner_optimizer.backward(
            loss, startup_program=startup_program
        )
        return params_grads

    def apply_optimize(self, loss, startup_program, params_grads):
        program = loss.block.program
        with program_guard(program, startup_program):
            optimize_ops = self.apply_gradients(params_grads)
        return optimize_ops

    def _is_the_backward_op(self, op):
        op_maker = core.op_proto_and_checker_maker
        backward = core.op_proto_and_checker_maker.OpRole.Backward
        if op_maker.kOpRoleVarAttrName() in op.attr_names and int(
            op.all_attrs()[op_maker.kOpRoleAttrName()]
        ) == int(backward):
            return True
        return False

    def _remove_op_role_var(self, param, grad):
        op_maker = core.op_proto_and_checker_maker
        op = grad.op
        assert self._is_the_backward_op(
            op
        ), f'grad.op={op} is not the backward op which produces the grad={grad.name}'

        block = grad.block
        var_attr = op.all_attrs()[op_maker.kOpRoleVarAttrName()]
        assert (
            param.name in var_attr
        ), f'when using GradientMergeOptimizer, param={param.name} must be in var_attr={var_attr}'
        assert (
            grad.name in var_attr
        ), f'when using GradientMergeOptimizer, grad={param.name} must be in var_attr={var_attr}'

        # remove (param, grad) from op_role_var
        var_attr.remove(param.name)
        var_attr.remove(grad.name)
        if len(var_attr) > 1:
            op._set_attr(op_maker.kOpRoleVarAttrName(), var_attr)
        else:
            op._remove_attr(op_maker.kOpRoleVarAttrName())

    def _add_gm_op_role_var(self, op, param, grad, cond):
        grad.op = op
        op_maker = core.op_proto_and_checker_maker
        backward = op_maker.OpRole.Backward

        # NOTE(wangxi). When distributed, we will insert grad_merge_all_reduce_op_handle
        # in multi_devices_graph_pass, which will allreduce(grad) if cond is True, else
        # do nothing.
        # In this way, the gradient can be merged first, and then communicate when the
        # condition is met, reducing the number of communications to increase the
        # speed.
        op._set_attr(self.GRAD_MERGE_COND_NAME, cond.name)
        op._set_attr(op_maker.kOpRoleAttrName(), backward)
        op._set_attr(op_maker.kOpRoleVarAttrName(), [param.name, grad.name])

    def _get_gm_cond_var(self, main_block):
        # Add const var
        k_step_var = paddle.static.create_global_var(
            name="gradient_merge_k",
            shape=[1],
            value=int(self.k_steps),
            dtype='int32',
            persistable=True,
            force_cpu=True,
        )

        zero_var = paddle.static.create_global_var(
            name="gradient_merge_zero",
            shape=[1],
            value=0,
            dtype='int32',
            persistable=True,
            force_cpu=True,
        )

        # Add step var & cond var
        step_var = paddle.static.create_global_var(
            name="gradient_merge_step",
            shape=[1],
            value=0,
            dtype='int32',
            persistable=True,
            force_cpu=True,
        )

        cond_var = main_block.create_var(
            name="gradient_merge_cond", shape=[1], dtype='bool'
        )

        with device_guard("cpu"):
            # step_var = (step_var + 1) % k_step
            paddle.increment(x=step_var, value=1.0)
            main_block.append_op(
                type='elementwise_mod',
                inputs={'X': step_var, 'Y': k_step_var},
                outputs={'Out': step_var},
                attrs={'axis': -1},
            )

            # cond_var = (step_var == 0)
            main_block.append_op(
                type='equal',
                inputs={'X': step_var, 'Y': zero_var},
                outputs={'Out': cond_var},
            )

        return cond_var

    def apply_gradients(self, params_grads):
        main_program = default_main_program()
        startup_program = default_startup_program()
        main_block = main_program.global_block()
        startup_block = startup_program.global_block()

        cond = self._get_gm_cond_var(main_block)

        # TODO(mapingshuo) support sparse embedding
        # step1: remove grad.op's op_role_var
        for param, grad in params_grads:
            assert (
                param.type != core.VarDesc.VarType.SELECTED_ROWS
            ), "SELECTED_ROWS is not supported in GradientMergeOptimizer for now"

            self._remove_op_role_var(param, grad)

        param_to_grad = {k.name: v for (k, v) in params_grads}
        param_names = param_to_grad.keys()
        param_to_gradient_merge = {}

        new_params_grads = []
        # step2: create gradient_merge var and init with 0
        # and update op_role_var
        for param, grad in params_grads:
            param_name = param.name
            param_var = main_block.var(param_name)
            assert param_var is not None
            gradient_merge_var = main_block.create_var(
                name=param_name + "@GRAD@GradientMerge",
                shape=param_var.shape,
                dtype=param_var.dtype,
                persistable=True,
            )
            param_to_gradient_merge[param_name] = gradient_merge_var

            startup_gradient_merge_var = startup_block.create_var(
                name=param_name + "@GRAD@GradientMerge",
                shape=param_var.shape,
                dtype=param_var.dtype,
                persistable=True,
            )
            startup_block.append_op(
                type="fill_constant",
                outputs={"Out": startup_gradient_merge_var},
                attrs={
                    "shape": param_var.shape,
                    "dtype": param_var.dtype,
                    "value": float(0),
                },
            )

            # grad_merge += grad
            new_grad_op = main_block.append_op(
                type="elementwise_add",
                inputs={'X': grad, 'Y': gradient_merge_var},
                outputs={'Out': gradient_merge_var},
                attrs={'axis': -1},
            )
            self._add_gm_op_role_var(
                new_grad_op, param, gradient_merge_var, cond
            )
            new_params_grads.append([param, gradient_merge_var])

        def true_apply_gradient():
            cur_block_idx = main_program.current_block_idx
            cur_block = main_program.current_block()

            # cur_block's forward_block & backward_block is itself
            cur_block._set_forward_block_idx(cur_block_idx)
            op_maker = core.op_proto_and_checker_maker

            if self.avg:
                for param, new_grad in new_params_grads:
                    # grad /= k_steps
                    cur_block.append_op(
                        type='scale',
                        inputs={'X': new_grad},
                        outputs={'Out': new_grad},
                        attrs={
                            'scale': 1.0 / self.k_steps,
                            'bias': 0.0,
                            'bias_after_scale': False,
                        },
                    )
                    new_grad.op._set_attr(
                        op_maker.kOpRoleAttrName(), op_maker.OpRole.Backward
                    )

            for param, new_grad in new_params_grads:
                # NOTE. regularization will append ops to grad.block,
                # while new_grad's real block is global_block,
                # but we want append regularization ops to cur_block,
                # so we set new_grad.block = cur_block
                new_grad.block = cur_block

            self._optimize_ops = self.inner_optimizer.apply_gradients(
                new_params_grads
            )

            # clear gradient_merge_vars
            for param, new_grad in new_params_grads:
                paddle.tensor.fill_constant(
                    shape=new_grad.shape,
                    dtype=new_grad.dtype,
                    value=0.0,
                    out=new_grad,
                )
                new_grad.op._set_attr(
                    op_maker.kOpRoleAttrName(), op_maker.OpRole.Optimize
                )

        # step3. apply gradient
        paddle.static.nn.cond(cond, true_fn=true_apply_gradient, false_fn=None)

        return self._optimize_ops

    def minimize(
        self, loss, startup_program=None, parameter_list=None, no_grad_set=None
    ):
        assert isinstance(loss, Variable), "The loss should be an Variable."

        params_grads = self.backward(
            loss,
            startup_program=startup_program,
            parameter_list=parameter_list,
            no_grad_set=no_grad_set,
        )

        optimize_ops = self.apply_optimize(
            loss, startup_program=startup_program, params_grads=params_grads
        )

        return optimize_ops, params_grads
