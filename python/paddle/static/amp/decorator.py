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

import types
import warnings

import paddle
from paddle.base import (
    core,
    default_main_program,
    default_startup_program,
    program_guard,
    unique_name,
)
from paddle.base.framework import auto_complete_op_role, in_pir_mode

from .amp_nn import check_finite_and_unscale, update_loss_scaling
from .fp16_lists import AutoMixedPrecisionLists, check_amp_dtype
from .fp16_utils import (
    cast_model_to_fp16,
    cast_parameters_to_fp16,
    update_role_var_grad,
)
from .function_overload import FunctionType, overload

OpRole = core.op_proto_and_checker_maker.OpRole


def _set_multi_precision(optimizer, multi_precision):
    if not isinstance(
        optimizer,
        (paddle.optimizer.Optimizer),
    ):
        raise RuntimeError(
            f"Current AMP training level is O2, optimizer is expected to be paddle.optimizer.Optimizer, but receive {type(optimizer)}."
        )

    if multi_precision and hasattr(optimizer, "_multi_precision"):
        optimizer._multi_precision = multi_precision


class OptimizerWithMixedPrecision:
    """
    Optimizer with mixed-precision (MP) training. This is a wrapper of a common
    optimizer, plus the support of mixed-precision pre-training. The object
    of this class almost has the same behavior as the common optimizer, with the
    methods `minimize()`, `backward()`, `apply_gradients()` implemented.
    Additionally, it enables the MP training automatically, i.e, the creation
    and maintenance of master parameters, scaling of loss, etc.

    Args:
        optimizer (Optimizer): A common Optimizer object.
        amp_lists (AutoMixedPrecisionLists): An AutoMixedPrecisionLists object.
        level(str): Auto mixed precision level. Accepted values are "O1", "O2" and "OD": At the O1 level, operators in the white list
             will use float16/bfloat16 inputs for calculations, and operators in the black list will use float32 inputs for calculations. At the O2
             level, model's parameters will be casted to float16/bfloat16 by using `decorator`, and operators that have all float16/bfloat16 inputs
             will be converted to float16/bfloat16, and that have any float32 input will be converted to float32. For the OD level, operators in
             default white list will compute in float16/bfloat16.
        dtype(str): Whether to use 'float16' or 'bfloat16'.
        init_loss_scaling (float): The initial loss scaling factor.
        use_dynamic_loss_scaling (bool): Whether to use dynamic loss scaling.
        incr_every_n_steps(int): Increases loss scaling every n consecutive
                                 steps with finite gradients.
        decr_every_n_nan_or_inf(int): Decreases loss scaling every n
                                      accumulated steps with nan or
                                      inf gradients.
        incr_ratio(float): The multiplier to use when increasing the loss
                           scaling.
        decr_ratio(float): The less-than-one-multiplier to use when decreasing
                           the loss scaling.
        use_amp_guard(bool): Whether to use `fp16_guard` when constructing the program.
                           Default None, which means that its value is equal to `use_pure_fp16`.
        use_master_grad(bool): Whether to use fp32 master gradients during optimizer. Default is False.
        use_promote(bool): Whether to promotes to fp32 when op has any float32 inputs. Default is False.
    """

    def __init__(
        self,
        optimizer,
        amp_lists,
        level,
        dtype,
        init_loss_scaling,
        use_dynamic_loss_scaling,
        incr_every_n_steps,
        decr_every_n_nan_or_inf,
        incr_ratio,
        decr_ratio,
        use_amp_guard=None,
        use_master_grad=False,
        use_promote=False,
    ):
        self._optimizer = optimizer
        self._amp_lists = amp_lists
        self._param_grads = None
        self._train_program = None

        self._is_distributed = False
        self._use_master_grad = False
        self._scaled_loss = None
        self._loss_scaling = None
        self._init_loss_scaling = init_loss_scaling
        self._use_dynamic_loss_scaling = use_dynamic_loss_scaling
        if dtype == "bfloat16":
            if use_dynamic_loss_scaling:
                self._use_dynamic_loss_scaling = False
                self._init_loss_scaling = 1.0
                warnings.warn(
                    "Dynamic loss scaling for bfloat16 amp training is disabled, and the init_loss_scaling is changed to 1.0 automatically by PaddlePaddle."
                )
            if in_pir_mode():
                self._amp_vartype = core.DataType.BFLOAT16
            else:
                self._amp_vartype = core.VarDesc.VarType.BF16
        else:
            if in_pir_mode():
                self._amp_vartype = core.DataType.FLOAT16
            else:
                self._amp_vartype = core.VarDesc.VarType.FP16

        self._learning_rate = optimizer._learning_rate
        self._learning_rate_map = optimizer._learning_rate_map
        self._use_pure_fp16 = level == "O2"
        if self._use_pure_fp16 and (dtype == "bfloat16" or dtype == "float16"):
            self._use_master_grad = use_master_grad
            self._optimizer._master_grad = use_master_grad
        self._amp_level = level
        self._use_fp16_guard = use_amp_guard
        self._to_fp16_var_names = None
        if self._use_dynamic_loss_scaling:
            self._incr_every_n_steps = incr_every_n_steps
            self._decr_every_n_nan_or_inf = decr_every_n_nan_or_inf
            self._incr_ratio = incr_ratio
            self._decr_ratio = decr_ratio
            self._num_good_steps = None
            self._num_bad_steps = None
        self.use_promote = use_promote

    def _set_distributed(self, flag):
        # if distributed, all cards will communication with each other,
        # overlap communication and computation by split the
        # check_finite_and_unscale op.
        self._is_distributed = flag

    def get_loss_scaling(self):
        """Return the real-time loss scaling factor."""
        assert (
            self._loss_scaling is not None
        ), 'Please call minimize() before calling get_loss_scaling().'
        return self._loss_scaling

    def get_scaled_loss(self):
        """Return the scaled loss.
        It's useful when you feed customed loss into executor.
        """
        return self._scaled_loss

    def _supports_check_nan_inf(self):
        return getattr(self._optimizer, "_supports_check_nan_inf", False)

    def _init_amp_var(self):
        if in_pir_mode():
            if self._use_dynamic_loss_scaling:
                self._num_good_steps = paddle.pir.core.create_persistable_value(
                    dtype='int32',
                    shape=[1],
                    name=unique_name.generate("num_good_steps"),
                    initializer=paddle.nn.initializer.ConstantInitializer(
                        value=0
                    ),
                )
                self._num_bad_steps = paddle.pir.core.create_persistable_value(
                    dtype='int32',
                    shape=[1],
                    name=unique_name.generate("num_bad_steps"),
                    initializer=paddle.nn.initializer.ConstantInitializer(
                        value=0
                    ),
                )

            if isinstance(self._optimizer._learning_rate, float):
                self._optimizer._learning_rate_map[
                    paddle.static.default_main_program()
                ] = paddle.pir.core.create_persistable_value(
                    dtype='float32',
                    shape=[1],
                    name=unique_name.generate("learning_rate"),
                    initializer=paddle.nn.initializer.ConstantInitializer(
                        value=float(self._optimizer._learning_rate)
                    ),
                )

            return

        self._loss_scaling = paddle.static.create_global_var(
            name=unique_name.generate("loss_scaling"),
            shape=[1],
            value=self._init_loss_scaling,
            dtype='float32',
            persistable=True,
        )

        if self._use_dynamic_loss_scaling:
            self._num_good_steps = paddle.static.create_global_var(
                name=unique_name.generate("num_good_steps"),
                shape=[1],
                value=0,
                dtype='int32',
                persistable=True,
            )
            self._num_bad_steps = paddle.static.create_global_var(
                name=unique_name.generate("num_bad_steps"),
                shape=[1],
                value=0,
                dtype='int32',
                persistable=True,
            )

        # Ensure the data type of learning rate vars is float32 (same as the
        # master parameter dtype)
        if isinstance(self._optimizer._learning_rate, float):
            self._optimizer._learning_rate_map[default_main_program()] = (
                paddle.static.create_global_var(
                    name=unique_name.generate("learning_rate"),
                    shape=[1],
                    value=float(self._optimizer._learning_rate),
                    dtype='float32',
                    persistable=True,
                )
            )

    def backward(
        self,
        loss,
        startup_program=None,
        parameter_list=None,
        no_grad_set=None,
        callbacks=None,
    ):
        """
        Backward propagation or auto differentiation for gradients' computation.

        Args:
            loss (Variable): The loss Variable to minimize.
            startup_program (Program|None): The startup Program for initializing
                                       parameters in `parameter_list`.
            parameter_list (list|None): A list of Variables to update.
            no_grad_set (set|None): A set of Variables should be ignored.
            callbacks (list|None): A list of callable objects to run when appending
                                   backward operator for one parameter.

        Returns:
            A list of (param, grad), which is a tuple of a parameter and its
            gradient respectively, and the scaled loss.
        """
        train_program = loss.block.program
        self._train_program = train_program
        self._float_status = None

        if in_pir_mode():
            with paddle.static.program_guard(
                self._train_program, startup_program
            ):
                self._init_amp_var()
                if self._scaled_loss is None:
                    self._scaled_loss = loss
                params_grads = self._optimizer.backward(
                    self._scaled_loss,
                    startup_program,
                    parameter_list,
                    no_grad_set,
                    callbacks,
                )
                return params_grads

        with program_guard(self._train_program, startup_program):
            self._init_amp_var()

            if self._use_pure_fp16:
                self._to_fp16_var_names = cast_model_to_fp16(
                    self._train_program,
                    self._amp_lists,
                    self._use_fp16_guard,
                    self._amp_vartype,
                    level='O2',
                    use_promote=self.use_promote,
                )
            else:
                # use_fp16_guard is not support amp-o1.
                cast_model_to_fp16(
                    self._train_program,
                    self._amp_lists,
                    use_fp16_guard=False,
                    dest_type=self._amp_vartype,
                    level=self._amp_level,
                    use_promote=self.use_promote,
                )

            if loss.dtype != core.VarDesc.VarType.FP32:
                loss = loss.astype('float32')
            # When not using dynamic loss scaling and the init loss scaling value is equal to 1.0,
            # the model can be optimized.
            if self._use_dynamic_loss_scaling or self._init_loss_scaling != 1.0:
                self._scaled_loss = loss * self._loss_scaling
            else:
                self._scaled_loss = loss

            params_grads = self._optimizer.backward(
                self._scaled_loss,
                startup_program,
                parameter_list,
                no_grad_set,
                callbacks,
            )
            if self._supports_check_nan_inf():
                self._add_cast_ops_to_startup_program(startup_program)
        return params_grads

    def _add_cast_ops_to_startup_program(self, startup_program):
        names = list(self._to_fp16_var_names) if self._to_fp16_var_names else []
        names.sort()
        startup_program = (
            default_startup_program()
            if startup_program is None
            else startup_program
        )
        block = startup_program.global_block()
        param_names = [p.name for p in block.all_parameters()]
        for name in names:
            if name not in param_names:
                continue

            tmp = block.create_var(dtype=core.VarDesc.VarType.FP32)
            block.append_op(
                type='assign', inputs={'X': [name]}, outputs={'Out': [tmp]}
            )
            block.append_op(
                type='cast',
                inputs={'X': [tmp]},
                outputs={'Out': [name]},
                attrs={
                    'in_dtype': core.VarDesc.VarType.FP32,
                    'out_dtype': self._amp_vartype,
                },
            )
        self._to_fp16_var_names = None

    def amp_init(
        self,
        place,
        scope=None,
        test_program=None,
        use_fp16_test=False,
        rewrite_master_weight=False,
    ):
        """
        Init the amp training, such as cast fp32 parameters to fp16 type.

        Args:
            place(CUDAPlace): place is used to initialize
                fp16 parameters with fp32 values.
            scope(Scope): The scope is used to find fp32 parameters.
            test_program(Program): The program is used for testing.
            use_fp16_test(bool): Whether to use fp16 testing.

        Examples:
            .. code-block:: python

                >>> import numpy as np
                >>> import paddle
                >>> import paddle.nn.functional as F
                >>> paddle.enable_static()

                >>> # doctest: +REQUIRES(env:GPU)
                >>> def run_example_code():
                ...     place = paddle.CUDAPlace(0)
                ...     exe = paddle.static.Executor(place)
                ...     data = paddle.static.data(name='X', shape=[None, 1, 28, 28], dtype='float32')
                ...     conv2d = paddle.static.nn.conv2d(input=data, num_filters=6, filter_size=3)
                ...     # 1) Use fp16_guard to control the range of fp16 kernels used.
                ...     with paddle.static.amp.fp16_guard():
                ...         bn = paddle.static.nn.batch_norm(input=conv2d, act="relu")
                ...         pool = F.max_pool2d(bn, kernel_size=2, stride=2)
                ...         hidden = paddle.static.nn.fc(pool, size=10)
                ...         loss = paddle.mean(hidden)
                ...     # 2) Create the optimizer and set `multi_precision` to True.
                ...     # Setting `multi_precision` to True can avoid the poor accuracy
                ...     # or the slow convergence in a way.
                ...     optimizer = paddle.optimizer.Momentum(learning_rate=0.01, multi_precision=True)
                ...     # 3) These ops in `custom_black_list` will keep in the float32 computation type.
                ...     amp_list = paddle.static.amp.CustomOpLists(
                ...         custom_black_list=['pool2d'])
                ...     # 4) The entry of Paddle AMP.
                ...     # Enable pure fp16 training by setting `use_pure_fp16` to True.
                ...     optimizer = paddle.static.amp.decorate(
                ...         optimizer,
                ...         amp_list,
                ...         init_loss_scaling=128.0,
                ...         use_dynamic_loss_scaling=True,
                ...         use_pure_fp16=True)
                ...     # If you don't use the default_startup_program(), you should pass
                ...     # your defined `startup_program` into `minimize`.
                ...     optimizer.minimize(loss)
                ...     exe.run(paddle.static.default_startup_program())
                ...     # 5) Use `amp_init` after FP32 parameters initialization(such as `exe.run(startup_program)`).
                ...     # If you want to perform the testing process, you should pass `test_program` into `amp_init`.
                ...     optimizer.amp_init(place, scope=paddle.static.global_scope())

                >>> if paddle.is_compiled_with_cuda() and len(paddle.static.cuda_places()) > 0:
                ...     run_example_code()
        """
        assert (
            self._train_program is not None
        ), "Please call the minimize method first."
        if self._use_pure_fp16:
            cast_parameters_to_fp16(
                place,
                self._train_program,
                scope,
                self._to_fp16_var_names,
                self._amp_vartype,
                rewrite_master_weight,
                self._optimizer._master_weights,
            )
        if test_program is not None:
            if self._use_pure_fp16:
                cast_model_to_fp16(
                    test_program,
                    self._amp_lists,
                    self._use_fp16_guard,
                    self._amp_vartype,
                    level='O2',
                    use_promote=self.use_promote,
                )
            elif use_fp16_test:
                # use_fp16_guard is not support amp-o1.
                cast_model_to_fp16(
                    test_program,
                    self._amp_lists,
                    use_fp16_guard=False,
                    dest_type=self._amp_vartype,
                    level=self._amp_level,
                    use_promote=self.use_promote,
                )

    def _append_cast_to_master_grad_op(self, param_grads):
        """
        Create master gradient vars and add cast gradient to master gradient op in main program

        Args:
          param_grads(list(tuple(Tensor, Tensor))): A list of (parameter, gradient) pair to update.

        Returns:
          list: A list of (parameter, master_gradient) pair. In the following grad clip step and optimizer step, params can be updated by master gradient. main_prog will also append cast ops before grad clip ops.

        """

        if not self._use_master_grad:
            return param_grads

        global_block = self._train_program.global_block()
        target_block = global_block
        if not in_pir_mode():
            current_block = self._train_program.current_block()
            if current_block.idx != global_block.idx:
                target_block = self._train_program.blocks[
                    current_block.backward_block_idx
                ]
        params_master_grads = []

        assert isinstance(
            target_block, (paddle.base.framework.Block, paddle.pir.Block)
        )

        if in_pir_mode():
            for p, g in param_grads:
                if g not in self._optimizer._master_grads:
                    if self._optimizer._is_dtype_fp16_or_bf16(g.dtype):
                        master_g = self._optimizer._create_master_grad(g)
                        params_master_grads.append((p, master_g))
                    else:
                        params_master_grads.append((p, g))
        else:
            # create
            for p, g in param_grads:
                if g.name not in self._optimizer._master_grads.keys():
                    if self._optimizer._is_dtype_fp16_or_bf16(g.dtype):
                        master_g = self._optimizer._create_master_grad(g)
                        params_master_grads.append((p, master_g))
                        target_block.append_op(
                            type="cast",
                            inputs={"X": [g]},
                            outputs={"Out": [master_g]},
                            attrs={
                                "in_dtype": g.dtype,
                                "out_dtype": master_g.dtype,
                            },
                        )
                    else:
                        params_master_grads.append((p, g))

        return params_master_grads

    def apply_gradients(self, params_grads):
        """
        Check scaled gradients to determine whether to update loss scaling and update
        parameters by their scaled gradients.

        Args:
            params_grads (list): A list of params and scaled grads.

        Returns:
            A list of optimize operators.
        """

        if not in_pir_mode():
            # Change the op_role_var attr for some ops, so that gradients
            # transferred across GPUs can be FP16.
            update_role_var_grad(self._train_program, params_grads)

        # Create master grad and add cast op into program
        params_grads = self._append_cast_to_master_grad_op(params_grads)

        # When not using dynamic loss scaling and the init loss scaling value is equal to 1.0,
        # the model can be optimized.
        if (
            not self._use_dynamic_loss_scaling
            and self._init_loss_scaling == 1.0
        ):
            return self._optimizer.apply_gradients(params_grads)

        if self._supports_check_nan_inf():
            self._optimizer._set_scale(self._loss_scaling)
            optimize_ops = self._optimizer.apply_gradients(params_grads)
            found_inf = self._optimizer._found_inf
            self._add_dynamic_loss_scaling(params_grads, found_inf)
            return optimize_ops

        found_inf = self._check_finite_and_unscale(params_grads)
        if self._use_dynamic_loss_scaling and (
            self._amp_vartype == paddle.float16
            or self._amp_vartype == core.DataType.FLOAT16
        ):
            self._add_dynamic_loss_scaling(params_grads, found_inf)

        # Pass found_inf to adam, to skip update for not only param, but also momentum and beta_pow
        # With fleet, optimizers are nested and the real optimizer set by user is the inner most one.
        real_optimizer = self._optimizer
        while hasattr(real_optimizer, "inner_opt"):
            real_optimizer = real_optimizer.inner_opt
        if isinstance(
            real_optimizer,
            (paddle.optimizer.Adam, paddle.optimizer.AdamW),
        ):
            # NOTE(zhiqiu): Since found_inf needs to be on cpu in adam op, we
            # copy it in advance to avoid multiple time copies.
            with self._train_program._optimized_guard([]):
                found_inf = paddle.tensor.creation._memcpy(
                    found_inf, paddle.CPUPlace()
                )
            real_optimizer._set_auxiliary_var('found_inf', found_inf)
        elif hasattr(real_optimizer, "_set_auxiliary_var"):
            real_optimizer._set_auxiliary_var('found_inf', found_inf)
        optimize_ops = self._optimizer.apply_gradients(params_grads)
        return optimize_ops

    def _split_grads(self, params_grads):
        grads = [g for _, g in params_grads]
        fp32_grads = [
            g
            for g in grads
            if g.dtype == paddle.float32 or g.dtype == core.DataType.FLOAT32
        ]
        fp16_grads = [g for g in grads if g.dtype == self._amp_vartype]
        assert len(fp32_grads) + len(fp16_grads) == len(
            grads
        ), "Data types of all grads must be either fp16/bf16 or fp32."
        return grads, fp32_grads, fp16_grads

    def _check_finite_and_unscale(self, params_grads):
        grads, fp32_grads, fp16_grads = self._split_grads(params_grads)
        found_infs = []

        if self._is_distributed:
            # if distributed, split check_finite_and_unscale to overlap
            # unscale with communication
            for p, g in params_grads:
                with self._train_program._optimized_guard([p, g]):
                    _, found_inf = check_finite_and_unscale(
                        [
                            g,
                        ],
                        self._loss_scaling,
                        name="find_infinite_scale",
                        float_status=self._float_status,
                    )
                    found_infs.append(found_inf)
        elif self._use_pure_fp16:
            if fp32_grads:
                with self._train_program._optimized_guard(fp32_grads):
                    _, fp32_found_inf = check_finite_and_unscale(
                        fp32_grads,
                        self._loss_scaling,
                        name="find_infinite_scale_fp32",
                        float_status=self._float_status,
                    )
                found_infs.append(fp32_found_inf)
            if fp16_grads:
                with self._train_program._optimized_guard(fp16_grads):
                    _, fp16_found_inf = check_finite_and_unscale(
                        fp16_grads,
                        self._loss_scaling,
                        name="find_infinite_scale_fp16",
                        float_status=self._float_status,
                    )
                found_infs.append(fp16_found_inf)
        else:
            with self._train_program._optimized_guard(grads):
                _, found_inf = check_finite_and_unscale(
                    grads,
                    self._loss_scaling,
                    name="find_infinite_scale",
                    float_status=self._float_status,
                )
                found_infs.append(found_inf)

        if len(found_infs) > 1:
            with self._train_program._optimized_guard([]):
                all_infs = paddle.concat(found_infs)
                found_inf = paddle.any(all_infs)
        else:
            found_inf = found_infs[0]

        return found_inf

    def _add_dynamic_loss_scaling(self, params_grads, found_inf):
        if self._supports_check_nan_inf():
            with self._train_program._optimized_guard([]):
                update_loss_scaling(
                    [],
                    found_inf,
                    self._loss_scaling,
                    self._num_good_steps,
                    self._num_bad_steps,
                    self._incr_every_n_steps,
                    self._decr_every_n_nan_or_inf,
                    self._incr_ratio,
                    self._decr_ratio,
                    stop_update=self._optimizer._get_stop_update_var(),
                    name="update_loss_scaling",
                )
            return

        grads, fp32_grads, fp16_grads = self._split_grads(params_grads)
        if self._use_pure_fp16:
            stop_update = False
            with self._train_program._optimized_guard([]):
                if fp32_grads:
                    update_loss_scaling(
                        fp32_grads,
                        found_inf,
                        self._loss_scaling,
                        self._num_good_steps,
                        self._num_bad_steps,
                        self._incr_every_n_steps,
                        self._decr_every_n_nan_or_inf,
                        self._incr_ratio,
                        self._decr_ratio,
                        stop_update=stop_update,
                        name="update_loss_scaling_fp32",
                    )
                    stop_update = True
                if fp16_grads:
                    update_loss_scaling(
                        fp16_grads,
                        found_inf,
                        self._loss_scaling,
                        self._num_good_steps,
                        self._num_bad_steps,
                        self._incr_every_n_steps,
                        self._decr_every_n_nan_or_inf,
                        self._incr_ratio,
                        self._decr_ratio,
                        stop_update=stop_update,
                        name="update_loss_scaling_fp16",
                    )
        else:
            with self._train_program._optimized_guard([]):
                update_loss_scaling(
                    grads,
                    found_inf,
                    self._loss_scaling,
                    self._num_good_steps,
                    self._num_bad_steps,
                    self._incr_every_n_steps,
                    self._decr_every_n_nan_or_inf,
                    self._incr_ratio,
                    self._decr_ratio,
                    name="update_loss_scaling",
                )

    def apply_optimize(self, loss, startup_program, params_grads):
        program = loss.block.program
        with paddle.static.program_guard(program, startup_program):
            optimize_ops = self.apply_gradients(params_grads)
        return optimize_ops

    def minimize(
        self, loss, startup_program=None, parameter_list=None, no_grad_set=None
    ):
        """
        Perform optimization by minimizing the given loss.

        Args:
            loss (Variable): The loss Variable.
            startup_program (Program): startup_program for initializing parameters
                in `parameter_list`.
            parameter_list (list): list of Variables to update.
            no_grad_set (set|None): set of Variables should be ignored.

        Returns:
            The scaled loss by scaling factor, the list of optimize ops, and a
            list of scaled parameters and gradients.
        """

        opt_dict = self._optimizer.__class__.__dict__
        if 'minimize' in opt_dict and isinstance(
            opt_dict['minimize'], types.FunctionType
        ):
            warnings.warn(
                "The decorated optimizer has its own `minimize` method, but it will not be executed."
            )

        with auto_complete_op_role(loss.block.program, op_role=OpRole.Backward):
            scaled_params_grads = self.backward(
                loss,
                startup_program=startup_program,
                parameter_list=parameter_list,
                no_grad_set=no_grad_set,
            )

        with auto_complete_op_role(loss.block.program, op_role=OpRole.Optimize):
            optimize_ops = self.apply_optimize(
                loss, startup_program, scaled_params_grads
            )

        return optimize_ops, scaled_params_grads


@overload(key=FunctionType.FP16_ONLY)
def decorate(
    optimizer,
    amp_lists=None,
    init_loss_scaling=2**15,
    incr_every_n_steps=1000,
    decr_every_n_nan_or_inf=2,
    incr_ratio=2.0,
    decr_ratio=0.8,
    use_dynamic_loss_scaling=True,
    use_pure_fp16=False,
    use_fp16_guard=None,
    use_bf16=False,
    use_promote=False,
):
    """
    Decorate the given optimizer to adapt to the mixed-precision training.

    Args:
        optimizer(Optimizer): A common Optimizer.
        amp_lists (CustomOpLists): An CustomOpLists object.
        init_loss_scaling(float): The initial loss scaling factor.
        incr_every_n_steps(int): Increases loss scaling every n consecutive
                                 steps with finite gradients.
        decr_every_n_nan_or_inf(int): Decreases loss scaling every n
                                      accumulated steps with nan or
                                      inf gradients.
        incr_ratio(float): The multiplier to use when increasing the loss
                           scaling.
        decr_ratio(float): The less-than-one-multiplier to use when decreasing
                           the loss scaling.
        use_dynamic_loss_scaling(bool): Whether to use dynamic loss scaling.
        use_pure_fp16(bool): Whether to use the pure fp16 training. Default False.
        use_fp16_guard(bool): Whether to use `fp16_guard` when constructing the program.
                           Default None, which means that its value equals to `use_pure_fp16`.
        use_bf16(bool): Whether to enable bfloat16 training. Default False.

    Returns:
        An optimizer acting like a normal one but with mixed-precision training
        enabled.

    Examples:
        .. code-block:: python
            :name: example-1

            # black&white list based strategy example
            >>> import paddle
            >>> import paddle.static as static

            >>> paddle.enable_static()

            >>> data = static.data(name='X', shape=[None, 1], dtype='float32')
            >>> hidden = static.nn.fc(x=data, size=10)
            >>> loss = paddle.mean(hidden)
            >>> optimizer = paddle.optimizer.Adam(learning_rate=0.001)

            >>> mp_optimizer = static.amp.decorate(
            ...         optimizer=optimizer, init_loss_scaling=8.0)

            >>> ops, param_grads = mp_optimizer.minimize(loss)
            >>> scaled_loss = mp_optimizer.get_scaled_loss()


        .. code-block:: python
            :name: example-2

            # pure fp16 training example
            >>> import numpy as np
            >>> import paddle
            >>> import paddle.nn.functional as F
            >>> paddle.enable_static()

            >>> # doctest: +REQUIRES(env:GPU)
            >>> def run_example_code():
            ...     place = paddle.CUDAPlace(0)
            ...     exe = paddle.static.Executor(place)
            ...     data = paddle.static.data(name='X', shape=[None, 1, 28, 28], dtype='float32')
            ...     conv2d = paddle.static.nn.conv2d(input=data, num_filters=6, filter_size=3)
            ...     # 1) Use fp16_guard to control the range of fp16 kernels used.
            ...     with paddle.static.amp.fp16_guard():
            ...         bn = paddle.static.nn.batch_norm(input=conv2d, act="relu")
            ...         pool = F.max_pool2d(bn, kernel_size=2, stride=2)
            ...         hidden = paddle.static.nn.fc(pool, size=10)
            ...         loss = paddle.mean(hidden)
            ...     # 2) Create the optimizer and set `multi_precision` to True.
            ...     # Setting `multi_precision` to True can avoid the poor accuracy
            ...     # or the slow convergence in a way.
            ...     optimizer = paddle.optimizer.Momentum(learning_rate=0.01, multi_precision=True)
            ...     # 3) These ops in `custom_black_list` will keep in the float32 computation type.
            ...     amp_list = paddle.static.amp.CustomOpLists(
            ...         custom_black_list=['pool2d'])
            ...     # 4) The entry of Paddle AMP.
            ...     # Enable pure fp16 training by setting `use_pure_fp16` to True.
            ...     optimizer = paddle.static.amp.decorate(
            ...         optimizer,
            ...         amp_list,
            ...         init_loss_scaling=128.0,
            ...         use_dynamic_loss_scaling=True,
            ...         use_pure_fp16=True)
            ...     # If you don't use the default_startup_program(), you should pass
            ...     # your defined `startup_program` into `minimize`.
            ...     optimizer.minimize(loss)
            ...     exe.run(paddle.static.default_startup_program())
            ...     # 5) Use `amp_init` after FP32 parameters initialization(such as `exe.run(startup_program)`).
            ...     # If you want to perform the testing process, you should pass `test_program` into `amp_init`.
            ...     optimizer.amp_init(place, scope=paddle.static.global_scope())

            >>> if paddle.is_compiled_with_cuda() and len(paddle.static.cuda_places()) > 0:
            ...     run_example_code()
    """
    amp_dtype = "bfloat16" if use_bf16 else "float16"
    if amp_lists is None:
        amp_lists = AutoMixedPrecisionLists(dtype=amp_dtype)

    if use_fp16_guard is None:
        use_fp16_guard = use_pure_fp16

    amp_level = "O2" if use_pure_fp16 else "O1"
    mp_optimizer = OptimizerWithMixedPrecision(
        optimizer,
        amp_lists,
        level=amp_level,
        dtype=amp_dtype,
        init_loss_scaling=init_loss_scaling,
        use_dynamic_loss_scaling=use_dynamic_loss_scaling,
        incr_every_n_steps=incr_every_n_steps,
        decr_every_n_nan_or_inf=decr_every_n_nan_or_inf,
        incr_ratio=incr_ratio,
        decr_ratio=decr_ratio,
        use_amp_guard=use_fp16_guard,
        use_promote=use_promote,
    )

    return mp_optimizer


@overload(key=FunctionType.COMMON)
def decorate(  # noqa: F811
    optimizer,
    amp_lists=None,
    level='O1',
    dtype='float16',
    master_weight=None,
    master_grad=False,
    init_loss_scaling=2**16,
    incr_every_n_steps=2000,
    decr_every_n_nan_or_inf=1,
    incr_ratio=2.0,
    decr_ratio=0.5,
    use_dynamic_loss_scaling=None,
    use_amp_guard=False,
    use_promote=False,
):
    """
    Decorate the given optimizer to adapt to the mixed-precision training.

    Args:
        optimizer(Optimizer): A common Optimizer.
        amp_lists(CustomOpLists, optional): An CustomOpLists object. The default
            white_list and black_list will be used for AMP training when it is
            not set. Default is None.
        level(str, optional): Auto mixed precision level. Accepted values are "O1", "O2" and "OD": At the O1 level, operators in the white list
             will use float16/bfloat16 inputs for calculations, and operators in the black list will use float32 inputs for calculations. At the O2
             level, model's parameters will be casted to float16/bfloat16 by using `decorator`, and operators that have all float16/bfloat16 inputs
             will be converted to float16/bfloat16, and that have any float32 input will be converted to float32. For the OD level, operators in
             default white list will compute in float16/bfloat16, and the others will compute in float32. Default is O1.
        dtype(str, optional): Whether to use 'float16' or 'bfloat16'. Default is 'float16'.
        master_weight(bool, optional): For level='O2', whether to use multi-precision
            during weight updating. If master_weight is None, in O2 level optimizer
            will use multi-precision. Default is None.
        master_grad(bool, optional): For level='O2', whether to use master_grad
            during weight updating. If master_grad is False, in O2 level optimizer
            will not use master grad. Default is False.
        init_loss_scaling(float, optional): The initial loss scaling factor.
            Default is 65536.
        incr_every_n_steps(int, optional): Increases loss scaling every n
            consecutive steps with finite gradients. Default is 2000.
        decr_every_n_nan_or_inf(int, optional): Decreases loss scaling every n
            accumulated steps with nan or inf gradients. Default is 1.
        incr_ratio(float, optional): The multiplier to use when increasing the
            loss scaling. Default is 2.
        decr_ratio(float, optional): The less-than-one-multiplier to use when
            decreasing the loss scaling. Default is 0.5.
        use_dynamic_loss_scaling(bool, None): Whether to use dynamic loss
            scaling. Default is None, which means True for float16, and False
            for bfloat16.

    Returns:
        An optimizer acting like a normal one but with mixed-precision training

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> paddle.enable_static()

            >>> # doctest: +REQUIRES(env:GPU)
            >>> class SimpleConvNet(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.conv = paddle.nn.Conv2D(in_channels=1, out_channels=6, kernel_size=3)
            ...         self.linear = paddle.nn.Linear(in_features=26, out_features=10)
            ...
            ...     def forward(self, x):
            ...         out = self.conv(x)
            ...         out = paddle.nn.functional.relu(out)
            ...         out = self.linear(out)
            ...         out = paddle.nn.functional.softmax(out)
            ...         return out

            >>> main_program = paddle.static.Program()
            >>> startup_program = paddle.static.Program()
            >>> with paddle.utils.unique_name.guard():
            ...     with paddle.static.program_guard(main_program, startup_program):
            ...         model = SimpleConvNet()
            ...         x = paddle.static.data(
            ...             name='input', shape=[None, 1, 28, 28], dtype='float32'
            ...         )
            ...         out = model(x)
            ...         loss = paddle.mean(out)
            ...         optimizer = paddle.optimizer.AdamW()
            ...         optimizer = paddle.static.amp.decorate(optimizer, level="O2", dtype="float16")
            ...         optimizer.minimize(loss)

            >>> if paddle.is_compiled_with_cuda() and len(paddle.static.cuda_places()) > 0:
            ...     place = paddle.CUDAPlace(0)
            ...     exe = paddle.static.Executor(place)
            ...     exe.run(startup_program)
            ...
            ...     # Call `amp_init` after FP32 parameters initialization, such as `exe.run(startup_program)`,
            ...     # to convert FP32 parameters to low precision FP16 / BF16.
            ...     optimizer.amp_init(place, scope=paddle.static.global_scope())

    """
    # check amp_level: O0-O2
    level = level.upper()
    if level not in ['O0', 'OD', 'O1', 'O2']:
        raise ValueError("level should be O0, OD, O1 or O2.")

    amp_dtype = check_amp_dtype(dtype)
    if amp_lists is None or level == 'OD':
        amp_lists = AutoMixedPrecisionLists(dtype=amp_dtype)

    if level == 'OD':
        if amp_lists is not None:
            warnings.warn(
                "If the Amp level is set to OD, the amp list will not be used."
            )
        amp_lists.black_list = amp_lists.all_list - amp_lists.white_list

    if use_dynamic_loss_scaling is None:
        use_dynamic_loss_scaling = dtype == "float16"

    if optimizer is not None:
        # support master_weight
        multi_precision = master_weight is not False
        _set_multi_precision(optimizer, multi_precision)

    mp_optimizer = OptimizerWithMixedPrecision(
        optimizer,
        amp_lists,
        level=level,
        dtype=amp_dtype,
        init_loss_scaling=init_loss_scaling,
        use_dynamic_loss_scaling=use_dynamic_loss_scaling,
        incr_every_n_steps=incr_every_n_steps,
        decr_every_n_nan_or_inf=decr_every_n_nan_or_inf,
        incr_ratio=incr_ratio,
        decr_ratio=decr_ratio,
        use_amp_guard=use_amp_guard,
        use_promote=use_promote,
        use_master_grad=master_grad,
    )

    return mp_optimizer
