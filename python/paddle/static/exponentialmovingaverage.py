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

import numpy as np

import paddle
from paddle.fluid import layers, unique_name
from paddle.fluid.framework import (
    Program,
    default_main_program,
    in_dygraph_mode,
    name_scope,
    program_guard,
)
from paddle.fluid.wrapped_decorator import signature_safe_contextmanager

__all__ = []


class ExponentialMovingAverage:
    r"""

    Compute the moving average of parameters with exponential decay.
    Given a parameter :math:`\\theta`, its exponential moving average (EMA)
    will be

    ..  math::

        \text{EMA}_0 & = 0

        \text{EMA}_t & = \text{decay} * \text{EMA}_{t-1} + (1 - \text{decay}) * \theta_t

    The average results calculated by **update()** method will be saved in
    temporary variables which are created and maintained by the object, and can
    be applied to parameters of current model by calling **apply()** method. And
    the **restore()** method is used to restore the parameters.

    **Bias correction**. All EMAs are initialized to :math:`0` and hence they will be
    zero biased, which can be corrected by divided by a factor
    :math:`(1 - \text{decay}^t)` , i.e., the actual EMAs applied to parameters
    when calling **apply()** method would be

    ..  math::

        \widehat{\text{EMA}}_t = \frac{\text{EMA}_t}{1 - \text{decay}^t}

    **Decay rate scheduling**. A large decay rate very close to 1 would result
    in that the averages move very slowly. And a better strategy is to set a
    relative smaller decay rate in the very beginning. The argument **thres_steps**
    allows users to pass a Variable to schedule the decay rate, in this case,
    the actual decay rate becomes

    ..  math::

        \min(\text{decay}, \frac{1 + \text{thres_steps}}{10 + \text{thres_steps}})

    Usually **thres_steps** can be the global training steps.


    Args:
        decay (float, optional): The exponential decay rate, usually close to 1, such as 0.999, 0.9999, ... . Default 0.999.
        thres_steps (Variable|None, optional): If not `None`, schedule the decay rate. Default None.
        name (str|None, optional): For detailed information, please refer to :ref:`api_guide_Name`. Usually name is no need to set and None by default.


    Examples:

        .. code-block:: python

            import numpy
            import paddle
            import paddle.static as static
            from paddle.static import ExponentialMovingAverage

            paddle.enable_static()

            data = static.data(name='x', shape=[-1, 5], dtype='float32')
            hidden = static.nn.fc(x=data, size=10)
            cost = paddle.mean(hidden)

            test_program = static.default_main_program().clone(for_test=True)
            optimizer = paddle.optimizer.Adam(learning_rate=0.001)
            optimizer.minimize(cost)

            ema = ExponentialMovingAverage(0.999)
            ema.update()

            place = paddle.CPUPlace()
            exe = static.Executor(place)
            exe.run(static.default_startup_program())

            for pass_id in range(3):
                for batch_id in range(6):
                    data = numpy.random.random(size=(10, 5)).astype('float32')
                    exe.run(program=static.default_main_program(),
                    feed={'x': data},
                    fetch_list=[cost.name])

                # usage 1
                with ema.apply(exe):
                    data = numpy.random.random(size=(10, 5)).astype('float32')
                    exe.run(program=test_program,
                        feed={'x': data},
                        fetch_list=[hidden.name])

                # usage 2
                with ema.apply(exe, need_restore=False):
                    data = numpy.random.random(size=(10, 5)).astype('float32')
                    exe.run(program=test_program,
                        feed={'x': data},
                        fetch_list=[hidden.name])
                ema.restore(exe)

    """

    def __init__(self, decay=0.999, thres_steps=None, name=None):
        if in_dygraph_mode():
            raise Exception(
                "In dygraph, don't support ExponentialMovingAverage."
            )
        self._decay = decay
        self._thres_steps = thres_steps
        self._name = name if name is not None else ''
        self._decay_var = self._get_ema_decay()

        self._step_counter_name = "@EMA_STEP_COUNTER@"
        self._params_tmps = []
        for param in default_main_program().global_block().all_parameters():
            if param.do_model_average:
                tmp = param.block.create_var(
                    name=unique_name.generate(
                        ".".join([self._name + param.name, 'ema_tmp'])
                    ),
                    dtype=param.dtype,
                    persistable=False,
                    stop_gradient=True,
                )
                self._params_tmps.append((param, tmp))

        self._ema_vars = {}
        for param, tmp in self._params_tmps:
            with param.block.program._optimized_guard([param, tmp]), name_scope(
                'moving_average'
            ):
                self._ema_vars[param.name] = self._create_ema_vars(param)

        self.apply_program = Program()
        block = self.apply_program.global_block()
        with program_guard(main_program=self.apply_program):
            decay_pow, global_step = self._get_decay_pow(block)
            for param, tmp in self._params_tmps:
                param = block._clone_variable(param)
                tmp = block._clone_variable(tmp)
                ema = block._clone_variable(self._ema_vars[param.name])
                paddle.assign(param, output=tmp)
                # bias correction
                param_val = paddle.static.nn.cond(
                    global_step > 0,
                    lambda: ema / (1.0 - decay_pow),
                    lambda: ema,
                )
                paddle.assign(param_val, output=param)
        self.restore_program = Program()
        block = self.restore_program.global_block()
        with program_guard(main_program=self.restore_program):
            for param, tmp in self._params_tmps:
                tmp = block._clone_variable(tmp)
                param = block._clone_variable(param)
                paddle.assign(tmp, output=param)

    def _get_ema_decay(self):
        with default_main_program()._lr_schedule_guard():
            decay_var = paddle.static.create_global_var(
                shape=[1],
                value=self._decay,
                dtype='float32',
                persistable=True,
                name="scheduled_ema_decay_rate",
            )

            if self._thres_steps is not None:
                decay_t = (self._thres_steps + 1.0) / (self._thres_steps + 10.0)
                decay_val = paddle.static.nn.cond(
                    decay_t < self._decay,
                    lambda: decay_t,
                    lambda: np.array([self._decay], dtype=np.float32),
                )
                paddle.assign(decay_val, decay_var)
        return decay_var

    def _get_decay_pow(self, block):
        global_step = paddle.static.create_global_var(
            name=self._step_counter_name,
            shape=[1],
            value=0,
            dtype='int64',
            persistable=True,
        )
        global_step = paddle.cast(global_step, "float32")
        decay_var = block._clone_variable(self._decay_var)
        decay_pow_acc = paddle.pow(decay_var, global_step)
        return decay_pow_acc, global_step

    def _create_ema_vars(self, param):
        param_ema = paddle.static.create_global_var(
            name=unique_name.generate(self._name + param.name + '_ema'),
            shape=param.shape,
            value=0.0,
            dtype=param.dtype,
            persistable=True,
        )

        return param_ema

    def update(self):
        """
        Update Exponential Moving Average. Should only call this method in
        train program.
        """
        global_step = layers.autoincreased_step_counter(
            counter_name=self._step_counter_name
        )
        param_master_emas = []
        for param, tmp in self._params_tmps:
            with param.block.program._optimized_guard([param, tmp]), name_scope(
                'moving_average'
            ):
                param_ema = self._ema_vars[param.name]
                if param.name + '.master' in self._ema_vars:
                    master_ema = self._ema_vars[param.name + '.master']
                    param_master_emas.append([param_ema, master_ema])
                else:
                    ema_t = param_ema * self._decay_var + param * (
                        1 - self._decay_var
                    )
                    paddle.assign(ema_t, output=param_ema)

        # for fp16 params
        for param_ema, master_ema in param_master_emas:
            default_main_program().global_block().append_op(
                type="cast",
                inputs={"X": master_ema},
                outputs={"Out": param_ema},
                attrs={
                    "in_dtype": master_ema.dtype,
                    "out_dtype": param_ema.dtype,
                },
            )

    @signature_safe_contextmanager
    def apply(self, executor, need_restore=True):
        """
        Apply moving average to parameters for evaluation.

        Args:
            executor (Executor): The Executor to execute applying.
            need_restore (bool, optional): Whether to restore parameters after
                applying. Default True.
        """
        executor.run(self.apply_program)
        try:
            yield
        finally:
            if need_restore:
                self.restore(executor)

    def restore(self, executor):
        """Restore parameters.

        Args:
            executor (Executor): The Executor to execute restoring.
        """
        executor.run(self.restore_program)
