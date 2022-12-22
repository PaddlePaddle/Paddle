#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np

import paddle
from . import layers
from .framework import Program, Variable, program_guard
from . import unique_name
from .layer_helper import LayerHelper
from .initializer import Constant


def _clone_var_(block, var):
    assert isinstance(var, Variable)
    return block.create_var(
        name=var.name,
        shape=var.shape,
        dtype=var.dtype,
        type=var.type,
        lod_level=var.lod_level,
        persistable=True,
    )


class Evaluator:
    """
    Warning: better to use the fluid.metrics.* things, more
    flexible support via pure Python and Operator, and decoupled
    with executor. Short doc are intended to urge new user
    start from Metrics.

    Base Class for all evaluators.

    Args:
        name(str): The name of evaluator. such as, "accuracy". Used for generate
            temporary variable name.
        main_program(Program, optional): The evaluator should be added to this
            main_program. Default default_main_program()
        startup_program(Program, optional):The parameter should be added to this
            startup_program. Default default_startup_program()

    Attributes:
        states(list): The list of state variables. states will be reset to zero
            when `reset` is invoked.
        metrics(list): The list of metrics variables. They will be calculate
            every mini-batch
    """

    def __init__(self, name, **kwargs):
        warnings.warn(
            "The %s is deprecated, because maintain a modified program inside evaluator cause bug easily, please use fluid.metrics.%s instead."
            % (self.__class__.__name__, self.__class__.__name__),
            Warning,
        )
        self.states = []
        self.metrics = []
        self.helper = LayerHelper(name, **kwargs)

    def reset(self, executor, reset_program=None):
        """
        reset metric states at the begin of each pass/user specified batch

        Args:
            executor(Executor|ParallelExecutor): a executor for executing the reset_program
            reset_program(Program): a single Program for reset process
        """
        if reset_program is None:
            reset_program = Program()

        with program_guard(main_program=reset_program):
            for var in self.states:
                assert isinstance(var, Variable)
                g_var = _clone_var_(reset_program.current_block(), var)
                layers.fill_constant(
                    shape=g_var.shape, value=0.0, dtype=g_var.dtype, out=g_var
                )

        executor.run(reset_program)

    def eval(self, executor, eval_program=None):
        """
        Evaluate the statistics merged by multiple mini-batches.
        Args:
            executor(Executor|ParallelExecutor): a executor for executing the eval_program
            eval_program(Program): a single Program for eval process
        """
        raise NotImplementedError()

    def _create_state(self, suffix, dtype, shape):
        """
        Create state variable.

        Args:
            suffix(str): the state suffix.
            dtype(str|core.VarDesc.VarType): the state data type
            shape(tuple|list): the shape of state

        Returns: State variable

        """
        state = self.helper.create_variable(
            name="_".join([unique_name.generate(self.helper.name), suffix]),
            persistable=True,
            dtype=dtype,
            shape=shape,
        )
        self.states.append(state)
        return state
