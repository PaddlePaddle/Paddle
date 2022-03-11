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

import threading
import inspect
from paddle.fluid.framework import Operator
from paddle.fluid.backward import _create_op_desc_


class ADRunnerState(threading.local):
    def __init__(self) -> None:
        super().__init__()
        self.tangent_set = set()
        self.dot_lookup = {}
        self.bar_lookup = {}
        self.runners = {
            'primitive': MakePrimitive(),
            'linearize': Linearize(),
            'transpose': Transpose(),
            'edit': EditProgram(),
        }
        self.runner = None

    def switch_runner(self, kind):
        self.runner = self.runners[kind]

    def clear_state(self):
        self.tangent_set = set()
        self.dot_lookup = {}
        self.bar_lookup = {}


adrunner_state = ADRunnerState()


def switch_runner(kind):
    adrunner_state.switch_runner(kind)


def get_current_runner():
    return adrunner_state.runner


class Runner(object):
    def run_op(self, op, *args, **kwargs):
        raise f'This `process_op` method is missing in {type(self)}.'


class MakePrimitive(Runner):
    def run_op(self, op, *args, **kwargs):
        primitivemaker = primitivemakers[op]
        primitive_fn = primitivemaker(*args, **kwargs)
        switch_runner('edit')
        nins = len(inspect.getargspec(primitive_fn).args)
        ins = [args[i].name for i in range(nins)]
        primitive_fn(*ins)
        switch_runner('primitive')
        return


class Linearize(Runner):
    def run_op(self, op, *args, **kwargs):
        linearizemaker = linearizemakers[op]
        linearize_fn = linearizemaker(*args, **kwargs)
        switch_runner('edit')
        nins = len(inspect.getargspec(primitive_fn).args)
        ins = [var.name for var in map(var2dot, args[0:nins])]
        out_dot = list(linearize_fn(ins))
        switch_runner('linearize')
        return out_dot


class Transpose(Runner):
    def run_op(self, op, *args, **kwargs):
        transposemaker = transposemakers[op]
        transpose_fn = transposemaker(*args, **kwargs)
        switch_runner('edit')
        out_bar = make_var(is_tangent=True)
        in_bars = transpose_fn(out_bar)
        switch_runner('transpose')
        return out_bar, in_bars


class Edit(Runner):
    def run_op(self, op, *args, **kwargs):
        nins = op.nins if op.nins is not None else len(args - op.nouts)
        op_desc = _create_op_desc(op.optype,
                                  to_in_dict(args[0:nins]),
                                  to_out_dict(args[nins + 1:len(args)]),
                                  dict(**kwargs))
        block = default_main_program().current_block()
        new_op_desc = block.append_op()
        new_op_desc.copy_from(op_desc)
        new_op = Operator(block=block, desc=new_op_desc)
        block.ops.append(new_op)
        return args[nins + 1:len(args)]
