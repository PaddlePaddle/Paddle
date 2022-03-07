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


class ADRunnerState(threading.local):
    def __init__(self) -> None:
        super().__init__()
        self.graph = None
        self.vars = []
        self.var_lookup = {}
        self.dot_lookup = {}
        self.bar_lookup = {}
        self.runners = {
            'graph': MakeGraph(),
            'jvp': JVP(),
            'transpose': Transpose(),
            'lower2prog': LowerToProgram()
        }
        self.runner = None

    def switch_runner(self, kind):
        self.runner = self.runners[kind]


adrunner_state = ADRunnerState()


def switch_runner(kind):
    adrunner_state.switch_runner(kind)


def get_current_runner():
    return adrunner_state.runner


class Runner(object):
    def run_op(self, op, *args, **kwargs):
        raise f'This `process_op` method is missing in {type(self)}.'


class LowerToProgram(Runner):
    pass


class MakeGraph(Runner):
    def run_op(self, op, *args, **kwargs):
        var, node = op(*args, **kwargs)
        current_graph().add_node(node, var)


class JVP(Runner):
    def run_op(self, op, *args, **kwargs):
        jvpmaker = jvpmakers[op]
        jvp_fn = jvpmaker(*args, **kwargs)
        switch_runner('graph')
        out_dot = jvp_fn(*map(var2dot, args))
        switch_runner('jvp')
        return out_dot


class Transpose(Runner):
    def run_op(self, op, *args, **kwargs):
        transposemaker = transposemakers[op]
        transpose_fn = transposemaker(*args, **kwargs)
        switch_runner('graph')
        out_bar = make_var(is_tangent=True)
        in_bars = transpose_fn(out_bar)
        switch_runner('transpose')
        return out_bar, in_bars
