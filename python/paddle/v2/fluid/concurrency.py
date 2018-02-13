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


# TODO: Variables: make_channel
# TODO: Operators: send, close_channel, recv, go, select
from layers.control_flow import BlockGuard
from layer_helper import LayerHelper


class GoGuard(BlockGuard):
    def __init__(self, go_op):
        if not isinstance(go_op, Go):
            raise TypeError("GoGuard takes a go op")
        super(GoGuard, self).__init__(go_op.helper.main_program)
        self.go_op = go_op

    def __enter__(self):
        return super(GoGuard, self).__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
        self.go_op.complete()
        return super(GoGuard, self).__exit__(exc_type, exc_val, exc_tb)


class Go(object):

    def __init__(self, name=None):
        self.helper = LayerHelper("go", name=name)

    def block(self):
        return GoGuard(self)

    def complete(self):
        main_program = self.helper.main_program
        go_block = main_program.current_block()
        parent_block = main_program.block(main_program.current_block()
                                          .parent_idx)

        x_name_list = set()
        out_vars = set()
        for op in go_block.ops:
            # Iterate over all operators, get all the inputs
            # and add as input to the Go operator.
            for iname in op.input_names:
                for in_var_name in op.input(iname):
                    x_name_list.add(in_var_name)

            # Iterate over all operators , get all the outputs
            # add to the output list of Go operator only if
            # they exist in the parent block.
            for oname in op.output_names:
                for out_var_name in op.output(oname):
                    if out_var_name in parent_block.vars:
                        out_vars.add(parent_block.var(out_var_name))

        parent_block.append_op(
            type='go',
            inputs={
                'X': [parent_block.var(x_name) for x_name in x_name_list]
            },
            outputs={'Out': out_vars},
            attrs={'sub_block': go_block})
