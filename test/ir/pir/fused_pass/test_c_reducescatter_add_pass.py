# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import paddle

paddle.pir.register_paddle_dialect()

paddle.enable_static()


def remove_shadow_output_and_fuse_c_reducescatter_assign_add(program):
    block = program.global_block()
    ops = block.ops
    need_remove_ops = []
    op_num = len(ops)
    for idx in range(op_num - 2):
        if (
            ops[idx].name() == 'pd_op.c_reducescatter'
            and ops[idx + 1].name() == "pd_op.assign"
            and ops[idx + 2].name() == "pd_op.add"
        ):
            assign_output = ops[idx + 1].result(0)
            for op in ops:
                if (
                    op.name() == 'builtin.shadow_output'
                    and op in assign_output.all_used_ops()
                ):
                    need_remove_ops.append(op)

    for op in need_remove_ops:
        block.remove_op(op)
    pm = paddle.pir.PassManager()
    pm.add_pass('fuse_c_reducescatter_add_pass', {})
    pm.run(program)
    return program


class TestFusedCReducescatterAssignAddPass(unittest.TestCase):
    def test_c_reducescatter_assign_add_pass(self):
        serialized_pir_program = '''
        {
            (%0) = "pd_op.data" () {dtype:(pd_op.DataType)float32,name:"Bias",persistable:[false],place:(pd_op.Place)Place(undefined:0),shape:(pd_op.IntArray)[2,2],stop_gradient:[true]} : () -> builtin.tensor<2x2xf32>
            (%1) = "pd_op.data" () {dtype:(pd_op.DataType)float32,name:"X",persistable:[false],place:(pd_op.Place)Place(undefined:0),shape:(pd_op.IntArray)[2,2],stop_gradient:[true]} : () -> builtin.tensor<2x2xf32>
            (%2) = "pd_op.c_reducescatter" (%1) {nranks:(Int32)1,persistable:[false],ring_id:(Int32)0,stop_gradient:[true],use_calc_stream:false} : (builtin.tensor<2x2xf32>) -> builtin.tensor<2x2xf32>
            (%3) = "pd_op.assign" (%2) {persistable:[false],stop_gradient:[true]} : (builtin.tensor<2x2xf32>) -> builtin.tensor<2x2xf32>
            (%4) = "pd_op.add" (%0, %3) {persistable:[false],stop_gradient:[true]} : (builtin.tensor<2x2xf32>, builtin.tensor<2x2xf32>) -> builtin.tensor<2x2xf32>
             () = "builtin.shadow_output" (%3) {output_name:"input0"} : (builtin.tensor<2x2xf32>) ->
         }
        '''
        pir_program = paddle.pir.parse_program(serialized_pir_program)

        remove_shadow_output_and_fuse_c_reducescatter_assign_add(pir_program)

        assert "pd_op.c_reducescatter\"" not in str(pir_program)
        assert "pd_op.c_reducescatter_add" in str(pir_program)


if __name__ == "__main__":
    unittest.main()
