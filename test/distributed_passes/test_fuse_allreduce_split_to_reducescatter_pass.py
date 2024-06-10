# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

program_txt = '''
{
    (%2) = "builtin.parameter" () {parameter_name:"linear_0.w_0",persistable:[true],stop_gradient:[false]} : () -> builtin.tensor<8192x28672xbf16>
    (%38) = "pd_op.data" () {dtype:(pd_op.DataType)bfloat16,name:"linear_0.tmp_0",persistable:[false],place:(pd_op.Place)Place(gpu:0),shape:(pd_op.IntArray)[4096,1,28672],stop_gradient:[false]} : () -> builtin.tensor<4096x1x28672xbf16>
    (%48) = "pd_op.data" () {dtype:(pd_op.DataType)bfloat16,name:"input",persistable:[false],place:(pd_op.Place)Place(gpu:0),shape:(pd_op.IntArray)[4096,1,28672],stop_gradient:[false]} : () -> builtin.tensor<4096x1x28672xbf16>
    (%50) = "pd_op.matmul" (%48, %2) {persistable:[false],stop_gradient:[false],transpose_x:false,transpose_y:true} : (builtin.tensor<4096x1x28672xbf16>, builtin.tensor<8192x28672xbf16>) -> builtin.tensor<4096x1x8192xbf16>
    (%57) = "pd_op.c_allreduce_sum_" (%50) {event_to_record:"event_7989",events_to_wait:[],execution_stream:"auto_parallel_mp",force_record_event:false,persistable:[false],ring_id:(Int32)36,stop_gradient:[false],use_calc_stream:true,use_model_parallel:true} : (builtin.tensor<4096x1x8192xbf16>) -> builtin.tensor<4096x1x8192xbf16>
    (%63) = "pd_op.assign" (%57) {persistable:[false],stop_gradient:[false]} : (builtin.tensor<4096x1x8192xbf16>) -> builtin.tensor<4096x1x8192xbf16>
    (%64) = "pd_op.full" () {dtype:(pd_op.DataType)int32,place:(pd_op.Place)Place(cpu),shape:(pd_op.IntArray)[1],stop_gradient:[true],value:(Float)0} : () -> builtin.tensor<1xi32>
    (%65) = "pd_op.split_with_num" (%63, %64) {num:(Int32)2,persistable:[false],stop_gradient:[false]} : (builtin.tensor<4096x1x8192xbf16>, builtin.tensor<1xi32>) -> vec[builtin.tensor<2048x1x8192xbf16>,builtin.tensor<2048x1x8192xbf16>]
    (%66) = "builtin.slice" (%65) {index:(Int32)0,persistable:[false],stop_gradient:[false]} : (vec[builtin.tensor<2048x1x8192xbf16>,builtin.tensor<2048x1x8192xbf16>]) -> builtin.tensor<2048x1x8192xbf16>
    (%67) = "pd_op.assign" (%66) {persistable:[false],stop_gradient:[false]} : (builtin.tensor<2048x1x8192xbf16>) -> builtin.tensor<2048x1x8192xbf16>
}'''


class TestPass(unittest.TestCase):
    def test_if_with_single_output(self):
        paddle.pir.register_paddle_dialect()
        ir_program = paddle.pir.parse_program(program_txt)
        pm = paddle.pir.PassManager()
        pm.add_pass('fuse_allreduce_split_to_reducescatter_pass', {})
        pm.run(ir_program)
        self.assertEqual(ir_program.global_block().num_ops(), 6)
        self.assertEqual(
            ir_program.global_block().ops[-2].name(), "pd_op.c_reducescatter"
        )


if __name__ == "__main__":
    unittest.main()
