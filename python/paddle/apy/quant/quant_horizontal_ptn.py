# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import abstract_drr
import pir
import quant_horizontal_tpl


@abstract_drr.register_drr_pass("quant_horizontal_fusion", nice=0)
class QuantHorizontalFusion(abstract_drr.DrrPass):
    def source_pattern(self, o, t):
        o.tie_op = o.ap_native_op("ap_op.facade")
        o.tie_op.custom_op_name = pir.a_str("ap_custom_op.tuple_identity")
        o.tie_op([t.input0, t.input1], [t.tie_out0, t.tie_out1])

        o.quant_x_op = o.ap_native_op("ap_op.facade")
        o.quant_x_op.custom_op_name = pir.a_str("ap_custom_op.facade_quant")
        o.quant_x_op([t.tie_out0], [t.output0, t.scale0])

        o.quant_y_op = o.ap_native_op("ap_op.facade")
        o.quant_y_op.custom_op_name = pir.a_str("ap_custom_op.facade_quant")
        o.quant_y_op([t.tie_out1], [t.output1, t.scale1])

    def result_pattern(self, o, t):
        o.fustion_op = o.ap_pattern_fusion_op(self.code_gen)
        o.fustion_op(
            [t.input0, t.input1], [t.output0, t.scale0, t.output1, t.scale1]
        )

    def constraint(self, o, t):
        return True

    def code_gen(self, ctx, o, t):
        template_module = quant_horizontal_tpl.QuantHorizontalTemplate()
        return template_module.compile(
            input0_karg=ctx.in_tensor_data_ptr_kernel_arg_id(t.input0),
            input1_karg=ctx.in_tensor_data_ptr_kernel_arg_id(t.input1),
            output0_karg=ctx.out_tensor_data_ptr_kernel_arg_id(t.output0),
            scale0_karg=ctx.out_tensor_data_ptr_kernel_arg_id(t.scale0),
            output1_karg=ctx.out_tensor_data_ptr_kernel_arg_id(t.output1),
            scale1_karg=ctx.out_tensor_data_ptr_kernel_arg_id(t.scale1),
        )
