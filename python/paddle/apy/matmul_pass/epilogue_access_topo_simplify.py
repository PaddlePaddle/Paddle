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

import access_topo_drr
import ap
import ir_tools
import pir
import topo_drr_pass


class RemoveDataOpPairPass(access_topo_drr.DrrPass):
    def __init__(self, src_data_op_name, dst_data_op_name):
        self.src_data_op_name = pir.a_str(src_data_op_name)
        self.dst_data_op_name = pir.a_str(dst_data_op_name)

    def source_pattern(self, o, t):
        o.src_data_op = o.ap_native_op("pd_op.data")
        o.src_data_op([], [t.input0])
        o.dst_data_op = o.ap_native_op("pd_op.data")
        o.dst_data_op([], [t.input1])
        o.up_spider_op = o.ap_native_op("ap_op.up_spider")
        o.up_spider_op([t.input0, t.input1], [])

    def constraint(self, o, t):
        return [o.src_data_op.name, o.dst_data_op.name] == [
            self.src_data_op_name,
            self.dst_data_op_name,
        ]

    def result_pattern(self, o, t):
        pass


class RemoveDataOp2SumOp2DataOpPass(access_topo_drr.DrrPass):
    def __init__(self, src_data_op_name, dst_data_op_name):
        self.src_data_op_name = pir.a_str(src_data_op_name)
        self.dst_data_op_name = pir.a_str(dst_data_op_name)

    def source_pattern(self, o, t):
        o.src_data_op = o.ap_native_op("pd_op.data")
        o.src_data_op.name = self.src_data_op_name
        o.src_data_op([], [t.input0])
        o.full_int_array_op = o.ap_native_op("pd_op.full_int_array")
        o.full_int_array_op([], [t.axis])
        o.sum_op = o.ap_native_op("pd_op.sum")
        o.sum_op([t.input0, t.axis], [t.sum_out])
        o.dst_data_op = o.ap_native_op("pd_op.data")
        o.dst_data_op.name = self.dst_data_op_name
        o.dst_data_op([], [t.input1])
        o.up_spider_op = o.ap_native_op("ap_op.up_spider")
        o.up_spider_op([t.sum_out, t.input1], [])

    def result_pattern(self, o, t):
        pass


class RemoveElementInputIndexPass(access_topo_drr.DrrPass):
    def __init__(self, src_data_op_name, dst_load_from_global_op_name):
        self.src_data_op_name = pir.a_str(src_data_op_name)
        self.dst_load_from_global_op_name = pir.a_str(
            dst_load_from_global_op_name
        )

    def source_pattern(self, o, t):
        o.src_data_op = o.ap_native_op("pd_op.data")
        o.src_data_op.name = self.src_data_op_name
        o.src_data_op([], [t.src_input])

        o.dst_load_from_global_op = o.ap_native_op("ap_op.load_from_global")
        o.dst_load_from_global_op.index_func_unique_id = (
            self.dst_load_from_global_op_name
        )
        o.dst_load_from_global_op(
            [t.dst_input], [t.dst_load_from_global_output]
        )
        o.up_spider_op = o.ap_native_op("ap_op.up_spider")
        o.up_spider_op([t.src_input, t.dst_load_from_global_output], [])

    def result_pattern(self, o, t):
        pass


class RemoveBroadcastInputIndexPass(access_topo_drr.DrrPass):
    def __init__(self, src_data_op_name, dst_load_from_global_op_name):
        self.src_data_op_name = pir.a_str(src_data_op_name)
        self.dst_load_from_global_op_name = pir.a_str(
            dst_load_from_global_op_name
        )

    def source_pattern(self, o, t):
        o.src_data_op = o.ap_native_op("pd_op.data")
        o.src_data_op.name = self.src_data_op_name
        o.src_data_op([], [t.input0])
        o.full_int_array_op = o.ap_native_op("pd_op.full_int_array")
        o.full_int_array_op([], [t.axis])
        o.sum_op = o.ap_native_op("pd_op.sum")
        o.sum_op([t.input0, t.axis], [t.sum_out])
        o.dst_load_from_global_op = o.ap_native_op("ap_op.load_from_global")
        o.dst_load_from_global_op.index_func_unique_id = (
            self.dst_load_from_global_op_name
        )
        o.dst_load_from_global_op(
            [t.dst_input], [t.dst_load_from_global_output]
        )
        o.up_spider_op = o.ap_native_op("ap_op.up_spider")
        o.up_spider_op([t.sum_out, t.dst_load_from_global_output], [])

    def result_pattern(self, o, t):
        pass


class RemoveOutputIndexPass(access_topo_drr.DrrPass):
    def __init__(self, src_data_op_name, dst_store_to_global_op_name):
        self.src_data_op_name = pir.a_str(src_data_op_name)
        self.dst_store_to_global_op_name = pir.a_str(
            dst_store_to_global_op_name
        )

    def source_pattern(self, o, t):
        o.src_data_op = o.ap_native_op("pd_op.data")
        o.src_data_op.name = self.src_data_op_name
        o.src_data_op([], [t.src_input])
        o.down_spider_op = o.ap_native_op("ap_op.down_spider")
        o.down_spider_op([t.src_input], [t.dst_output_val])
        o.dst_store_to_global_op = o.ap_native_op("ap_op.store_to_global")
        o.dst_store_to_global_op.index_func_unique_id = (
            self.dst_store_to_global_op_name
        )
        o.dst_store_to_global_op([t.dst_output, t.dst_output_val], [])

    def result_pattern(self, o, t):
        pass


def simplify_epilogue_program(
    program, anchor_data_op_name, number_of_inputs, number_of_outputs
):
    print("before-umprime: ", program)
    # umprime passes
    pass_manager = ir_tools.create_pass_manager()
    pass_manager.add_pass(ir_tools.create_access_topo_drr_pass("umprime"))
    pass_manager.add_pass(ir_tools.create_dce_pass())
    pass_manager.run(program)
    print("before-access_topo_pass", program)
    init_pass_manager = ir_tools.create_pass_manager()
    init_down_spider = topo_drr_pass.InitDownSpiderAccessTopoPass(
        anchor_data_op_name
    )
    init_pass_manager.add_pass(
        ir_tools.create_access_topo_drr_one_step_pass(init_down_spider)
    )
    outputs_name_list = ap.map(lambda i: f"output{i}", range(number_of_outputs))
    inputs_name_list = (
        ap.map(lambda i: f"input{i + 2}", range(number_of_inputs - 2))
        if number_of_inputs > 2
        else []
    )
    print('inputs_name_list: ', ', '.join(inputs_name_list))
    init_fake_data_for_yield_input = (
        topo_drr_pass.FakeDataForYieldAccessTopoPass(outputs_name_list)
    )
    init_pass_manager.add_pass(
        ir_tools.create_access_topo_drr_one_step_pass(
            init_fake_data_for_yield_input
        )
    )
    init_pass_manager.run(program)
    print("after-init-access_topo_pass", program)
    pass_manager = ir_tools.create_pass_manager()
    pass_manager.add_pass(ir_tools.create_access_topo_drr_pass("default"))
    pass_manager.add_pass(ir_tools.create_dce_pass())
    pass_manager.run(program)
    print("after-apply-access_topo_pass", program)
    pass_manager = ir_tools.create_pass_manager()
    ap.map(
        lambda dst_name: pass_manager.add_pass(
            ir_tools.create_access_topo_drr_one_step_pass(
                RemoveDataOpPairPass(
                    src_data_op_name=anchor_data_op_name,
                    dst_data_op_name=dst_name,
                )
            )
        ),
        inputs_name_list,
    )
    ap.map(
        lambda dst_name: pass_manager.add_pass(
            ir_tools.create_access_topo_drr_one_step_pass(
                RemoveDataOp2SumOp2DataOpPass(
                    src_data_op_name=anchor_data_op_name,
                    dst_data_op_name=dst_name,
                )
            )
        ),
        inputs_name_list,
    )

    ap.map(
        lambda dst_name: pass_manager.add_pass(
            ir_tools.create_access_topo_drr_one_step_pass(
                RemoveDataOpPairPass(
                    src_data_op_name=anchor_data_op_name,
                    dst_data_op_name=dst_name,
                )
            )
        ),
        outputs_name_list,
    )
    pass_manager.add_pass(ir_tools.create_dce_pass())
    pass_manager.run(program)
    print("after-remove-input-output-access_topo_pass", program)
    return program
