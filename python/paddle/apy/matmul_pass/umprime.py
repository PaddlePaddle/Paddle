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
import pir


@access_topo_drr.register_drr_pass("pd_op_static_relu", tag="umprime")
class PdOpReluAccessTopoPass(access_topo_drr.DrrPass):
    def __init__(self):
        self.zero = pir.a_f64(ap.DataValue.float64("0"))

    def source_pattern(self, o, t):
        o.full_op = o.ap_native_op("pd_op.full")
        o.full_op([], [t.intermediate])
        o.maximum_op = o.ap_native_op("pd_op.maximum")
        o.maximum_op([t.input, t.intermediate], [t.output])

    def constraint(self, o, t):
        return o.full_op.value == self.zero

    def result_pattern(self, o, t):
        o.result_op = o.ap_native_op("pd_op.relu")
        o.result_op([t.input], [t.output])


@access_topo_drr.register_drr_pass("pd_op_dynamic_relu", tag="umprime")
class PdOpDynReluAccessTopoPass(access_topo_drr.DrrPass):
    def __init__(self):
        self.zero = pir.a_f64(ap.DataValue.float64("0"))

    def source_pattern(self, o, t):
        o.full_op = o.ap_native_op("pd_op.full")
        o.full_op([], [t.intermediate0])
        o.generate_shape_op = o.ap_native_op("cinn_op.generate_shape")
        o.generate_shape_op([t.input0], [t.intermediate1])
        o.expand_op = o.ap_native_op("pd_op.expand")
        o.expand_op([t.intermediate0, t.intermediate1], [t.intermediate2])
        o.maximum_op = o.ap_native_op("pd_op.maximum")
        o.maximum_op([t.input1, t.intermediate2], [t.output])

    def constraint(self, o, t):
        return o.full_op.value == self.zero

    def result_pattern(self, o, t):
        o.result_op = o.ap_native_op("pd_op.relu")
        o.result_op([t.input1], [t.output])
