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


class InsertReshapeBeforeYieldPass(access_topo_drr.DrrPass):

    def source_pattern(self, o, t):
        o.yield_op = o.ap_native_op("cf.yield")
        o.yield_op([t.output], [])

    def result_pattern(self, o, t):
        t.declare_internal_native_ir_value("reshaped_output")
        o.reshape_op = o.ap_native_op("cinn_op.reshape")
        o.reshape_op.shape = lambda o, t: pir.a_array(
            [pir.a_i32(ap.DataValue.int32("-1"))]
        )
        o.reshape_op([t.output], [t.reshaped_output])
        o.yield_op = o.ap_native_op("cf.yield")
        o.yield_op([t.reshaped_output], [])
