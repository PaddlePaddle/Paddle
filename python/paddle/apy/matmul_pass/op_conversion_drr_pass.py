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


@access_topo_drr.register_drr_pass("pd_op_cast", tag="default")
class PdOpCastAccessTopoPass(access_topo_drr.DrrPass):

    def source_pattern(self, o, t):
        o.cast_op = o.ap_native_op("pd_op.cast")
        o.cast_op([t.input], [t.output])

    def result_pattern(self, o, t):
        o.result_op = o.ap_native_op("pd_op.relu")
        o.result_op([t.input], [t.output])


@access_topo_drr.register_drr_pass("pd_op_tanh", tag="default")
class PdOpTanhAccessTopoPass(access_topo_drr.DrrPass):

    def source_pattern(self, o, t):
        o.tanh_op = o.ap_native_op("pd_op.tanh")
        o.tanh_op([t.input], [t.output])

    def result_pattern(self, o, t):
        o.result_op = o.ap_native_op("pd_op.relu")
        o.result_op([t.input], [t.output])


@access_topo_drr.register_drr_pass("pd_op_floor", tag="default")
class PdOpFloorAccessTopoPass(access_topo_drr.DrrPass):

    def source_pattern(self, o, t):
        o.floor_op = o.ap_native_op("pd_op.floor")
        o.floor_op([t.input], [t.output])

    def result_pattern(self, o, t):
        o.result_op = o.ap_native_op("pd_op.relu")
        o.result_op([t.input], [t.output])


@access_topo_drr.register_drr_pass("pd_op_erf", tag="default")
class PdOpErfAccessTopoPass(access_topo_drr.DrrPass):

    def source_pattern(self, o, t):
        o.erf_op = o.ap_native_op("pd_op.erf")
        o.erf_op([t.input], [t.output])

    def result_pattern(self, o, t):
        o.result_op = o.ap_native_op("pd_op.relu")
        o.result_op([t.input], [t.output])


@access_topo_drr.register_drr_pass("pd_op_elementwise_pow", tag="default")
class PdOpElementwisePowAccessTopoPass(access_topo_drr.DrrPass):

    def source_pattern(self, o, t):
        o.source_op = o.ap_native_op("pd_op.elementwise_pow")
        o.source_op([t.input0, t.input1], [t.output])

    def result_pattern(self, o, t):
        o.result_op = o.ap_native_op("pd_op.add")
        o.result_op([t.input0, t.input1], [t.output])


@access_topo_drr.register_drr_pass("pd_op_exp", tag="default")
class PdOpExpAccessTopoPass(access_topo_drr.DrrPass):

    def source_pattern(self, o, t):
        o.exp_op = o.ap_native_op("pd_op.exp")
        o.exp_op([t.input], [t.output])

    def result_pattern(self, o, t):
        o.result_op = o.ap_native_op("pd_op.relu")
        o.result_op([t.input], [t.output])


@access_topo_drr.register_drr_pass("cinn_op_scale", tag="default")
class CinnOpScaleAccessTopoPass(access_topo_drr.DrrPass):

    def source_pattern(self, o, t):
        o.scale_op = o.ap_native_op("cinn_op.scale")
        o.scale_op([t.input], [t.output])

    def result_pattern(self, o, t):
        o.result_op = o.ap_native_op("pd_op.relu")
        o.result_op([t.input], [t.output])


@access_topo_drr.register_drr_pass("pd_op_sin", tag="default")
class PdOpSinAccessTopoPass(access_topo_drr.DrrPass):

    def source_pattern(self, o, t):
        o.sin_op = o.ap_native_op("pd_op.sin")
        o.sin_op([t.input], [t.output])

    def result_pattern(self, o, t):
        o.result_op = o.ap_native_op("pd_op.relu")
        o.result_op([t.input], [t.output])


@access_topo_drr.register_drr_pass("cinn_op_yield_store", tag="default")
class CinnOpYieldStoreAccessTopoPass(access_topo_drr.DrrPass):

    def source_pattern(self, o, t):
        o.yield_op = o.ap_native_op("cinn_op.yield_store")
        o.yield_op([t.input], [t.output])

    def result_pattern(self, o, t):
        o.result_op = o.ap_native_op("pd_op.relu")
        o.result_op([t.input], [t.output])


@access_topo_drr.register_drr_pass("pd_op_subtract", tag="default")
class PdOpSubtractAccessTopoPass(access_topo_drr.DrrPass):

    def source_pattern(self, o, t):
        o.source_op = o.ap_native_op("pd_op.subtract")
        o.source_op([t.input0, t.input1], [t.output])

    def result_pattern(self, o, t):
        o.result_op = o.ap_native_op("pd_op.add")
        o.result_op([t.input0, t.input1], [t.output])


@access_topo_drr.register_drr_pass("pd_op_divide", tag="default")
class PdOpDivideAccessTopoPass(access_topo_drr.DrrPass):

    def source_pattern(self, o, t):
        o.source_op = o.ap_native_op("pd_op.divide")
        o.source_op([t.input0, t.input1], [t.output])

    def result_pattern(self, o, t):
        o.result_op = o.ap_native_op("pd_op.add")
        o.result_op([t.input0, t.input1], [t.output])


@access_topo_drr.register_drr_pass("pd_op_multiply", tag="default")
class PdOpMultiplyAccessTopoPass(access_topo_drr.DrrPass):

    def source_pattern(self, o, t):
        o.source_op = o.ap_native_op("pd_op.multiply")
        o.source_op([t.input0, t.input1], [t.output])

    def result_pattern(self, o, t):
        o.result_op = o.ap_native_op("pd_op.add")
        o.result_op([t.input0, t.input1], [t.output])


@access_topo_drr.register_drr_pass("pd_op_maximum", tag="default")
class PdOpMaximumAccessTopoPass(access_topo_drr.DrrPass):

    def source_pattern(self, o, t):
        o.source_op = o.ap_native_op("pd_op.maximum")
        o.source_op([t.input0, t.input1], [t.output])

    def result_pattern(self, o, t):
        o.result_op = o.ap_native_op("pd_op.add")
        o.result_op([t.input0, t.input1], [t.output])


@access_topo_drr.register_drr_pass("pd_op_left_full_add", tag="default")
class PdOpLeftFullAddAccessTopoPass(access_topo_drr.DrrPass):

    def source_pattern(self, o, t):
        o.full_op = o.ap_native_op("pd_op.full")
        o.full_op([], [t.intermediate])
        o.source_op = o.ap_native_op("pd_op.add")
        o.source_op([t.intermediate, t.input], [t.output])

    def result_pattern(self, o, t):
        o.result_op = o.ap_native_op("pd_op.relu")
        o.result_op([t.input], [t.output])


@access_topo_drr.register_drr_pass("pd_op_right_full_add", tag="default")
class PdOpRightFullAddAccessTopoPass(access_topo_drr.DrrPass):

    def source_pattern(self, o, t):
        o.full_op = o.ap_native_op("pd_op.full")
        o.full_op([], [t.intermediate])
        o.source_op = o.ap_native_op("pd_op.add")
        o.source_op([t.input, t.intermediate], [t.output])

    def result_pattern(self, o, t):
        o.result_op = o.ap_native_op("pd_op.relu")
        o.result_op([t.input], [t.output])


@access_topo_drr.register_drr_pass(
    "full_generate_shape_expand_left_add", tag="default"
)
class FullGenerateShapeExpandLeftAddAccessTopoPass(access_topo_drr.DrrPass):

    def source_pattern(self, o, t):
        o.full = o.ap_native_op("pd_op.full")
        o.full([], [t.intermediate0])
        o.generate_shape = o.ap_native_op("cinn_op.generate_shape")
        o.generate_shape([t.input0], [t.intermediate1])
        o.expand = o.ap_native_op("pd_op.expand")
        o.expand([t.intermediate0, t.intermediate1], [t.expanded_input])
        o.add = o.ap_native_op("pd_op.add")
        o.add([t.expanded_input, t.input0], [t.output])

    def result_pattern(self, o, t):
        o.relu = o.ap_native_op("pd_op.relu")
        o.relu([t.input0], [t.output])


@access_topo_drr.register_drr_pass(
    "full_generate_shape_expand_right_add", tag="default"
)
class FullGenerateShapeExpandRightAddAccessTopoPass(access_topo_drr.DrrPass):

    def source_pattern(self, o, t):
        o.full = o.ap_native_op("pd_op.full")
        o.full([], [t.intermediate0])
        o.generate_shape = o.ap_native_op("cinn_op.generate_shape")
        o.generate_shape([t.input0], [t.intermediate1])
        o.expand = o.ap_native_op("pd_op.expand")
        o.expand([t.intermediate0, t.intermediate1], [t.expanded_input])
        o.add = o.ap_native_op("pd_op.add")
        o.add([t.input0, t.expanded_input], [t.output])

    def result_pattern(self, o, t):
        o.relu = o.ap_native_op("pd_op.relu")
        o.relu([t.input0], [t.output])
