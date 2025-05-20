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


class FakeDataForYieldAccessTopoPass(access_topo_drr.DrrPass):

    def __init__(self, fake_data_names):
        self.num_outputs = len(fake_data_names)
        self.fake_data_names = fake_data_names
        self.undefined_place = pir.a_place(pir.UndefinedPlace())

    def source_pattern(self, o, t):
        o.yield_op = o.ap_native_op("cf.yield")

        def get_yield_input(i):
            return getattr(t, f"output{i}")

        o.yield_op(ap.map(get_yield_input, range(self.num_outputs)), [])

    def result_pattern(self, o, t):
        self.result_pattern_data_op(o, t)
        self.result_pattern_up_spider(o, t)

    def result_pattern_data_op(self, o, t):
        ap.map(
            lambda i: self.data_op_for_output(o, t, i), range(self.num_outputs)
        )

    def data_op_for_output(self, o, t, i):
        t.declare_internal_native_ir_value(f"data_out{i}")
        data_op_unique_name = f"data_op_for_output{i}"
        setattr(o, data_op_unique_name, o.ap_native_op("pd_op.data"))
        data_op = getattr(o, data_op_unique_name)
        data_op.name = lambda o, t: self.get_name(o, t, i)
        data_op.shape = lambda o, t: self.get_shape(o, t, i)
        data_op.dtype = lambda o, t: self.get_dtype(o, t, i)
        data_op.place = lambda o, t: self.get_place(o, t, i)
        data_op([], [getattr(t, f"data_out{i}")])

    def get_name(self, o, t, i):
        return pir.a_str(self.fake_data_names[i])

    def get_shape(self, o, t, i):
        ir_tensor = getattr(t, f"output{i}")

        def GetDims(dtype, dims, data_layout):
            return dims

        return pir.a_intarray(ir_tensor.type.match(t_dtensor=GetDims))

    def get_dtype(self, o, t, i):
        ir_tensor = getattr(t, f"output{i}")

        def GetDtype(dtype, dims, data_layout):
            return dtype

        return pir.a_dtype(ir_tensor.type.match(t_dtensor=GetDtype))

    def get_place(self, o, t, i):
        return self.undefined_place

    def result_pattern_up_spider(self, o, t):
        ap.map(
            lambda i: self.up_spider_for_output(o, t, i),
            range(self.num_outputs),
        )

    def up_spider_for_output(self, o, t, i):
        t.declare_internal_native_ir_value(f"add_out{i}")
        up_spider_op_name = f"up_spider_op{i}"
        setattr(o, up_spider_op_name, o.ap_native_op("ap_op.up_spider"))
        getattr(o, up_spider_op_name)(
            [getattr(t, f"output{i}"), getattr(t, f"data_out{i}")], []
        )


class FakeDataStoreToGlobalForYieldAccessTopoPass(access_topo_drr.DrrPass):

    def __init__(self, fake_data_names):
        self.num_outputs = len(fake_data_names)
        self.fake_data_names = fake_data_names
        self.undefined_place = pir.a_place(pir.UndefinedPlace())

    def source_pattern(self, o, t):
        o.yield_op = o.ap_native_op("cf.yield")

        def get_yield_input(i):
            return getattr(t, f"output{i}")

        o.yield_op(ap.map(get_yield_input, range(self.num_outputs)), [])

    def result_pattern(self, o, t):
        self.result_pattern_data_op(o, t)
        self.result_pattern_store_to_global_op(o, t)

    def result_pattern_data_op(self, o, t):
        ap.map(
            lambda i: self.data_op_for_output(o, t, i), range(self.num_outputs)
        )

    def data_op_for_output(self, o, t, i):
        t.declare_internal_native_ir_value(f"data_out{i}")
        data_op_unique_name = f"data_op_for_output{i}"
        setattr(o, data_op_unique_name, o.ap_native_op("pd_op.data"))
        data_op = getattr(o, data_op_unique_name)
        data_op.name = lambda o, t: self.get_name(o, t, i)
        data_op.shape = lambda o, t: self.get_shape(o, t, i)
        data_op.dtype = lambda o, t: self.get_dtype(o, t, i)
        data_op.place = lambda o, t: self.get_place(o, t, i)
        data_op([], [getattr(t, f"data_out{i}")])

    def get_name(self, o, t, i):
        return pir.a_str(self.fake_data_names[i])

    def get_shape(self, o, t, i):
        ir_tensor = getattr(t, f"output{i}")

        def GetDims(dtype, dims, data_layout):
            return dims

        return pir.a_intarray(ir_tensor.type.match(t_dtensor=GetDims))

    def get_dtype(self, o, t, i):
        ir_tensor = getattr(t, f"output{i}")

        def GetDtype(dtype, dims, data_layout):
            return dtype

        return pir.a_dtype(ir_tensor.type.match(t_dtensor=GetDtype))

    def get_place(self, o, t, i):
        return self.undefined_place

    def result_pattern_store_to_global_op(self, o, t):
        ap.map(
            lambda i: self.store_to_global_op_for_output(o, t, i),
            range(self.num_outputs),
        )

    def store_to_global_op_for_output(self, o, t, i):
        store_to_global_op_name = f"store_to_global_op{i}"
        setattr(
            o, store_to_global_op_name, o.ap_native_op("ap_op.store_to_global")
        )
        store_to_global_op = getattr(o, store_to_global_op_name)
        store_to_global_op.index_func_unique_id = lambda o, t: pir.a_str(
            self.fake_data_names[i]
        )
        store_to_global_op(
            [getattr(t, f"data_out{i}"), getattr(t, f"output{i}")], []
        )


class ConvertUpSpiderStoreDataOpToYieldOpPass(access_topo_drr.DrrPass):

    def source_pattern(self, o, t):
        o.data_op = o.ap_native_op("pd_op.data")
        o.data_op([], [t.input1])
        o.load_from_global_op = o.ap_native_op("ap_op.load_from_global")
        o.load_from_global_op([t.input1], [t.tmp1])
        o.up_spider_op = o.ap_native_op("ap_op.up_spider")
        o.up_spider_op([t.input0, t.tmp1], [])

    def result_pattern(self, o, t):
        o.yield_op = o.ap_native_op("cf.yield")
        o.yield_op([t.input0], [])


class ConvertDownSpiderStoreDataOpToYieldOpPass(access_topo_drr.DrrPass):

    def source_pattern(self, o, t):
        o.data_mm_op = o.ap_native_op("pd_op.data")
        o.data_mm_op([], [t.input1])
        o.down_spider_op = o.ap_native_op("ap_op.down_spider")
        o.down_spider_op([t.input1], [t.tmp1])
        o.store_to_global = o.ap_native_op("ap_op.store_to_global")
        o.store_to_global([t.input0, t.tmp1], [])

    def result_pattern(self, o, t):
        o.yield_op = o.ap_native_op("cf.yield")
        o.yield_op([t.input0], [])


class InitDownSpiderAccessTopoPass(access_topo_drr.DrrPass):

    def __init__(self, data_input_name):
        self.data_input_name_attr = pir.a_str(data_input_name)

    def source_pattern(self, o, t):
        o.data_op = o.ap_native_op("pd_op.data")
        o.data_op([], [t.output])

    def constraint(self, o, t):
        return o.data_op.name == self.data_input_name_attr

    def result_pattern(self, o, t):
        t.declare_internal_native_ir_value("input")
        o.new_data_op = o.ap_native_op("pd_op.data")
        o.new_data_op.name = lambda o, t: o.data_op.name
        o.new_data_op.shape = lambda o, t: o.data_op.shape
        o.new_data_op.dtype = lambda o, t: o.data_op.dtype
        o.new_data_op.place = lambda o, t: o.data_op.place
        o.new_data_op([], [t.input])
        o.down_spider = o.ap_native_op("ap_op.down_spider")
        o.down_spider([t.input], [t.output])


class InitNaiveLoadFromGlobalAccessTopoPass(access_topo_drr.DrrPass):

    def __init__(self, data_input_name):
        self.data_input_name_attr = pir.a_str(data_input_name)

    def source_pattern(self, o, t):
        o.data_op = o.ap_native_op("pd_op.data")
        o.data_op([], [t.output])

    def constraint(self, o, t):
        return o.data_op.name == self.data_input_name_attr

    def result_pattern(self, o, t):
        t.declare_internal_native_ir_value("input")
        o.new_data_op = o.ap_native_op("pd_op.data")
        o.new_data_op.name = lambda o, t: o.data_op.name
        o.new_data_op.shape = lambda o, t: o.data_op.shape
        o.new_data_op.dtype = lambda o, t: o.data_op.dtype
        o.new_data_op.place = lambda o, t: o.data_op.place
        o.new_data_op([], [t.input])
        o.load_from_global = o.ap_native_op("ap_op.load_from_global")
        o.load_from_global.index_func_unique_id = (
            lambda o, t: self.data_input_name_attr
        )
        o.load_from_global([t.input], [t.output])


class ReplaceWithLoadFromRegisterPass(access_topo_drr.DrrPass):

    def __init__(self, name, register_var_name):
        self.name = pir.a_str(name)
        self.register_var_name = pir.a_str(register_var_name)

    def source_pattern(self, o, t):
        o.data_op = o.ap_native_op("pd_op.data")
        o.data_op([], [t.input])
        o.load_from_global = o.ap_native_op("ap_op.load_from_global")
        o.load_from_global.index_func_unique_id = self.name
        o.load_from_global([t.input], [t.output])

    def result_pattern(self, o, t):
        o.load_from_register = o.ap_native_op("ap_op.load_from_register")
        o.load_from_register.name = lambda o, t: self.name
        o.load_from_register.register_var_name = (
            lambda o, t: self.register_var_name
        )
        o.load_from_register.type = lambda o, t: pir.a_type(t.output.type)
        o.load_from_register.symbolic_shape_or_data = lambda o, t: pir.a_symbol(
            t.output.get_symbolic_shape_or_data()
        )
        o.load_from_register([], [t.output])


class ReplaceWithStoreToRegisterPass(access_topo_drr.DrrPass):

    def __init__(self, name, register_var_name):
        self.name = pir.a_str(name)
        self.register_var_name = pir.a_str(register_var_name)

    def source_pattern(self, o, t):
        o.data_op = o.ap_native_op("pd_op.data")
        o.data_op([], [t.output])
        o.store_to_global_op = o.ap_native_op("ap_op.store_to_global")
        o.store_to_global_op.index_func_unique_id = self.name
        o.store_to_global_op([t.output, t.output_val], [])

    def result_pattern(self, o, t):
        o.store_to_register_op = o.ap_native_op("ap_op.store_to_register")
        o.store_to_register_op.name = lambda o, t: self.name
        o.store_to_register_op.register_var_name = (
            lambda o, t: self.register_var_name
        )
        o.store_to_register_op([t.output_val], [])


@access_topo_drr.register_drr_pass("down_spider_relu", tag="default")
class DownSpiderReluAccessTopoPass(access_topo_drr.DrrPass):

    def source_pattern(self, o, t):
        o.spider0 = o.ap_native_op("ap_op.down_spider")
        o.spider0([t.input], [t.tmp])
        o.relu1 = o.ap_native_op("pd_op.relu")
        o.relu1([t.tmp], [t.output])

    def result_pattern(self, o, t):
        o.fustion_op = o.ap_native_op("ap_op.down_spider")
        o.fustion_op([t.input], [t.output])


@access_topo_drr.register_drr_pass(
    "down_spider_load_from_global", tag="default"
)
class DownSpiderLoadFromGlobalAccessTopoPass(access_topo_drr.DrrPass):

    def source_pattern(self, o, t):
        o.spider0 = o.ap_native_op("ap_op.down_spider")
        o.spider0([t.input], [t.tmp])
        o.load_from_global_op = o.ap_native_op("ap_op.load_from_global")
        o.load_from_global_op([t.tmp], [t.output])

    def result_pattern(self, o, t):
        o.fustion_op = o.ap_native_op("ap_op.down_spider")
        o.fustion_op([t.input], [t.output])


@access_topo_drr.register_drr_pass("down_spider_up_spider", tag="default")
class DownSpiderUpSpiderAccessTopoPass(access_topo_drr.DrrPass):

    def source_pattern(self, o, t):
        o.down_spider_op = o.ap_native_op("ap_op.down_spider")
        o.down_spider_op([t.input], [t.tmp0])
        o.up_spider_op = o.ap_native_op("ap_op.up_spider")
        o.up_spider_op([t.tmp0, t.input], [])

    def result_pattern(self, o, t):
        pass


@access_topo_drr.register_drr_pass("left_down_spider_add", tag="default")
class LeftDownSpiderAddAccessTopoPass(access_topo_drr.DrrPass):

    def source_pattern(self, o, t):
        o.spider = o.ap_native_op("ap_op.down_spider")
        o.spider([t.input0], [t.tmp0])
        o.add = o.ap_native_op("pd_op.add")
        o.add([t.tmp0, t.input1], [t.output])

    def result_pattern(self, o, t):
        o.down_spider = o.ap_native_op("ap_op.down_spider")
        o.down_spider([t.input0], [t.output])
        o.up_spider = o.ap_native_op("ap_op.up_spider")
        o.up_spider([t.input0, t.input1], [])


@access_topo_drr.register_drr_pass("right_down_spider_add", tag="default")
class RightDownSpiderAddAccessTopoPass(access_topo_drr.DrrPass):

    def source_pattern(self, o, t):
        o.spider = o.ap_native_op("ap_op.down_spider")
        o.spider([t.input0], [t.tmp0])
        o.add = o.ap_native_op("pd_op.add")
        o.add([t.input1, t.tmp0], [t.output])

    def result_pattern(self, o, t):
        o.down_spider = o.ap_native_op("ap_op.down_spider")
        o.down_spider([t.input0], [t.output])
        o.up_spider = o.ap_native_op("ap_op.up_spider")
        o.up_spider([t.input0, t.input1], [])


@access_topo_drr.register_drr_pass("expand_up_spider", tag="default")
class ExpandUpSpiderAccessTopoPass(access_topo_drr.DrrPass):

    def source_pattern(self, o, t):
        o.expand = o.ap_native_op("pd_op.expand")
        o.expand([t.input1, t.input2], [t.expanded_input])
        o.up_spider = o.ap_native_op("ap_op.up_spider")
        o.up_spider([t.input0, t.expanded_input], [])

    def constraint(self, o, t):
        input_shape = t.input1.symbolic_shape_to_list()
        output_shape = t.expanded_input.symbolic_shape_to_list()
        rank_diff = len(output_shape) - len(input_shape)
        return rank_diff > 0

        # TODO: Get Inner Expanded Axes
        def GetInnerExpanded_axes(i):
            if input_shape[i] == output_shape[i + rank_diff]:
                return []
            else:
                return [i]

        input_rank = len(t.input1.symbolic_shape_to_list())
        inner_expanded_axes = ap.flat_map(
            GetInnerExpanded_axes, range(input_rank)
        )
        return rank_diff > 0 and len(inner_expanded_axes) == 0

    def result_pattern(self, o, t):
        t.declare_internal_native_ir_value("reduced_input")
        o.sum = o.ap_native_op("pd_op.sum")
        o.sum.axis = self.get_axis
        o.sum.keepdim = self.get_keepdim
        o.sum([t.input0], [t.reduced_input])
        o.up_spider = o.ap_native_op("ap_op.up_spider")
        o.up_spider([t.reduced_input, t.input1], [])

    def get_keepdim(self, o, t):
        return pir.a_bool(False)

    def get_axis(self, o, t):
        input_rank = len(t.input1.symbolic_shape_to_list())
        output_rank = len(t.expanded_input.symbolic_shape_to_list())
        axes = range(output_rank - input_rank)
        return pir.a_intarray(axes)


@access_topo_drr.register_drr_pass("cinn_broadcast_up_spider", tag="default")
class CinnBroadcastUpSpiderAccessTopoPass(access_topo_drr.DrrPass):

    def source_pattern(self, o, t):
        o.broadcast_op = o.ap_native_op("cinn_op.broadcast")
        o.broadcast_op([t.input1], [t.expanded_input])
        o.up_spider = o.ap_native_op("ap_op.up_spider")
        o.up_spider([t.input0, t.expanded_input], [])

    def constraint(self, o, t):
        input_shape = t.input1.symbolic_shape_to_list()
        output_shape = t.expanded_input.symbolic_shape_to_list()
        rank_diff = len(output_shape) - len(input_shape)
        return rank_diff > 0

        # TODO: Get Inner Expanded Axes
        def GetInnerExpanded_axes(i):
            if input_shape[i] == output_shape[i + rank_diff]:
                return []
            else:
                return [i]

        input_rank = len(t.input1.symbolic_shape_to_list())
        inner_expanded_axes = ap.flat_map(
            GetInnerExpanded_axes, range(input_rank)
        )
        return rank_diff > 0 and len(inner_expanded_axes) == 0

    def result_pattern(self, o, t):
        t.declare_internal_native_ir_value("reduced_input")
        o.sum = o.ap_native_op("pd_op.sum")
        o.sum.axis = self.get_axis
        o.sum.keepdim = self.get_keepdim
        o.sum([t.input0], [t.reduced_input])
        o.up_spider = o.ap_native_op("ap_op.up_spider")
        o.up_spider([t.reduced_input, t.input1], [])

    def get_keepdim(self, o, t):
        return pir.a_bool(False)

    def get_axis(self, o, t):
        input_rank = len(t.input1.symbolic_shape_to_list())
        output_rank = len(t.expanded_input.symbolic_shape_to_list())
        axes = range(output_rank - input_rank)
        return pir.a_intarray(axes)


@access_topo_drr.register_drr_pass("right_down_spider_up_spider", tag="default")
class RightDownSpiderUpSpiderAccessTopoPass(access_topo_drr.DrrPass):

    def source_pattern(self, o, t):
        o.expand = o.ap_native_op("ap_op.down_spider")
        o.expand([t.input1], [t.output1])
        o.up_spider = o.ap_native_op("ap_op.up_spider")
        o.up_spider([t.input0, t.output1], [])

    def result_pattern(self, o, t):
        o.up_spider = o.ap_native_op("ap_op.up_spider")
        o.up_spider([t.input0, t.input1], [])


@access_topo_drr.register_drr_pass("left_down_spider_up_spider", tag="default")
class LeftDownSpiderUpSpiderAccessTopoPass(access_topo_drr.DrrPass):

    def source_pattern(self, o, t):
        o.expand = o.ap_native_op("ap_op.down_spider")
        o.expand([t.input0], [t.output0])
        o.up_spider = o.ap_native_op("ap_op.up_spider")
        o.up_spider([t.output0, t.input1], [])

    def result_pattern(self, o, t):
        o.up_spider = o.ap_native_op("ap_op.up_spider")
        o.up_spider([t.input0, t.input1], [])


@access_topo_drr.register_drr_pass(
    "triangle_left_down_spider_up_spider", tag="default"
)
class TriangleLeftDownSpiderUpSpiderAccessTopoPass(access_topo_drr.DrrPass):

    def source_pattern(self, o, t):
        o.expand = o.ap_native_op("ap_op.down_spider")
        o.expand([t.input0], [t.output0])
        o.up_spider = o.ap_native_op("ap_op.up_spider")
        o.up_spider([t.input0, t.output0], [])

    def result_pattern(self, o, t):
        pass
