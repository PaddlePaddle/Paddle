# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid.framework as framework
from paddle.fluid.optimizer import Optimizer
import paddle.fluid.core as core
import numpy as np

registerd_op = {
    "elementwise_add": "AddParser",
    "matmul": "MatMulParser",
    "mul": "MulParser",
    "relu": "ReluParser",
    "softmax_with_cross_entropy": "SoftmaxWithCrossEntropyParser",
    "shape": "ShapeParser",
    "fill_constant": "FillConstantParser",
    "reduce_sum": "ReduceSumParser",
    # "reduce_sum_grad": "ReduceSumGradParser"
}
global_cnt = -1


class AscendHelper(object):
    def __init__(self):
        self.dtype_map = {
            0: core.GEDataType.DT_BOOL,
            1: core.GEDataType.DT_INT16,
            2: core.GEDataType.DT_INT32,
            3: core.GEDataType.DT_INT64,
            4: core.GEDataType.DT_FLOAT16,
            5: core.GEDataType.DT_FLOAT,
            6: core.GEDataType.DT_DOUBLE
        }

    def dtype2ge(self, dtype):
        assert dtype in self.dtype_map, "dtype[%d] is not supported %d" % (
            dtype)
        return self.dtype_map[dtype]


class AscendParserFactory(object):
    def __init__(self, graph, var2geop):
        self.graph = graph
        self.var2geop = var2geop

    def create_parse(self, parser_class):
        try:
            parser = globals()[parser_class](self.graph, self.var2geop)
            return parser
        except:
            raise ValueError("parser class %s does not exist" % parser_class)


class AscendParserBase(object):
    def __init__(self, graph, var2geop):
        self.graph = graph
        self.var2geop = var2geop
        self.op = None
        self.ascend_helper = AscendHelper()

    def get_ge_input(self, input_var_name):
        assert input_var_name in self.var2geop, "var %s not created before" % (
            input_var_name)
        return self.var2geop[input_var_name]

    def update_output(self, graph, geop_list):
        assert self.op.output_arg_names[
            0] not in self.var2geop, "var %s has been another op's output" % (
                self.op.output_arg_names[0])
        self.var2geop[self.op.output_arg_names[0]] = geop_list[-1]
        for geop in geop_list:
            self.graph.add_op(geop)
        # print("update_output for op", self.op.type)

    def apply(self, op):
        self.op = op
        assert self.op.type == self.parser_name, "op [%s] != parser_name[%s]" % (
            self.op.type, self.parser_name)
        print("begin to parse op %s" % (self.parser_name))
        geop_list = self._apply()
        self.update_output(self.graph, geop_list)

    def getid(self):
        global global_cnt
        global_cnt += 1
        return "." + str(global_cnt)

    def create_ge_tensor(self, shape, dtype, value):
        # just stub
        tensor_desc = core.GETensorDesc(
            core.GEShape(shape), core.GEFormat.FORMAT_ND,
            self.ascend_helper.dtype2ge(dtype))
        # tensor_desc = core.GETensorDesc(core.GEShape(shape), core.GEFormat.FORMAT_ND, core.GEDataType.DT_FLOAT)
        #c1_tensor_desc.set_real_dim_cnt(1)
        tensor = core.GETensor(tensor_desc)

        data = (value * np.ones((shape))).reshape(shape).astype(
            "float32")  #TODO paddle dtype to np type
        buf = data.tobytes()
        data_8 = np.frombuffer(buf, dtype=np.uint8)
        tensor.set_data(data_8)
        return tensor

    def create_shape_tensor(self):
        # just stub
        tensor_desc = core.GETensorDesc(
            core.GEShape([2]), core.GEFormat.FORMAT_ND,
            core.GEDataType.DT_INT32)
        #c1_tensor_desc.set_real_dim_cnt(1)
        tensor = core.GETensor(tensor_desc)

        data = np.ones(
            (2)).astype("int32").reshape([2])  #TODO paddle dtype to np type
        data[0] = 2
        buf = data.tobytes()
        data_8 = np.frombuffer(buf, dtype=np.uint8)
        tensor.set_data(data_8)
        return tensor


class AddParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(AddParser, self).__init__(graph, var2geop)
        self.parser_name = "elementwise_add"

    def _apply(self):
        data_x1_shape = self.get_ge_input(self.op.input_arg_names[0])
        data_x2_shape = self.get_ge_input(self.op.input_arg_names[1])
        add = core.GEOperatorFactory.create_operator(
            "add" + self.getid(), "Add").set_input(
                "x1", data_x1_shape).set_input("x2", data_x2_shape)
        return [add]


class ReduceSumParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(ReduceSumParser, self).__init__(graph, var2geop)
        self.parser_name = "reduce_sum"

    def _apply(self):
        x = self.get_ge_input(self.op.input_arg_names[0])
        axes = self.op.attr("dim")
        keep_dims = self.op.attr("keep_dim")
        print("axes in reduce_sum: ", axes)
        print("keep_dims in reduce_sum: ", keep_dims)
        reduce_sum = core.GEOperatorFactory.create_operator(
            "reduce_sum" + self.getid(), "ReduceSumD").set_input(
                "x", x).set_attr_vec_int32("axes", axes).set_attr_bool(
                    "keep_dims", keep_dims)
        return [reduce_sum]


class ReduceSumGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(ReduceSumGradParser, self).__init__(graph, var2geop)
        self.parser_name = "reduce_sum_grad"

    def _apply(self):
        print("self.op.input_arg_names[0]: ", self.op.input_arg_names[0])
        x = self.get_ge_input(self.op.input_arg_names[0])
        input = self.get_ge_input(self.op.input_arg_names[1])
        shape_tensor = self.create_shape_tensor()
        shape_tensor = core.GEOperatorFactory.create_operator(
            "shape" + self.getid(), "Shape").set_input("x", input)
        reduce_sum = core.GEOperatorFactory.create_operator(
            "broadcast_to_d" + self.getid(), "BroadcastTo").set_input(
                "x", x).set_input("shape", shape_tensor)
        return [shape_tensor, reduce_sum]


class MatMulParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(MatMulParser, self).__init__(graph, var2geop)
        self.parser_name = "matmul"

    def _apply(self):
        x1 = self.get_ge_input(self.op.input_arg_names[0])
        data_x2_shape = self.get_ge_input(self.op.input_arg_names[1])
        matmul = core.GEOperatorFactory.create_operator(
            "matmul" + self.getid(), "MatMul").set_input("x1", x1).set_input(
                "x2", data_x2_shape)
        return [matmul]


class MulParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(MulParser, self).__init__(graph, var2geop)
        self.parser_name = "mul"

    def _apply(self):
        data_x1_shape = self.get_ge_input(self.op.input_arg_names[0])
        data_x2_shape = self.get_ge_input(self.op.input_arg_names[1])
        matmul = core.GEOperatorFactory.create_operator(
            "mul" + self.getid(), "MatMul").set_input(
                "x1", data_x1_shape).set_input("x2", data_x2_shape)
        return [matmul]


class ReluParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(ReluParser, self).__init__(graph, var2geop)
        self.parser_name = "relu"

    def _apply(self):
        data_x1_shape = self.get_ge_input(self.op.input_arg_names[0])
        relu = core.GEOperatorFactory.create_operator(
            "relu" + self.getid(), "Relu").set_input("x", data_x1_shape)
        return [relu]


class SoftmaxWithCrossEntropyParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(SoftmaxWithCrossEntropyParser, self).__init__(graph, var2geop)
        self.parser_name = "softmax_with_cross_entropy"

    def _apply(self):
        label = self.get_ge_input(self.op.input_arg_names[0])
        logits = self.get_ge_input(self.op.input_arg_names[1])
        label_index = self.get_ge_input("label_index")

        reduce_max = core.GEOperatorFactory.create_operator(
            "reduce_max" + self.getid(), "ReduceMaxD").set_input(
                "x", logits).set_attr_vec_int32("axes", [1]).set_attr_bool(
                    "keep_dims", True)
        sub = core.GEOperatorFactory.create_operator(
            "sub" + self.getid(), "Sub").set_input("x1", logits).set_input(
                "x2", reduce_max)  #[2, 3]
        exp = core.GEOperatorFactory.create_operator("exp" + self.getid(),
                                                     "Exp").set_input("x", sub)
        reduce_sum = core.GEOperatorFactory.create_operator(
            "reduce_sum" + self.getid(), "ReduceSumD").set_input(
                "x", exp).set_attr_vec_int32("axes", [1]).set_attr_bool(
                    "keep_dims", True)
        log = core.GEOperatorFactory.create_operator(
            "log" + self.getid(), "Log").set_input("x", reduce_sum)
        tmp = core.GEOperatorFactory.create_operator(
            "sub" + self.getid(), "Sub").set_input("x1", log).set_input("x2",
                                                                        sub)
        res = core.GEOperatorFactory.create_operator(
            "gather" + self.getid(), "GatherNd").set_input("x", tmp).set_input(
                "indices", label_index)
        return [reduce_max, sub, exp, reduce_sum, log, tmp, res]


class ShapeParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(ShapeParser, self).__init__(graph, var2geop)
        self.parser_name = "shape"

    def _apply(self):
        x = self.get_ge_input(self.op.input_arg_names[0])
        shape = core.GEOperatorFactory.create_operator(
            "shape" + self.getid(), "Shape").set_input("x", x)
        return [shape]


class FillConstantParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(FillConstantParser, self).__init__(graph, var2geop)
        self.parser_name = "fill_constant"

    def _apply(self):
        shape = self.op.attr("shape")
        dtype = self.op.attr("dtype")
        value = self.op.attr("value")
        print("shape: ", shape)
        print("dtype: ", dtype)
        print("value: ", value)
        tensor = self.create_ge_tensor(shape, dtype, value)
        const = core.GEOperatorFactory.create_operator(
            "const" + self.getid(), "Const").set_attr_tensor("value", tensor)
        if self.op.block.var(self.op.output('Out')[0]).persistable:
            print("%s is Persistable in fill_constant" %
                  (self.op.output('Out')[0]))
            var = core.GEOperatorFactory.create_operator(
                self.op.output('Out')[0], "Variable")
            var.update_output_desc("y",
                                   core.GETensorDesc(
                                       core.GEShape(shape),
                                       core.GEFormat.FORMAT_ND,
                                       core.GEDataType.DT_FLOAT))
            assign = core.GEOperatorFactory.create_operator(
                "assign" + self.getid(), "Assign").set_input(
                    "value", const).set_input("ref", var)
        else:
            print(
                "self.op.output('Out')[0] is not persistable in fill_constant")
        return [const]
