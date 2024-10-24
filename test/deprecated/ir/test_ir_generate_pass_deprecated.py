#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

import paddle
from paddle.base import core
from paddle.incubate.passes import ir
from paddle.static import InputSpec


# 0: ewadd(X=mul(X=x, Y=w), Y=b) => fc(Input=x, W=w, Bias=b)
# 1: relu(X=ewadd(X=mul(X=x, Y=w), Y=b)) => fc(Input=x, W=w, Bias=b)
@ir.RegisterPass
def generate_fc_fuse():
    def create_pass_pair(with_relu):
        def pattern(x, w, b):
            mul = ir.PassDesc.OP.mul(X=x, Y=w)
            ewadd = ir.PassDesc.OP.elementwise_add(X=mul, Y=b)
            if with_relu:
                return ir.PassDesc.OP.relu(X=ewadd)
            else:
                return ewadd

        def replace(x, w, b):
            fc = ir.PassDesc.OP.fc(Input=x, W=w, Bias=b)
            fc.Attr("in_num_col_dims").MappedPattern(
                op="mul", name="x_num_col_dims"
            )
            if with_relu:
                fc.SetAttr("activation_type", "relu")
            return fc

        return pattern, replace

    return list(map(create_pass_pair, [True, False]))


# add(X=add(X=x, Y=y), Y=z) => sum(X=[x, y, z])
@ir.RegisterPass
def multi_add_to_sum_v1():
    pattern = lambda x, y, z: paddle.add(paddle.add(x, y), z)
    replace = lambda x, y, z: paddle.add_n([x, y, z])
    return pattern, replace


@ir.RegisterPass
def multi_add_to_sum_v2():
    def pattern(x, y, z):
        ewadd1 = ir.PassDesc.OP.elementwise_add(X=x, Y=y)
        ewadd2 = ir.PassDesc.OP.elementwise_add(X=ewadd1, Y=z)
        return ewadd2

    replace = lambda x, y, z: ir.PassDesc.OP.sum(X=[x, y, z])
    return pattern, replace


@ir.RegisterPass
def multi_add_to_sum_v3():
    pattern = lambda x, y, z: paddle.add(paddle.add(x, y), z)
    replace = lambda x, y, z: ir.PassDesc.OP.sum(X=[x, y, z])
    return pattern, replace


# mul(x, y1), mul(x, y2) => slice(mul(x, concat(y1, y2)))
@ir.RegisterPass(
    input_specs={
        'x': InputSpec([16, 32]),
        'y1': InputSpec([32, 12]),
        'y2': InputSpec([32, 48]),
    }
)
def generate_combine_mul_v1():
    def pattern(x, y1, y2):
        mul1 = paddle.matmul(x, y1)
        mul2 = paddle.matmul(x, y2)
        return mul1, mul2

    def replace(x, y1, y2):
        concat_out = paddle.concat([y1, y2], axis=-1)
        mul_out = paddle.matmul(x, concat_out)
        out1 = paddle.slice(mul_out, axes=[1], starts=[0], ends=[12])
        out2 = paddle.slice(mul_out, axes=[1], starts=[12], ends=[60])
        return out1, out2

    return pattern, replace


@ir.RegisterPass
def generate_combine_mul_v2():
    def pattern(x, y1, y2):
        mul1 = ir.PassDesc.OP.matmul_v2(X=x, Y=y1)
        mul2 = ir.PassDesc.OP.matmul_v2(X=x, Y=y2)
        return mul1, mul2

    def replace(x, y1, y2):
        concat = ir.PassDesc.OP.concat(X=[y1, y2])
        matmul = ir.PassDesc.OP.matmul_v2(X=x, Y=concat)
        out1 = ir.PassDesc.OP.slice(Input=matmul)
        out2 = ir.PassDesc.OP.slice(Input=matmul)
        return out1, out2

    return pattern, replace


# reshape(reshape(x)) => x
@ir.RegisterPass(input_specs={'x': InputSpec([10, 16, 16])})
def generate_simplify_inference_v1():
    def pattern(x):
        transpose = paddle.transpose(x, [0, 2, 1])
        return paddle.transpose(transpose, [0, 2, 1])

    return pattern, lambda x: x


@ir.RegisterPass
def generate_simplify_inference_v2():
    def pattern(x):
        op1 = ir.PassDesc.OP.transpose2
        op2 = ir.PassDesc.OP.transpose2
        # op2.Attr("axis").EQ(op1.Attr("axis"))
        return op2(X=op1(X=x).Output("Out")).Output("Out")

    return pattern, lambda x: x


@ir.RegisterPass
def generate_layer_norm_fuse_pass():
    def pattern(x, gamma, beta):
        gamma.Attr("shape").Size().EQ(1)
        gamma.Attr("shape")[0].EQ(x.Attr("shape")[-1])
        beta.Attr("shape").EQ(gamma.Attr("shape"))

        mean1 = ir.PassDesc.OP.reduce_mean(X=x)
        mean1.SetAttr("dim", [-1])
        mean1.SetAttr("reduce_all", False)
        mean1.SetAttr("keep_dim", True)
        ewsub = ir.PassDesc.OP.elementwise_sub(X=x, Y=mean1)
        pow = ir.PassDesc.OP.pow(X=ewsub)
        pow.SetAttr("factor", 2.0)
        mean2 = ir.PassDesc.OP.reduce_mean(X=pow)
        mean2.SetAttr("dim", [-1])
        mean2.SetAttr("reduce_all", False)
        mean2.SetAttr("keep_dim", True)
        scale = ir.PassDesc.OP.scale(X=mean2)
        sqrt = ir.PassDesc.OP.sqrt(X=scale)
        ewdiv = ir.PassDesc.OP.elementwise_sub(X=ewsub, Y=sqrt)
        ewmul = ir.PassDesc.OP.elementwise_mul(X=ewdiv, Y=gamma)
        return ir.PassDesc.OP.elementwise_add(X=ewmul, Y=beta)

    def replace(x, gamma, beta):
        layer_norm = ir.PassDesc.OP.layer_norm(X=x, Scale=gamma, Bias=beta)
        layer_norm.SetAttr("begin_norm_axis", x.Attr("shape").Size() - 1)
        layer_norm.Attr("epsilon").MappedPattern(op="scale", name="bias")
        layer_norm.SetAttr("is_test", True)
        return layer_norm.Output("Y")

    return pattern, replace


@ir.RegisterPass
def unimplemented_operand_exception():
    def pattern(x, y):
        return ir.PassDesc.OP.elementwise_add(X=x, Y=y)

    def replace(x, y):
        out = ir.PassDesc.OP.elementwise_add(X=x, Y=y)
        out.SetAttr("axis", x.Attr("shape") - 1)
        return out

    return pattern, replace


@ir.RegisterPass
def unimplemented_operation_exception():
    def pattern(x, y):
        return ir.PassDesc.OP.elementwise_add(X=x, Y=y)

    def replace(x, y):
        out = ir.PassDesc.OP.elementwise_add(X=x, Y=y)
        out.SetAttr("axis", x.Attr("shape").Size() + 1)
        return out

    return pattern, replace


def get_multi_pass_desc_from_str(s):
    multi_pass_desc = ir.pass_desc_pb2.MultiPassDesc()
    multi_pass_desc.ParseFromString(s)
    return multi_pass_desc


class TestGeneratePass(unittest.TestCase):
    def convert_ops_to_op_dicts(self, ops):
        op_dicts = {}
        for op in ops:
            op_list = op_dicts.get(op.type)
            if isinstance(op_list, list):
                op_list.append(op)
            else:
                op_dicts[op.type] = [op]
        return op_dicts

    def test_has_attr(self):
        self.assertFalse(hasattr(ir.PassDesc.OP, '__name__'))

    def test_exception(self):
        paddle.enable_static()
        program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(program, startup_program):
            x = paddle.static.data("x", [10, 10], "float32")
            y = paddle.static.data("y", [10, 10], "float32")
            paddle.add(x, y)
        graph = core.Graph(program.desc)
        with self.assertRaises(NotImplementedError):
            core.get_pass("unimplemented_operand_exception").apply(graph)
        with self.assertRaises(NotImplementedError):
            core.get_pass("unimplemented_operation_exception").apply(graph)

    def test_generate_fc_fuse(self):
        def _check_fc_fuse_pass(pass_desc, with_relu):
            pattern_op_dicts = self.convert_ops_to_op_dicts(pass_desc.pattern)
            replace_op_dicts = self.convert_ops_to_op_dicts(pass_desc.replace)
            self.assertEqual(len(pattern_op_dicts.get("mul", [])), 1)
            self.assertEqual(
                len(pattern_op_dicts.get("elementwise_add", [])), 1
            )
            if with_relu:
                self.assertEqual(len(pattern_op_dicts.get("relu", [])), 1)
                pattern_op_num = 3  # relu, ewadd, mul
            else:
                pattern_op_num = 2  # ewadd, mul
            self.assertEqual(len(pass_desc.var_maps), 4)
            self.assertEqual(len(pass_desc.pattern), pattern_op_num)
            self.assertEqual(len(pass_desc.replace), 1)
            self.assertEqual(len(pass_desc.op_attr_maps), 1)

        helper = ir.RegisterPassHelper(generate_fc_fuse())
        s = helper.SerializeMultiPassDesc()
        multi_pass_desc = get_multi_pass_desc_from_str(s)
        self.assertEqual(len(multi_pass_desc.pass_descs), 2)
        _check_fc_fuse_pass(multi_pass_desc.pass_descs[0], True)
        _check_fc_fuse_pass(multi_pass_desc.pass_descs[1], False)

    def check_multi_add_to_sum(self, pass_type):
        program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(program, startup_program):
            x = paddle.static.data("x", [10, 10, 10], "float32")
            y = paddle.static.data("y", [10, 10, 10], "float32")
            z = paddle.static.data("z", [10, 10, 10], "float32")
            add_1 = paddle.add(paddle.add(x, y), z)
            matmul_1 = paddle.matmul(add_1, z)
            add_tmp = paddle.add(x, y)
            add_2 = paddle.add(add_tmp, z)
            matmul_2 = paddle.matmul(add_2, add_tmp)
            out = paddle.add(matmul_1, matmul_2)
        graph = core.Graph(program.desc)
        before_node_nums = len(graph.nodes())
        core.get_pass(pass_type).apply(graph)
        after_node_nums = len(graph.nodes())
        self.assertEqual(after_node_nums, before_node_nums - 2)
        after_program = paddle.base.framework.IrGraph(graph).to_program()
        executor = paddle.static.Executor(paddle.CPUPlace())
        executor.run(startup_program)
        feed = {
            "x": np.random.random([10, 10, 10]).astype("float32"),
            "y": np.random.random([10, 10, 10]).astype("float32"),
            "z": np.random.random([10, 10, 10]).astype("float32"),
        }
        before_out = executor.run(program, feed=feed, fetch_list=[out])
        after_out = executor.run(after_program, feed=feed, fetch_list=[out])
        np.testing.assert_allclose(before_out, after_out, rtol=1e-05)

    def test_multi_add_to_sum(self):
        paddle.enable_static()
        self.check_multi_add_to_sum("multi_add_to_sum_v1")
        self.check_multi_add_to_sum("multi_add_to_sum_v2")
        self.check_multi_add_to_sum("multi_add_to_sum_v3")

    def test_generate_combine_mul_v1(self):
        paddle.enable_static()
        program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(program, startup_program):
            x = paddle.static.data("x", [16, 32])
            y = paddle.static.data("y", [32, 12])
            z = paddle.static.data("z", [32, 48])
            out1 = paddle.matmul(x, y)
            out2 = paddle.matmul(x, z)
        graph = core.Graph(program.desc)
        before_node_nums = len(graph.nodes())
        core.get_pass("generate_combine_mul_v1").apply(graph)
        after_node_nums = len(graph.nodes())
        self.assertEqual(after_node_nums, before_node_nums + 4)
        after_program = paddle.base.framework.IrGraph(graph).to_program()
        executor = paddle.static.Executor(paddle.CPUPlace())
        executor.run(startup_program)
        feed = {
            "x": np.random.random([16, 32]).astype("float32"),
            "y": np.random.random([32, 12]).astype("float32"),
            "z": np.random.random([32, 48]).astype("float32"),
        }
        before_out1, before_out2 = executor.run(
            program, feed=feed, fetch_list=[out1, out2]
        )
        after_out1, after_out2 = executor.run(
            after_program, feed=feed, fetch_list=[out1, out2]
        )
        np.testing.assert_allclose(before_out1, after_out1, rtol=1e-05)
        np.testing.assert_allclose(before_out2, after_out2, rtol=1e-05)

    def test_generate_combine_mul_v2(self):
        helper = ir.RegisterPassHelper([generate_combine_mul_v2()])
        s = helper.SerializeMultiPassDesc()
        multi_pass_desc = get_multi_pass_desc_from_str(s)
        self.assertEqual(len(multi_pass_desc.pass_descs), 1)
        pass_desc = multi_pass_desc.pass_descs[0]
        self.assertEqual(len(pass_desc.var_maps), 5)
        self.assertEqual(len(pass_desc.pattern), 2)
        self.assertEqual(len(pass_desc.replace), 4)
        pattern_op_dicts = self.convert_ops_to_op_dicts(pass_desc.pattern)
        replace_op_dicts = self.convert_ops_to_op_dicts(pass_desc.replace)
        self.assertEqual(len(pattern_op_dicts.get("matmul_v2", [])), 2)
        self.assertEqual(len(replace_op_dicts.get("concat", [])), 1)
        self.assertEqual(len(replace_op_dicts.get("matmul_v2", [])), 1)
        self.assertEqual(len(replace_op_dicts.get("slice", [])), 2)

    def check_generate_simplify_inference(self, pass_type):
        paddle.enable_static()
        program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(program, startup_program):
            x = paddle.static.data("x", [10, 16, 16], "float32")
            x1 = paddle.transpose(paddle.transpose(x, [0, 2, 1]), [0, 2, 1])
            tmp = paddle.transpose(x, [0, 2, 1])
            x2 = paddle.transpose(tmp, [0, 2, 1])
            out = paddle.add(x1, paddle.matmul(x2, tmp))
        graph = core.Graph(program.desc)
        before_node_nums = len(graph.nodes())
        core.get_pass(pass_type).apply(graph)
        after_node_nums = len(graph.nodes())
        self.assertEqual(after_node_nums, before_node_nums - 6)
        after_program = paddle.base.framework.IrGraph(graph).to_program()
        executor = paddle.static.Executor(paddle.CPUPlace())
        executor.run(startup_program)
        feed = {"x": np.random.random([10, 16, 16]).astype("float32")}
        before_out = executor.run(program, feed=feed, fetch_list=[out])
        after_out = executor.run(after_program, feed=feed, fetch_list=[out])
        np.testing.assert_allclose(before_out, after_out, rtol=1e-05)

    def test_generate_simplify_inference(self):
        self.check_generate_simplify_inference("generate_simplify_inference_v1")
        self.check_generate_simplify_inference("generate_simplify_inference_v2")

    def test_generate_layer_norm_fuse_pass(self):
        paddle.enable_static()
        program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(program, startup_program):
            x = paddle.static.data("x", [3, 64, 120], "float32")
            gamma = paddle.static.create_parameter(
                shape=[120], dtype="float32", is_bias=True
            )
            beta = paddle.static.create_parameter(
                shape=[120], dtype="float32", is_bias=True
            )

            x_sub_mean = x - paddle.mean(x, axis=-1, keepdim=True)
            std_dev = paddle.mean(x_sub_mean.pow(2), axis=-1, keepdim=True)
            lnorm = x_sub_mean - (std_dev + 1e-5).sqrt()
            out = lnorm * gamma + beta
        graph = core.Graph(program.desc)
        before_node_nums = len(graph.nodes())
        core.get_pass("generate_layer_norm_fuse_pass").apply(graph)
        after_node_nums = len(graph.nodes())
        self.assertEqual(after_node_nums, before_node_nums - 14)
        after_program = paddle.base.framework.IrGraph(graph).to_program()
        executor = paddle.static.Executor(paddle.CPUPlace())
        executor.run(startup_program)
        feed = {"x": np.random.random([3, 64, 120]).astype("float32")}
        before_out = executor.run(program, feed=feed, fetch_list=[out])
        after_out = executor.run(after_program, feed=feed, fetch_list=[out])
        np.testing.assert_allclose(before_out, after_out, rtol=1e-05)
