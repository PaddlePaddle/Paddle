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
import paddle
from paddle.static import InputSpec
import paddle.fluid.core as core
import paddle.fluid.ir_pass as ir_pass
import numpy as np


@ir_pass.RegisterPass("test_generate_pass_v1", False, False)
def generate_pass_v1():
    # pattern
    def pattern(x, w, b, out):
        mul = ir_pass.Op("matmul_v2")
        mul.SetInput(X=x, Y=w).SetOutput(out=ir_pass.Var("out"))
        mul.Attr("use_mkldnn").Set(False)
        mul.Attr("scale_y").Set([1.0])
        ewadd = ir_pass.Op("elementwise_add")
        ewadd.SetInput(X=mul.Output("out")[0], bias=b).SetOutput(out=out)
        return mul, ewadd

    # replace
    def replace(x, w, b, out):
        fc = ir_pass.Op("fc")
        fc.SetInput(Input=x, W=w, Bias=b).SetOutput(Out=out)
        # attr map
        fc.Attr("in_num_col_dims").Reuse("mul", "x_num_col_dims")
        fc.Attr("activation_type").Set("")
        return fc

    return pattern, replace


@ir_pass.RegisterPass("test_generate_pass_v2")
def generate_pass_v2():
    # pattern
    def pattern(x, y, z):
        add_out = paddle.add(x, y)
        return paddle.add(add_out, z)

    # replace
    def replace(x, y, z):
        return paddle.add_n([x, y, z])

    return pattern, replace


class FCFusePassTest(unittest.TestCase):
    def test_check_pass_v1(self):
        paddle.enable_static()
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            data = paddle.static.data("data", [32, 32])
            weight = paddle.static.data("weight", [32, 32])
            bias = paddle.static.data("bias", [32, 32])
            relu_out = paddle.nn.functional.relu(data)
            mul_out = paddle.matmul(relu_out, weight)
            ewadd_out = paddle.add(mul_out, bias, name="out")
        graph = core.Graph(main_program.desc)
        before_nodes_num = len(graph.nodes())
        test_pass = core.get_pass("test_generate_pass_v1")
        test_pass.apply(graph)
        trans_pass = core.get_pass("graph_to_program_pass")
        after_program = paddle.static.Program()
        trans_pass.set_not_owned("program", after_program.desc)
        trans_pass.apply(graph)
        after_program.blocks = [
            paddle.fluid.framework.Block(after_program, i)
            for i in range(after_program.desc.num_blocks())
        ]
        after_program._sync_with_cpp()
        exe = paddle.static.Executor()
        feed_vars = {
            "data": np.random.random([32, 32]).astype("float32"),
            "weight": np.random.random([32, 32]).astype("float32"),
            "bias": np.random.random([32, 32]).astype("float32"),
        }
        before_ret, = exe.run(main_program, feed=feed_vars, fetch_list=["out"])
        after_ret, = exe.run(after_program, feed=feed_vars, fetch_list=["out"])
        after_nodes_num = len(graph.nodes())
        self.assertEqual(before_nodes_num, after_nodes_num + 2)
        #self.assertTrue(np.allclose(before_ret, after_ret))

    @unittest.skipIf(True, "")
    def test_check_pass_v2(self):
        paddle.enable_static()
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data("x", [10])
            y = paddle.static.data("y", [10])
            z = paddle.static.data("z", [10])
            add_out = paddle.add(x, y)
            paddle.add(add_out, z, name="out")
        graph = core.Graph(main_program.desc)
        before_nodes_num = len(graph.nodes())
        test_pass = core.get_pass("test_generate_pass_v2")
        test_pass.apply(graph)
        trans_pass = core.get_pass("graph_to_program_pass")
        after_program = paddle.static.Program()
        trans_pass.set_not_owned("program", after_program.desc)
        trans_pass.apply(graph)
        after_program.blocks = [
            paddle.fluid.framework.Block(after_program, i)
            for i in range(after_program.desc.num_blocks())
        ]
        after_program._sync_with_cpp()
        exe = paddle.static.Executor()
        feed_vars = {
            "x": np.random.randn(10).astype("float32"),
            "y": np.random.randn(10).astype("float32"),
            "z": np.random.randn(10).astype("float32"),
        }
        before_ret, = exe.run(main_program, feed=feed_vars, fetch_list=["out"])
        after_ret, = exe.run(after_program, feed=feed_vars, fetch_list=["out"])
        after_nodes_num = len(graph.nodes())
        self.assertEqual(before_nodes_num, after_nodes_num + 2)
        self.assertTrue(all(before_ret == after_ret))
