#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.op import Operator

paddle.enable_static()


class TestSGDOp(OpTest):
    def setUp(self):
        self.op_type = "sgd"
        self.conf()
        w = np.random.random((self.h, self.w)).astype("float32")
        g = np.random.random((self.h, self.w)).astype("float32")
        lr = np.array([0.1]).astype("float32")

        self.inputs = {'Param': w, 'Grad': g, 'LearningRate': lr}
        self.outputs = {'ParamOut': w - lr * g}

    def conf(self):
        self.h = 102
        self.w = 105

    def test_check_output(self):
        self.check_output()


class TestSGDOpCase8X(TestSGDOp):
    def conf(self):
        self.h = 10
        self.w = 64


class TestSparseSGDOp(unittest.TestCase):
    def check_with_place(self, place):
        scope = core.Scope()

        # create and initialize Grad Variable
        height = 10
        rows = [0, 4, 7]
        self.conf()

        grad_selected_rows = scope.var('Grad').get_selected_rows()
        grad_selected_rows.set_height(height)
        grad_selected_rows.set_rows(rows)
        np_array = np.ones((len(rows), self.row_numel)).astype("float32")
        np_array[0, 0] = 2.0
        np_array[2, 8] = 4.0

        grad_tensor = grad_selected_rows.get_tensor()
        grad_tensor.set(np_array, place)

        # create and initialize Param Variable
        param = scope.var('Param').get_tensor()
        param_array = np.full((height, self.row_numel), 5.0).astype("float32")
        param.set(param_array, place)

        # create and initialize LeraningRate Variable
        lr = scope.var('LearningRate').get_tensor()
        lr_array = np.full((1), 2.0).astype("float32")
        lr.set(lr_array, place)

        # create and run sgd operator
        sgd_op = Operator(
            "sgd",
            Param='Param',
            Grad='Grad',
            ParamOut='Param',
            LearningRate='LearningRate',
        )
        sgd_op.run(scope, place)

        # get and compare result
        result_array = np.array(param)

        # rows[0] = 0, 5.0 - 2.0 * 2.0
        self.assertAlmostEqual(1.0, result_array[rows[0], 0])
        # rows[0] = 0, 5.0 - 2.0 * 1.0
        self.assertAlmostEqual(3.0, result_array[rows[0], 2])
        # 5.0 - 2.0 * 0.0
        self.assertAlmostEqual(5.0, result_array[1, 0])
        # rows[1] = 4, 5.0 - 2.0 * 1.0
        self.assertAlmostEqual(3.0, result_array[rows[1], 10])
        # 5.0 - 2.0 * 0.0
        self.assertAlmostEqual(5.0, result_array[5, 8])
        # rows[2] = 7, 5.0 - 2.0 * 1.0
        self.assertAlmostEqual(3.0, result_array[rows[2], 1])
        # rows[2] = 7, 5.0 - 2.0 * 4.0
        self.assertAlmostEqual(-3.0, result_array[rows[2], 8])

    def test_sparse_sgd(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        for place in places:
            self.check_with_place(place)

    def conf(self):
        self.row_numel = 12


class TestSparseSGDOpCase8X(TestSparseSGDOp):
    def conf(self):
        self.row_numel = 16


class TestSGDOpOptimizeSelectedRows(unittest.TestCase):
    def check_with_place(self, place):
        scope = core.Scope()

        row_width = 12
        # create and initialize Grad Variable
        grad_height = 10
        grad_rows = [0, 4, 7]

        grad_selected_rows = scope.var('Grad').get_selected_rows()
        grad_selected_rows.set_height(grad_height)
        grad_selected_rows.set_rows(grad_rows)
        grad_array = np.ones((len(grad_rows), row_width)).astype("float32")
        grad_array[0, 0] = 2.0
        grad_array[2, 8] = 4.0

        grad_tensor = grad_selected_rows.get_tensor()
        grad_tensor.set(grad_array, place)

        # create and initialize Param Variable
        # create and initialize W Variable
        param_rows = [0, 1, 2, 3, 4, 5, 6, 7]

        # init Param
        w_selected_rows = scope.var('Param').get_selected_rows()
        w_selected_rows.set_height(len(param_rows))
        w_selected_rows.set_rows(param_rows)
        w_selected_rows.sync_index()
        w_array = np.ones((len(param_rows), row_width)).astype("float32")
        for i in range(len(param_rows)):
            w_array[i] *= i
        w_tensor = w_selected_rows.get_tensor()
        w_tensor.set(w_array, place)

        w_before_optimize = np.array(w_tensor)

        # create and initialize LeraningRate Variable
        lr_value = 0.1
        lr = scope.var('LearningRate').get_tensor()
        lr_array = np.full((1), lr_value).astype("float32")
        lr.set(lr_array, place)

        # optimize with Python
        w_after_optimize = np.copy(w_before_optimize)
        for index, id in enumerate(grad_rows):
            w_after_optimize[id] = (
                w_before_optimize[id] - lr_value * grad_array[index]
            )

        # create and run sgd operator
        sgd_op = Operator(
            "sgd",
            Param='Param',
            Grad='Grad',
            ParamOut='Param',
            LearningRate='LearningRate',
        )
        sgd_op.run(scope, place)

        # get and compare result
        result_array = np.array(w_tensor)
        assert (result_array == w_after_optimize).all()

    def test_sparse_parameter_sgd(self):
        places = [core.CPUPlace()]
        # do not support GPU kernel currently
        for place in places:
            self.check_with_place(place)


class TestSGDOpWithLargeInput(unittest.TestCase):
    def runTest(self):
        paddle.enable_static()
        data = fluid.layers.fill_constant(shape=[1], value=128, dtype='int64')
        label = fluid.layers.fill_constant(
            shape=[1, 150], value=0.5, dtype='float32'
        )
        emb = paddle.static.nn.embedding(
            input=data, size=(10000000, 150), dtype='float32'
        )
        out = paddle.nn.functional.normalize(x=emb, axis=-1)

        cost = paddle.nn.functional.square_error_cost(input=out, label=label)
        avg_cost = paddle.mean(cost)
        sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
        sgd_optimizer.minimize(avg_cost)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        compiled_prog = fluid.compiler.CompiledProgram(
            fluid.default_main_program()
        )
        result = exe.run(compiled_prog, fetch_list=[avg_cost])


class TestSGDV2(unittest.TestCase):
    def test_sgd_dygraph(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear = paddle.nn.Linear(13, 5)
        # This can be any optimizer supported by dygraph.
        adam = paddle.optimizer.SGD(
            learning_rate=0.01,
            parameters=linear.parameters(),
            weight_decay=0.01,
        )
        out = linear(a)
        out.backward()
        adam.step()
        adam.clear_gradients()

    def test_sgd(self):
        paddle.enable_static()

        def check_sgd_optimizer(optimizer_attr):
            init_program = paddle.static.Program()
            program = paddle.static.Program()
            block = program.global_block()
            mul_x = block.create_parameter(
                dtype="float32",
                shape=[5, 10],
                lod_level=0,
                name="mul.x",
                optimize_attr=optimizer_attr,
            )
            mul_y = block.create_var(
                dtype="float32", shape=[10, 8], lod_level=0, name="mul.y"
            )
            mul_out = block.create_var(
                dtype="float32", shape=[5, 8], lod_level=0, name="mul.out"
            )
            mean_out = block.create_var(
                dtype="float32", shape=[1], lod_level=0, name="mean.out"
            )
            block.append_op(
                type="mul",
                inputs={"X": mul_x, "Y": mul_y},
                outputs={"Out": mul_out},
                attrs={"x_num_col_dims": 1},
            )
            block.append_op(
                type="mean", inputs={"X": mul_out}, outputs={"Out": mean_out}
            )
            sgd_optimizer = paddle.optimizer.SGD(learning_rate=0.01)
            opts, _ = sgd_optimizer.minimize(mean_out, init_program)
            return opts

        opts = check_sgd_optimizer({'learning_rate': 1.1})
        self.assertEqual(len(opts), 2)
        self.assertEqual([op.type for op in opts], ["scale", "sgd"])

        opts = check_sgd_optimizer({'learning_rate': 1.0})
        self.assertEqual(len(opts), 1)
        self.assertEqual([op.type for op in opts], ["sgd"])

    def test_raise_error(self):
        self.assertRaises(ValueError, paddle.optimizer.SGD, learning_rate=None)

    def test_sgd_group_dygraph(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear_1 = paddle.nn.Linear(13, 5)
        linear_2 = paddle.nn.Linear(5, 3)
        # This can be any optimizer supported by dygraph.
        adam = paddle.optimizer.SGD(
            learning_rate=0.01,
            parameters=[
                {'params': linear_1.parameters()},
                {
                    'params': linear_2.parameters(),
                    'weight_decay': 0.001,
                    'learning_rate': 0.1,
                },
            ],
            weight_decay=0.01,
        )
        out = linear_1(a)
        out = linear_2(out)
        out.backward()
        adam.step()
        adam.clear_gradients()


class TestSGDMultiPrecision2_0(unittest.TestCase):
    def dygraph_sgd_mp(self, mp):
        paddle.disable_static()
        paddle.seed(10)
        paddle.set_device('gpu')
        input = paddle.randn((2, 2))
        model = paddle.nn.Linear(2, 2)
        optimizer = paddle.optimizer.SGD(
            parameters=model.parameters(), multi_precision=mp
        )
        if mp:
            model = paddle.amp.decorate(models=model, level='O2')
            scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

        for idx in range(5):
            if mp:
                with paddle.amp.auto_cast(level='O2'):
                    output = model(input)
                    loss = paddle.mean(output)
                scaled = scaler.scale(loss)
                scaled.backward()
                scaler.minimize(optimizer, scaled)
                optimizer.clear_grad()
            else:
                output = model(input)
                loss = paddle.mean(output)
                optimizer.step()
                optimizer.clear_grad()

        return output, model.parameters()

    def static_sgd_mp(self, mp):
        paddle.enable_static()
        paddle.seed(10)
        np.random.seed(10)
        exe = paddle.static.Executor('gpu')
        train_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        optimizer = paddle.optimizer.SGD(multi_precision=mp)

        if mp:
            optimizer = paddle.static.amp.decorate(
                optimizer,
                init_loss_scaling=128.0,
                use_dynamic_loss_scaling=True,
                use_pure_fp16=True,
                use_fp16_guard=False,
            )
        with paddle.static.program_guard(train_program, startup_program):
            if mp:
                data = paddle.static.data(
                    shape=[2, 2], name='X', dtype='float16'
                )
            else:
                data = paddle.static.data(
                    shape=[2, 2], name='X', dtype='float32'
                )
            hidden = paddle.static.nn.fc(x=data, size=10)
            loss = paddle.mean(hidden)
            optimizer.minimize(loss)
        exe.run(startup_program)

        if mp:
            optimizer.amp_init(place='gpu', scope=paddle.static.global_scope())
            x = np.random.random(size=(2, 2)).astype('float16')
        else:
            x = np.random.random(size=(2, 2)).astype('float32')
        out = []
        for idx in range(5):
            (loss_data,) = exe.run(
                train_program, feed={"X": x}, fetch_list=[loss.name]
            )
            out.append(loss_data)
        return out

    def test_main(self):
        if not paddle.is_compiled_with_cuda():
            return
        "Test dygraph mode"
        output1_dy, params1_dy = self.dygraph_sgd_mp(mp=True)
        output2_dy, params2_dy = self.dygraph_sgd_mp(mp=False)
        np.testing.assert_allclose(
            output1_dy.astype('float32').numpy(),
            output2_dy.astype('float32').numpy(),
            rtol=1e-05,
            atol=0.1,
        )
        for idx in range(len(params1_dy)):
            np.testing.assert_allclose(
                params1_dy[idx].astype('float32').numpy(),
                params2_dy[idx].astype('float32').numpy(),
                rtol=1e-05,
                atol=0.1,
            )
        "Test static graph mode"
        output1_st = self.static_sgd_mp(mp=True)
        output2_st = self.static_sgd_mp(mp=False)
        for idx in range(len(output1_st)):
            np.testing.assert_allclose(
                output1_st[idx].astype('float32'),
                output2_st[idx].astype('float32'),
                rtol=1e-05,
                atol=0.1,
            )


class TestSGDMultiPrecision1_0(unittest.TestCase):
    def dygraph_sgd_mp(self, mp):
        paddle.disable_static()
        paddle.seed(10)
        paddle.set_device('gpu')
        input = paddle.randn((2, 2))
        model = paddle.nn.Linear(2, 2)
        optimizer = paddle.fluid.optimizer.SGD(
            learning_rate=0.001,
            parameter_list=model.parameters(),
            multi_precision=mp,
        )
        if mp:
            model = paddle.amp.decorate(models=model, level='O2')
            scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

        for idx in range(5):
            if mp:
                with paddle.amp.auto_cast(level='O2'):
                    output = model(input)
                    loss = paddle.mean(output)
                scaled = scaler.scale(loss)
                scaled.backward()
                scaler.minimize(optimizer, scaled)
                optimizer.clear_gradients()
            else:
                output = model(input)
                loss = paddle.mean(output)
                optimizer.minimize(loss)
                optimizer.clear_gradients()

        return output, model.parameters()

    def static_sgd_mp(self, mp):
        paddle.enable_static()
        paddle.seed(10)
        np.random.seed(10)
        exe = paddle.static.Executor('gpu')
        train_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        optimizer = paddle.fluid.optimizer.SGD(
            learning_rate=0.001, multi_precision=mp
        )

        if mp:
            optimizer = paddle.static.amp.decorate(
                optimizer,
                init_loss_scaling=128.0,
                use_dynamic_loss_scaling=True,
                use_pure_fp16=True,
                use_fp16_guard=False,
            )
        with paddle.static.program_guard(train_program, startup_program):
            if mp:
                data = paddle.static.data(
                    shape=[2, 2], name='X', dtype='float16'
                )
            else:
                data = paddle.static.data(
                    shape=[2, 2], name='X', dtype='float32'
                )
            hidden = paddle.static.nn.fc(x=data, size=10)
            loss = paddle.mean(hidden)
            optimizer.minimize(loss)
        exe.run(startup_program)

        if mp:
            optimizer.amp_init(place='gpu', scope=paddle.static.global_scope())
            x = np.random.random(size=(2, 2)).astype('float16')
        else:
            x = np.random.random(size=(2, 2)).astype('float32')
        out = []
        for idx in range(5):
            (loss_data,) = exe.run(
                train_program, feed={"X": x}, fetch_list=[loss.name]
            )
            out.append(loss_data)
        return out

    def test_main(self):
        if not paddle.is_compiled_with_cuda():
            return
        "Test dygraph mode"
        output1_dy, params1_dy = self.dygraph_sgd_mp(mp=True)
        output2_dy, params2_dy = self.dygraph_sgd_mp(mp=False)
        np.testing.assert_allclose(
            output1_dy.astype('float32').numpy(),
            output2_dy.astype('float32').numpy(),
            rtol=1e-05,
            atol=0.1,
        )
        for idx in range(len(params1_dy)):
            np.testing.assert_allclose(
                params1_dy[idx].astype('float32').numpy(),
                params2_dy[idx].astype('float32').numpy(),
                rtol=1e-05,
                atol=0.1,
            )
        "Test static graph mode"
        output1_st = self.static_sgd_mp(mp=True)
        output2_st = self.static_sgd_mp(mp=False)
        for idx in range(len(output1_st)):
            np.testing.assert_allclose(
                output1_st[idx].astype('float32'),
                output2_st[idx].astype('float32'),
                rtol=1e-05,
                atol=0.1,
            )


if __name__ == "__main__":
    unittest.main()
