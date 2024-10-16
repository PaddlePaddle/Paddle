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

import importlib
import os
import tempfile
import unittest
import warnings

import numpy as np

import paddle
from paddle import base
from paddle.base import core, executor
from paddle.base.compiler import CompiledProgram
from paddle.base.framework import Program, program_guard
from paddle.distributed.io import (
    load_inference_model_distributed,
    save_persistables,
)
from paddle.static.io import load_inference_model, save_inference_model

paddle.enable_static()


class InferModel:
    def __init__(self, list):
        self.program = list[0]
        self.feed_var_names = list[1]
        self.fetch_vars = list[2]


class TestBook(unittest.TestCase):
    def test_fit_line_inference_model(self):
        root_path = tempfile.TemporaryDirectory()
        MODEL_DIR = os.path.join(root_path.name, "inference_model")
        UNI_MODEL_DIR = os.path.join(root_path.name, "inference_model1")

        init_program = Program()
        program = Program()

        with program_guard(program, init_program):
            x = paddle.static.data(name='x', shape=[-1, 2], dtype='float32')
            y = paddle.static.data(name='y', shape=[-1, 1], dtype='float32')

            y_predict = paddle.static.nn.fc(x=x, size=1, activation=None)

            cost = paddle.nn.functional.square_error_cost(
                input=y_predict, label=y
            )
            avg_cost = paddle.mean(cost)

            sgd_optimizer = paddle.optimizer.SGD(learning_rate=0.001)
            sgd_optimizer.minimize(avg_cost, init_program)

        place = core.CPUPlace()
        exe = executor.Executor(place)

        exe.run(init_program, feed={}, fetch_list=[])

        for i in range(100):
            tensor_x = np.array([[1, 1], [1, 2], [3, 4], [5, 2]]).astype(
                "float32"
            )
            tensor_y = np.array([[-2], [-3], [-7], [-7]]).astype("float32")

            exe.run(
                program,
                feed={'x': tensor_x, 'y': tensor_y},
                fetch_list=[avg_cost],
            )

        # Separated model and unified model
        save_inference_model(
            MODEL_DIR, [x, y], [avg_cost], exe, program=program
        )
        save_inference_model(
            UNI_MODEL_DIR,
            [x, y],
            [avg_cost],
            exe,
            program=program,
        )
        main_program = program.clone()._prune_with_input(
            feeded_var_names=["x", "y"], targets=[avg_cost]
        )
        params_str = save_persistables(exe, None, main_program, None)

        expected = exe.run(
            program, feed={'x': tensor_x, 'y': tensor_y}, fetch_list=[avg_cost]
        )[0]

        importlib.reload(executor)  # reload to build a new scope

        model_0 = InferModel(load_inference_model(MODEL_DIR, exe))
        with open((UNI_MODEL_DIR + '.pdmodel'), "rb") as f:
            model_str = f.read()
        model_1 = InferModel(load_inference_model(UNI_MODEL_DIR, exe))

        # To be compatible with load_inference_model_distributed function
        tmp_model_filename = MODEL_DIR + '.pdmodel'
        tmp_params_filename = MODEL_DIR + '.pdiparams'
        model_2 = InferModel(
            load_inference_model_distributed(
                root_path.name,
                exe,
                model_filename=tmp_model_filename,
                params_filename=tmp_params_filename,
            )
        )

        model_3 = InferModel(
            load_inference_model_distributed(None, exe, model_str, params_str)
        )

        for model in [model_0, model_1, model_2, model_3]:
            outs = exe.run(
                model.program,
                feed={
                    model.feed_var_names[0]: tensor_x,
                    model.feed_var_names[1]: tensor_y,
                },
                fetch_list=model.fetch_vars,
            )
            actual = outs[0]

            self.assertEqual(model.feed_var_names, ["x", "y"])
            self.assertEqual(len(model.fetch_vars), 1)
            print(f"fetch {model.fetch_vars[0]}")
            self.assertEqual(expected, actual)

        root_path.cleanup()

        self.assertRaises(
            ValueError,
            paddle.static.io.load_inference_model,
            None,
            exe,
            model_filename=model_str,
            params_filename=None,
        )
        self.assertRaises(
            ValueError,
            load_inference_model_distributed,
            None,
            exe,
            model_str,
            None,
        )


class TestSaveInferenceModel(unittest.TestCase):

    def test_save_inference_model(self):
        root_path = tempfile.TemporaryDirectory()
        MODEL_DIR = os.path.join(root_path.name, "inference_model2")
        init_program = paddle.static.Program()
        program = paddle.static.Program()

        # fake program without feed/fetch
        with paddle.static.program_guard(program, init_program):
            x = paddle.static.data(name='x', shape=[-1, 2], dtype='float32')
            y = paddle.static.data(name='y', shape=[-1, 1], dtype='float32')

            y_predict = paddle.static.nn.fc(x, size=1, activation=None)

            cost = paddle.nn.functional.square_error_cost(
                input=y_predict, label=y
            )
            avg_cost = paddle.mean(cost)

        place = core.CPUPlace()
        exe = executor.Executor(place)
        exe.run(init_program, feed={}, fetch_list=[])

        save_inference_model(
            MODEL_DIR, [x, y], [avg_cost], exe, program=program
        )
        root_path.cleanup()

    def test_save_inference_model_with_auc(self):
        root_path = tempfile.TemporaryDirectory()
        MODEL_DIR = os.path.join(root_path.name, "inference_model4")
        init_program = paddle.static.Program()
        program = paddle.static.Program()

        # fake program without feed/fetch
        with paddle.static.program_guard(program, init_program):
            x = paddle.static.data(name='x', shape=[-1, 2], dtype='float32')
            y = paddle.static.data(name='y', shape=[-1, 1], dtype='int32')
            predict = paddle.static.nn.fc(x, size=2, activation='softmax')
            acc = paddle.static.accuracy(input=predict, label=y)
            auc_var, batch_auc_var, auc_states = paddle.static.auc(
                input=predict, label=y
            )
            cost = paddle.nn.functional.cross_entropy(
                input=predict, label=y, reduction='none', use_softmax=False
            )
            avg_cost = paddle.mean(x=cost)

        place = core.CPUPlace()
        exe = executor.Executor(place)
        exe.run(init_program, feed={}, fetch_list=[])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            save_inference_model(
                MODEL_DIR, [x, y], [avg_cost], exe, program=program
            )
            root_path.cleanup()
            expected_warn = "Be sure that you have set auc states to 0 before saving inference model."
            self.assertTrue(len(w) > 0)
            self.assertTrue(expected_warn == str(w[0].message))


class TestInstance(unittest.TestCase):
    #
    def test_save_inference_model(self):
        root_path = tempfile.TemporaryDirectory()
        MODEL_DIR = os.path.join(root_path.name, "inference_model3")
        init_program = paddle.static.Program()
        program = paddle.static.Program()

        # fake program without feed/fetch
        with paddle.static.program_guard(program, init_program):
            x = paddle.static.data(name='x', shape=[-1, 2], dtype='float32')
            y = paddle.static.data(name='y', shape=[-1, 1], dtype='float32')

            y_predict = paddle.static.nn.fc(x, size=1, activation=None)

            cost = paddle.nn.functional.square_error_cost(
                input=y_predict, label=y
            )
            avg_cost = paddle.mean(cost)

        place = core.CPUPlace()
        exe = executor.Executor(place)
        exe.run(init_program, feed={}, fetch_list=[])

        # will print warning message

        cp_prog = CompiledProgram(program)

        save_inference_model(
            MODEL_DIR, [x, y], [avg_cost], exe, program=cp_prog
        )
        self.assertRaises(
            TypeError,
            save_inference_model,
            [MODEL_DIR, [x, y], [avg_cost], [], cp_prog],
        )
        root_path.cleanup()


class TestSaveInferenceModelNew(unittest.TestCase):
    #
    def test_save_and_load_inference_model(self):
        root_path = tempfile.TemporaryDirectory()
        MODEL_DIR = os.path.join(root_path.name, "inference_model5")
        init_program = paddle.static.default_startup_program()
        program = paddle.static.default_main_program()

        # fake program without feed/fetch
        with paddle.static.program_guard(program, init_program):
            x = paddle.static.data(name='x', shape=[-1, 2], dtype='float32')
            y = paddle.static.data(name='y', shape=[-1, 1], dtype='float32')

            y_predict = paddle.static.nn.fc(x, size=1, activation=None)

            cost = paddle.nn.functional.square_error_cost(
                input=y_predict, label=y
            )
            avg_cost = paddle.mean(cost)

            sgd_optimizer = paddle.optimizer.SGD(learning_rate=0.001)
            sgd_optimizer.minimize(avg_cost, init_program)

        place = core.CPUPlace()
        exe = base.Executor(place)
        exe.run(init_program, feed={}, fetch_list=[])

        tensor_x = np.array([[1, 1], [1, 2], [5, 2]]).astype("float32")
        tensor_y = np.array([[-2], [-3], [-7]]).astype("float32")
        for i in range(3):
            exe.run(
                program,
                feed={'x': tensor_x, 'y': tensor_y},
                fetch_list=[avg_cost],
            )

        self.assertRaises(
            ValueError,
            paddle.static.save_inference_model,
            None,
            ['x', 'y'],
            [avg_cost],
            exe,
        )
        self.assertRaises(
            ValueError,
            paddle.static.save_inference_model,
            MODEL_DIR + "/",
            [x, y],
            [avg_cost],
            exe,
        )
        self.assertRaises(
            ValueError,
            paddle.static.save_inference_model,
            MODEL_DIR,
            ['x', 'y'],
            [avg_cost],
            exe,
        )
        self.assertRaises(
            ValueError,
            paddle.static.save_inference_model,
            MODEL_DIR,
            'x',
            [avg_cost],
            exe,
        )
        self.assertRaises(
            ValueError,
            paddle.static.save_inference_model,
            MODEL_DIR,
            [x, y],
            ['avg_cost'],
            exe,
        )
        self.assertRaises(
            ValueError,
            paddle.static.save_inference_model,
            MODEL_DIR,
            [x, y],
            'avg_cost',
            exe,
        )

        if paddle.framework.in_pir_mode():
            MODEL_SUFFIX = ".json"
        else:
            MODEL_SUFFIX = ".pdmodel"

        model_path = MODEL_DIR + "_isdir" + MODEL_SUFFIX
        os.makedirs(model_path)
        self.assertRaises(
            ValueError,
            paddle.static.save_inference_model,
            MODEL_DIR + "_isdir",
            [x, y],
            [avg_cost],
            exe,
        )
        os.rmdir(model_path)

        params_path = MODEL_DIR + "_isdir" + MODEL_SUFFIX
        os.makedirs(params_path)
        self.assertRaises(
            ValueError,
            paddle.static.save_inference_model,
            MODEL_DIR + "_isdir",
            [x, y],
            [avg_cost],
            exe,
        )
        os.rmdir(params_path)

        paddle.static.io.save_inference_model(
            MODEL_DIR, [x, y], [avg_cost], exe
        )

        self.assertTrue(os.path.exists(MODEL_DIR + MODEL_SUFFIX))
        self.assertTrue(os.path.exists(MODEL_DIR + ".pdiparams"))

        expected = exe.run(
            program, feed={'x': tensor_x, 'y': tensor_y}, fetch_list=[avg_cost]
        )[0]

        importlib.reload(executor)  # reload to build a new scope

        self.assertRaises(
            ValueError, paddle.static.load_inference_model, None, exe
        )
        self.assertRaises(
            ValueError, paddle.static.load_inference_model, MODEL_DIR + "/", exe
        )
        self.assertRaises(
            ValueError, paddle.static.load_inference_model, [MODEL_DIR], exe
        )
        self.assertRaises(
            ValueError,
            paddle.static.load_inference_model,
            MODEL_DIR,
            exe,
            pserver_endpoints=None,
        )
        self.assertRaises(
            ValueError,
            paddle.static.load_inference_model,
            MODEL_DIR,
            exe,
            unsupported_param=None,
        )
        self.assertRaises(
            (TypeError, RuntimeError, ValueError),
            paddle.static.load_inference_model,
            None,
            exe,
            model_filename="illegal",
            params_filename="illegal",
        )

        model = InferModel(
            paddle.static.io.load_inference_model(MODEL_DIR, exe)
        )
        root_path.cleanup()

        outs = exe.run(
            model.program,
            feed={
                model.feed_var_names[0]: tensor_x,
                model.feed_var_names[1]: tensor_y,
            },
            fetch_list=model.fetch_vars,
        )
        actual = outs[0]

        self.assertEqual(model.feed_var_names, ["x", "y"])
        self.assertEqual(len(model.fetch_vars), 1)
        self.assertEqual(expected, actual)
        # test save_to_file content type should be bytes
        self.assertRaises(ValueError, paddle.static.io.save_to_file, '', 123)
        # test _get_valid_program
        self.assertRaises(TypeError, paddle.static.io._get_valid_program, 0)
        p = paddle.static.Program()
        cp = CompiledProgram(p)
        paddle.static.io._get_valid_program(cp)
        self.assertTrue(paddle.static.io._get_valid_program(cp) is p)
        cp._program = None
        self.assertRaises(TypeError, paddle.static.io._get_valid_program, cp)

    def test_serialize_program_and_persistables(self):
        init_program = base.default_startup_program()
        program = base.default_main_program()

        # fake program without feed/fetch
        with program_guard(program, init_program):
            x = paddle.static.data(name='x', shape=[-1, 2], dtype='float32')
            y = paddle.static.data(name='y', shape=[-1, 1], dtype='float32')

            y_predict = paddle.static.nn.fc(x, size=1, activation=None)

            cost = paddle.nn.functional.square_error_cost(
                input=y_predict, label=y
            )
            avg_cost = paddle.mean(cost)

            sgd_optimizer = paddle.optimizer.SGD(learning_rate=0.001)
            sgd_optimizer.minimize(avg_cost, init_program)

        place = core.CPUPlace()
        exe = executor.Executor(place)
        exe.run(init_program, feed={}, fetch_list=[])

        tensor_x = np.array([[1, 1], [1, 2], [5, 2]]).astype("float32")
        tensor_y = np.array([[-2], [-3], [-7]]).astype("float32")
        for i in range(3):
            exe.run(
                program,
                feed={'x': tensor_x, 'y': tensor_y},
                fetch_list=[avg_cost],
            )

        # test if return type of serialize_program is bytes
        res1 = paddle.static.io.serialize_program([x, y], [avg_cost])
        self.assertTrue(isinstance(res1, bytes))
        # test if return type of serialize_persistables is bytes
        res2 = paddle.static.io.serialize_persistables([x, y], [avg_cost], exe)
        self.assertTrue(isinstance(res2, bytes))
        # test if variables in program is empty
        res = paddle.static.io._serialize_persistables(Program(), None)
        self.assertIsNone(res)
        self.assertRaises(
            TypeError,
            paddle.static.io.deserialize_persistables,
            None,
            None,
            None,
        )

    def test_normalize_program(self):
        init_program = paddle.static.default_startup_program()
        program = paddle.static.default_main_program()

        # fake program without feed/fetch
        with paddle.static.program_guard(program, init_program):
            x = paddle.static.data(name='x', shape=[-1, 2], dtype='float32')
            y = paddle.static.data(name='y', shape=[-1, 1], dtype='float32')

            y_predict = paddle.static.nn.fc(x, size=1, activation=None)

            cost = paddle.nn.functional.square_error_cost(
                input=y_predict, label=y
            )
            avg_cost = paddle.mean(cost)

            sgd_optimizer = paddle.optimizer.SGD(learning_rate=0.001)
            sgd_optimizer.minimize(avg_cost, init_program)

        place = core.CPUPlace()
        exe = executor.Executor(place)
        exe.run(init_program, feed={}, fetch_list=[])

        tensor_x = np.array([[1, 1], [1, 2], [5, 2]]).astype("float32")
        tensor_y = np.array([[-2], [-3], [-7]]).astype("float32")
        for i in range(3):
            exe.run(
                program,
                feed={'x': tensor_x, 'y': tensor_y},
                fetch_list=[avg_cost],
            )

        # test if return type of serialize_program is bytes
        res = paddle.static.normalize_program(program, [x, y], [avg_cost])
        self.assertTrue(isinstance(res, paddle.static.Program))
        # test program type
        self.assertRaises(
            TypeError, paddle.static.normalize_program, None, [x, y], [avg_cost]
        )
        # test feed_vars type
        self.assertRaises(
            TypeError, paddle.static.normalize_program, program, 'x', [avg_cost]
        )
        # test fetch_vars type
        self.assertRaises(
            TypeError,
            paddle.static.normalize_program,
            program,
            [x, y],
            'avg_cost',
        )


if __name__ == '__main__':
    unittest.main()
