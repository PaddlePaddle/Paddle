# copyright (c) 2023 paddlepaddle authors. all rights reserved.
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

import os
import tempfile
import unittest

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle.nn import Conv2D, Linear, ReLU, Sequential
from paddle.nn.quant.format import LinearDequanter, LinearQuanter
from paddle.quantization import PTQ, QuantConfig
from paddle.quantization.observers import AbsmaxObserver
from paddle.quantization.observers.abs_max import AbsmaxObserverLayer


class LeNetDygraph(paddle.nn.Layer):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.features = Sequential(
            Conv2D(1, 6, 3, stride=1, padding=1),
            ReLU(),
            paddle.nn.MaxPool2D(2, 2),
            Conv2D(6, 16, 5, stride=1, padding=0),
            ReLU(),
            paddle.nn.MaxPool2D(2, 2),
        )

        if num_classes > 0:
            self.fc = Sequential(
                Linear(576, 120), Linear(120, 84), Linear(84, 10)
            )

    def forward(self, inputs):
        x = self.features(inputs)
        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.fc(x)
        out = F.relu(x)
        return out


class TestPTQ(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.temp_dir.name, 'ptq')

    def tearDown(self):
        self.temp_dir.cleanup()

    def _get_model_for_ptq(self):
        observer = AbsmaxObserver(quant_bits=8)
        model = LeNetDygraph()
        model.eval()
        q_config = QuantConfig(activation=observer, weight=observer)
        ptq = PTQ(q_config)
        quant_model = ptq.quantize(model)
        return quant_model, ptq

    def _count_layers(self, model, layer_type):
        count = 0
        for _layer in model.sublayers(True):
            if isinstance(_layer, layer_type):
                count += 1
        return count

    def test_quantize(self):
        ptq_model, _ = self._get_model_for_ptq()
        image = paddle.rand([1, 1, 32, 32], dtype="float32")
        out = ptq_model(image)
        self.assertIsNotNone(out)

        observer_count = self._count_layers(ptq_model, AbsmaxObserverLayer)
        self.assertEqual(observer_count, 14)

    def test_convert(self):
        quant_model, ptq = self._get_model_for_ptq()

        image = paddle.rand([1, 1, 32, 32], dtype="float32")
        out = quant_model(image)
        converted_model = ptq.convert(quant_model)
        out = converted_model(image)
        self.assertIsNotNone(out)

        observer_count = self._count_layers(
            converted_model, AbsmaxObserverLayer
        )
        quanter_count = self._count_layers(converted_model, LinearQuanter)
        dequanter_count = self._count_layers(converted_model, LinearDequanter)
        self.assertEqual(observer_count, 0)
        self.assertEqual(dequanter_count, 14)
        self.assertEqual(quanter_count, 9)

        save_path = os.path.join(self.temp_dir.name, 'int8_infer')
        paddle.jit.save(converted_model, save_path, [image])

        paddle.enable_static()
        exe = paddle.static.Executor(paddle.CPUPlace())
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            [
                inference_program,
                feed_target_names,
                fetch_targets,
            ] = paddle.static.load_inference_model(save_path, exe)
        tensor_img = np.array(
            np.random.random((1, 1, 32, 32)), dtype=np.float32
        )
        results = exe.run(
            inference_program,
            feed={feed_target_names[0]: tensor_img},
            fetch_list=fetch_targets,
        )
        self.assertIsNotNone(results)
        paddle.disable_static()

    def test_convert_2times(self):
        quant_model, ptq = self._get_model_for_ptq()

        image = paddle.rand([1, 1, 32, 32], dtype="float32")
        out = quant_model(image)
        converted_model = ptq.convert(quant_model)
        converted_model = ptq.convert(converted_model)
        out = converted_model(image)
        self.assertIsNotNone(out)

        observer_count = self._count_layers(
            converted_model, AbsmaxObserverLayer
        )
        quanter_count = self._count_layers(converted_model, LinearQuanter)
        dequanter_count = self._count_layers(converted_model, LinearDequanter)
        self.assertEqual(observer_count, 0)
        self.assertEqual(dequanter_count, 14)
        self.assertEqual(quanter_count, 9)

        save_path = os.path.join(self.temp_dir.name, 'int8_infer')
        paddle.jit.save(converted_model, save_path, [image])

        paddle.enable_static()
        exe = paddle.static.Executor(paddle.CPUPlace())
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            [
                inference_program,
                feed_target_names,
                fetch_targets,
            ] = paddle.static.load_inference_model(save_path, exe)
        tensor_img = np.array(
            np.random.random((1, 1, 32, 32)), dtype=np.float32
        )
        results = exe.run(
            inference_program,
            feed={feed_target_names[0]: tensor_img},
            fetch_list=fetch_targets,
        )
        self.assertIsNotNone(results)
        paddle.disable_static()


class TestPTQFP8(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.temp_dir.name, 'ptq')

    def tearDown(self):
        self.temp_dir.cleanup()

    def _get_model_for_ptq(self):
        weight_observer = AbsmaxObserver(quant_bits=(4, 3))
        act_observer = AbsmaxObserver(quant_bits=(5, 2))
        model = LeNetDygraph()
        model.eval()
        q_config = QuantConfig(activation=act_observer, weight=weight_observer)
        ptq = PTQ(q_config)
        quant_model = ptq.quantize(model)
        return quant_model, ptq

    def _count_layers(self, model, layer_type):
        count = 0
        for _layer in model.sublayers(True):
            if isinstance(_layer, layer_type):
                count += 1
        return count

    def test_quantize(self):
        ptq_model, _ = self._get_model_for_ptq()
        image = paddle.rand([1, 1, 32, 32], dtype="float32")
        out = ptq_model(image)
        self.assertIsNotNone(out)

        observer_count = self._count_layers(ptq_model, AbsmaxObserverLayer)
        self.assertEqual(observer_count, 14)

    def test_convert(self):
        quant_model, ptq = self._get_model_for_ptq()

        image = paddle.rand([1, 1, 32, 32], dtype="float32")
        out = quant_model(image)
        converted_model = ptq.convert(quant_model)
        out = converted_model(image)
        self.assertIsNotNone(out)

        observer_count = self._count_layers(
            converted_model, AbsmaxObserverLayer
        )
        quanter_count = self._count_layers(converted_model, LinearQuanter)
        dequanter_count = self._count_layers(converted_model, LinearDequanter)
        self.assertEqual(observer_count, 0)
        self.assertEqual(dequanter_count, 14)
        self.assertEqual(quanter_count, 9)

        save_path = os.path.join(self.temp_dir.name, 'int8_infer')
        paddle.jit.save(converted_model, save_path, [image])

        paddle.enable_static()
        exe = paddle.static.Executor(paddle.CPUPlace())
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            [
                inference_program,
                feed_target_names,
                fetch_targets,
            ] = paddle.static.load_inference_model(save_path, exe)
        tensor_img = np.array(
            np.random.random((1, 1, 32, 32)), dtype=np.float32
        )
        results = exe.run(
            inference_program,
            feed={feed_target_names[0]: tensor_img},
            fetch_list=fetch_targets,
        )
        self.assertIsNotNone(results)
        paddle.disable_static()

    def test_convert_2times(self):
        quant_model, ptq = self._get_model_for_ptq()

        image = paddle.rand([1, 1, 32, 32], dtype="float32")
        out = quant_model(image)
        converted_model = ptq.convert(quant_model)
        converted_model = ptq.convert(converted_model)
        out = converted_model(image)
        self.assertIsNotNone(out)

        observer_count = self._count_layers(
            converted_model, AbsmaxObserverLayer
        )
        quanter_count = self._count_layers(converted_model, LinearQuanter)
        dequanter_count = self._count_layers(converted_model, LinearDequanter)
        self.assertEqual(observer_count, 0)
        self.assertEqual(dequanter_count, 14)
        self.assertEqual(quanter_count, 9)

        save_path = os.path.join(self.temp_dir.name, 'int8_infer')
        paddle.jit.save(converted_model, save_path, [image])

        paddle.enable_static()
        exe = paddle.static.Executor(paddle.CPUPlace())
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            [
                inference_program,
                feed_target_names,
                fetch_targets,
            ] = paddle.static.load_inference_model(save_path, exe)
        tensor_img = np.array(
            np.random.random((1, 1, 32, 32)), dtype=np.float32
        )
        results = exe.run(
            inference_program,
            feed={feed_target_names[0]: tensor_img},
            fetch_list=fetch_targets,
        )
        self.assertIsNotNone(results)
        paddle.disable_static()


if __name__ == '__main__':
    unittest.main()
