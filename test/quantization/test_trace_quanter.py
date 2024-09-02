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
"""The quantizer layers should be traced by paddle.jit.save function."""
import os
import tempfile
import unittest

import paddle
from paddle.quantization import QAT, QuantConfig
from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver
from paddle.quantization.quanters.abs_max import (
    FakeQuanterWithAbsMaxObserverLayer,
)
from paddle.vision.models import resnet18


class TestPTQ(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir="./")
        self.path = os.path.join(self.temp_dir.name, 'ptq')

    def tearDown(self):
        self.temp_dir.cleanup()

    def _get_model_for_qat(self):
        observer = FakeQuanterWithAbsMaxObserver()
        model = resnet18()
        model.train()
        q_config = QuantConfig(activation=None, weight=None)
        q_config.add_type_config(
            paddle.nn.Conv2D, activation=observer, weight=observer
        )
        qat = QAT(q_config)
        quant_model = qat.quantize(model)
        return quant_model, qat

    def _count_layers(self, model, layer_type):
        count = 0
        for _layer in model.sublayers(True):
            if isinstance(_layer, layer_type):
                count += 1
        return count

    def test_trace(self):
        quant_model, ptq = self._get_model_for_qat()
        image = paddle.rand([1, 3, 32, 32], dtype="float32")
        quantizer_count_in_dygraph = self._count_layers(
            quant_model, FakeQuanterWithAbsMaxObserverLayer
        )
        save_path = os.path.join(self.path, 'int8_infer')
        paddle.jit.save(quant_model, save_path, [image])
        print(f"quant_model is saved into {save_path}")

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
        quantizer_count_in_static_model = 0

        if paddle.base.framework.in_pir_mode():
            for _op in inference_program.global_block().ops:
                if (
                    "fake_quantize_dequantize_moving_average_abs_max"
                    in _op.name()
                ):
                    quantizer_count_in_static_model += 1
        else:
            for _op in inference_program.global_block().ops:
                if (
                    _op.type
                    == "fake_quantize_dequantize_moving_average_abs_max"
                ):
                    quantizer_count_in_static_model += 1
        self.assertEqual(
            quantizer_count_in_dygraph, quantizer_count_in_static_model
        )
        paddle.disable_static()


if __name__ == '__main__':
    unittest.main()
