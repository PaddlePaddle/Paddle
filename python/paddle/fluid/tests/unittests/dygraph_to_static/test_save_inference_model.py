#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
from paddle.jit import ProgramTranslator
from paddle.jit.api import declarative
from paddle.jit.dy2static.partial_program import partial_program_from
from paddle.jit.translated_layer import INFER_MODEL_SUFFIX, INFER_PARAMS_SUFFIX

SEED = 2020

np.random.seed(SEED)

place = (
    fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda() else fluid.CPUPlace()
)
program_translator = ProgramTranslator()


class SimpleFcLayer(fluid.dygraph.Layer):
    def __init__(self, fc_size):
        super().__init__()
        self._linear = paddle.nn.Linear(fc_size, fc_size)

    @declarative
    def forward(self, x):
        y = self._linear(x)
        z = self._linear(y)
        out = paddle.mean(z)
        return out, y


class TestDyToStaticSaveInferenceModel(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_inference_model(self):
        fc_size = 20
        x_data = np.random.random((fc_size, fc_size)).astype('float32')
        with fluid.dygraph.guard(place):
            fluid.default_startup_program().random_seed = SEED
            fluid.default_main_program().random_seed = SEED

            x = fluid.dygraph.to_variable(x_data)
            layer = SimpleFcLayer(fc_size)
            adam = fluid.optimizer.SGD(
                learning_rate=0.1, parameter_list=layer.parameters()
            )

            for i in range(5):
                loss, pred = layer(x)
                loss.backward()
                adam.minimize(loss)
                layer.clear_gradients()
            # test for saving model in dygraph.guard
            infer_model_prefix = os.path.join(
                self.temp_dir.name, "test_dy2stat_inference_in_guard/model"
            )
            infer_model_dir = os.path.join(
                self.temp_dir.name, "test_dy2stat_inference_in_guard"
            )
            paddle.jit.save(
                layer=layer,
                path=infer_model_prefix,
                input_spec=[x],
                output_spec=[pred],
            )
            # Check the correctness of the inference
            dygraph_out, _ = layer(x)
        self.check_save_inference_model(layer, [x_data], dygraph_out.numpy())
        self.check_save_inference_model(
            layer, [x_data], dygraph_out.numpy(), fetch=[loss]
        )
        self.check_save_inference_model(
            layer, [x_data], dygraph_out.numpy(), feed=[x]
        )

    def check_save_inference_model(
        self, model, inputs, gt_out, feed=None, fetch=None
    ):

        expected_persistable_vars = set([p.name for p in model.parameters()])

        infer_model_prefix = os.path.join(
            self.temp_dir.name, "test_dy2stat_inference/model"
        )
        infer_model_dir = os.path.join(
            self.temp_dir.name, "test_dy2stat_inference"
        )
        model_filename = "model" + INFER_MODEL_SUFFIX
        params_filename = "model" + INFER_PARAMS_SUFFIX
        paddle.jit.save(
            layer=model,
            path=infer_model_prefix,
            input_spec=feed if feed else None,
            output_spec=fetch if fetch else None,
        )
        # Check the correctness of the inference
        infer_out = self.load_and_run_inference(
            infer_model_dir, model_filename, params_filename, inputs
        )
        np.testing.assert_allclose(gt_out, infer_out, rtol=1e-05)

    def load_and_run_inference(
        self, model_path, model_filename, params_filename, inputs
    ):
        paddle.enable_static()
        exe = fluid.Executor(place)
        [
            inference_program,
            feed_target_names,
            fetch_targets,
        ] = fluid.io.load_inference_model(
            dirname=model_path,
            executor=exe,
            model_filename=model_filename,
            params_filename=params_filename,
        )
        results = exe.run(
            inference_program,
            feed=dict(zip(feed_target_names, inputs)),
            fetch_list=fetch_targets,
        )

        return np.array(results[0])


class TestPartialProgramRaiseError(unittest.TestCase):
    def test_param_type(self):
        program_translator = ProgramTranslator()
        program_translator.enable(True)
        x_data = np.random.random((20, 20)).astype('float32')

        with fluid.dygraph.guard(fluid.CPUPlace()):
            net = SimpleFcLayer(20)
            x = fluid.dygraph.to_variable(x_data)
            out = net(x)

            program_cache = net.forward.program_cache
            _, (concrete_program, _) = program_cache.last()

            params = concrete_program.parameters

            concrete_program.parameters = params[0]
            # TypeError: Type of self._params should be list or tuple,
            # but received <class 'paddle.fluid.framework.ParamBase'>.
            with self.assertRaises(TypeError):
                partial_program_from(concrete_program)

            params[0] = "linear.w.0"
            concrete_program.parameters = params
            # TypeError: Type of self._params[0] should be framework.ParamBase,
            # but received <type 'str'>.
            with self.assertRaises(TypeError):
                partial_program_from(concrete_program)


if __name__ == '__main__':
    unittest.main()
