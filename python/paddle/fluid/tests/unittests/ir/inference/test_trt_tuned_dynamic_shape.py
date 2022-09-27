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

import unittest
import numpy as np
import paddle

paddle.enable_static()
import paddle.fluid as fluid
from paddle.inference import Config, Predictor, create_predictor


class TRTTunedDynamicShapeTest(unittest.TestCase):

    def get_model(self):
        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)

        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            data = fluid.data(name="data",
                              shape=[-1, 6, 64, 64],
                              dtype="float32")
            conv_out = fluid.layers.conv2d(input=data,
                                           num_filters=3,
                                           filter_size=3,
                                           groups=1,
                                           padding=0,
                                           bias_attr=False,
                                           act=None)
        exe.run(startup_program)
        serialized_program = paddle.static.serialize_program(
            data, conv_out, program=main_program)
        serialized_params = paddle.static.serialize_persistables(
            data, conv_out, executor=exe, program=main_program)
        return serialized_program, serialized_params

    def get_config(self, model, params, tuned=False):
        config = Config()
        config.set_model_buffer(model, len(model), params, len(params))
        config.enable_use_gpu(100, 0)
        config.set_optim_cache_dir('tuned_test')
        if tuned:
            config.collect_shape_range_info('shape_range.pbtxt')
        else:
            config.enable_tensorrt_engine(
                workspace_size=1024,
                max_batch_size=1,
                min_subgraph_size=0,
                precision_mode=paddle.inference.PrecisionType.Float32,
                use_static=True,
                use_calib_mode=False)
            config.enable_tuned_tensorrt_dynamic_shape('shape_range.pbtxt',
                                                       True)

        return config

    def predictor_run(self, config, in_data):
        predictor = create_predictor(config)
        in_names = predictor.get_input_names()
        in_handle = predictor.get_input_handle(in_names[0])
        in_handle.copy_from_cpu(in_data)
        predictor.run()

    def test_tuned_dynamic_shape_run(self):
        program, params = self.get_model()

        config = self.get_config(program, params, tuned=True)
        self.predictor_run(config, np.ones((1, 6, 64, 64)).astype(np.float32))

        config2 = self.get_config(program, params, tuned=False)
        self.predictor_run(config2, np.ones((1, 6, 32, 32)).astype(np.float32))


if __name__ == '__main__':
    unittest.main()
