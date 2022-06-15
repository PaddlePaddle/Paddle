# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

paddle.enable_static()
import numpy as np
import paddle.fluid as fluid
from paddle.inference import Config, create_predictor


def get_sample_model():
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.program_guard(main_program, startup_program):
        data = fluid.data(name="data", shape=[-1, 6, 64, 64], dtype="float32")
        conv_out = fluid.layers.conv2d(input=data,
                                       num_filters=3,
                                       filter_size=3,
                                       groups=1,
                                       padding=0,
                                       bias_attr=False,
                                       act=None)
    exe.run(startup_program)
    serialized_program = paddle.static.serialize_program(data,
                                                         conv_out,
                                                         program=main_program)
    serialized_params = paddle.static.serialize_persistables(
        data, conv_out, executor=exe, program=main_program)
    return serialized_program, serialized_params


class TestInferenceMkldnnAPI(unittest.TestCase):

    def get_config(self, model, params):
        config = Config()
        config.set_model_buffer(model, len(model), params, len(params))
        config.enable_mkldnn()
        config.enable_mkldnn_fc_passes()
        return config

    def test_apis(self):
        program, params = get_sample_model()
        config = self.get_config(program, params)
        predictor = create_predictor(config)
        in_names = predictor.get_input_names()
        in_handle = predictor.get_input_handle(in_names[0])
        in_data = np.ones((1, 6, 32, 32)).astype(np.float32)
        in_handle.copy_from_cpu(in_data)
        predictor.run()


if __name__ == '__main__':
    unittest.main()
