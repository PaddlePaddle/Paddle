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

from auto_scan_test import PassAutoScanTest
from program_config import TensorConfig, ProgramConfig, OpConfig
import numpy as np
from functools import partial
import unittest
import hypothesis.strategies as st


class TestConvConcatActivationMkldnnFusePass(PassAutoScanTest):

    def sample_program_config(self, draw):
        data_format = draw(st.sampled_from(['NCHW', 'NHWC']))
        dilations = draw(st.sampled_from([[2, 2]]))
        padding_algorithm = draw(st.sampled_from(['VALID']))
        groups = draw(st.sampled_from([4]))
        paddings = draw(st.sampled_from([[0, 3]]))
        strides = draw(st.sampled_from([[1, 2]]))
        axis = draw(st.sampled_from([0]))
        activation_type = draw(
            st.sampled_from([
                'relu', 'gelu', 'swish', 'mish', 'sqrt', 'hard_swish',
                'sigmoid', 'abs', 'relu6', 'clip', 'tanh', 'hard_sigmoid',
                'leaky_relu'
            ]))

        def generate_data(input_type):
            if input_type == 'NCHW':
                return np.random.random([16, 48, 64, 64]).astype(np.float32)
            elif input_type == 'NHWC':
                return np.random.random([16, 64, 64, 48]).astype(np.float32)
            elif input_type == 'weights':
                return np.random.random([16, int(48 / groups), 3,
                                         3]).astype(np.float32)

        conv2d_op1 = OpConfig(type='conv2d',
                              inputs={
                                  'Input': ['conv_input_1'],
                                  'Filter': ['conv_weights_1']
                              },
                              outputs={'Output': ['conv_output_1']},
                              attrs={
                                  'data_format': data_format,
                                  'dilations': dilations,
                                  'padding_algorithm': padding_algorithm,
                                  'groups': groups,
                                  'paddings': paddings,
                                  'strides': strides
                              })

        conv2d_op2 = OpConfig(type='conv2d',
                              inputs={
                                  'Input': ['conv_input_2'],
                                  'Filter': ['conv_weights_2']
                              },
                              outputs={'Output': ['conv_output_2']},
                              attrs={
                                  'data_format': data_format,
                                  'dilations': dilations,
                                  'padding_algorithm': padding_algorithm,
                                  'groups': groups,
                                  'paddings': paddings,
                                  'strides': strides
                              })

        concat_op = OpConfig(type='concat',
                             inputs={'X': ['conv_output_1', 'conv_output_2']},
                             outputs={'Out': ['concat_output']},
                             attrs={'axis': axis})

        if activation_type == 'relu6':
            activation_op = OpConfig(activation_type,
                                     inputs={'X': ['concat_output']},
                                     outputs={'Out': ['activation_output']},
                                     threshold=draw(
                                         st.floats(min_value=1.0,
                                                   max_value=10.0)))
        elif activation_type == 'leaky_relu':
            activation_op = OpConfig(activation_type,
                                     inputs={'X': ['concat_output']},
                                     outputs={'Out': ['activation_output']},
                                     alpha=draw(
                                         st.floats(min_value=0.1,
                                                   max_value=1.0)))
        elif activation_type == 'swish':
            activation_op = OpConfig(activation_type,
                                     inputs={'X': ['concat_output']},
                                     outputs={'Out': ['activation_output']},
                                     beta=draw(
                                         st.floats(min_value=0.1,
                                                   max_value=1.0)))
        elif activation_type == 'clip':
            activation_op = OpConfig(
                activation_type,
                inputs={'X': ['concat_output']},
                outputs={'Out': ['activation_output']},
                min=draw(st.floats(min_value=0.1, max_value=0.49)),
                max=draw(st.floats(min_value=0.5, max_value=1.0)))
        else:
            activation_op = OpConfig(activation_type,
                                     inputs={'X': ['concat_output']},
                                     outputs={'Out': ['activation_output']})

        model_net = [conv2d_op1, conv2d_op2, concat_op, activation_op]

        program_config = ProgramConfig(
            ops=model_net,
            inputs={
                'conv_input_1':
                TensorConfig(data_gen=partial(generate_data, data_format)),
                'conv_input_2':
                TensorConfig(data_gen=partial(generate_data, data_format))
            },
            weights={
                'conv_weights_1':
                TensorConfig(data_gen=partial(generate_data, 'weights')),
                'conv_weights_2':
                TensorConfig(data_gen=partial(generate_data, 'weights'))
            },
            outputs=['activation_output'])

        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_mkldnn=True)
        yield config, ['conv2d', 'conv2d', 'concat'], (1e-5, 1e-5)

    def test(self):
        self.run_and_statis(quant=False,
                            passes=['conv_activation_mkldnn_fuse_pass'])


if __name__ == '__main__':
    unittest.main()
