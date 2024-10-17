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


def fused_adam_step(inputs, attributes, num):
    '''
    Simulate one step of the fused_adam optimizer
    :param inputs: dict of inputs
    :param attributes: dict of attributes
    :return tuple: tuple of output params, moments1, moments2, moments2_max, beta1_pows, beta2_pows
    '''
    params = inputs['Params']
    grads = inputs['Grads']
    moments1 = inputs['Moments1']
    moments2 = inputs['Moments2']
    moments2_max = inputs['Moments2Max']
    lr = inputs['LearningRate']
    beta1_pows = inputs['Beta1Pows']
    beta2_pows = inputs['Beta2Pows']

    params_out = []
    moments1_out = []
    moments2_out = []
    moments2_max_out = []
    beta1_pows_out = []
    beta2_pows_out = []

    epsilon = attributes['epsilon']

    if 'beta1' in attributes:
        beta1 = attributes['beta1']
    else:
        beta1 = inputs['Beta1Tensor'][0][0]
    if 'beta2' in attributes:
        beta2 = attributes['beta2']
    else:
        beta2 = inputs['Beta2Tensor'][0][0]

    amsgrad = attributes['amsgrad']

    for i in range(num):
        _moment1_out = beta1 * moments1[i][1] + (1 - beta1) * grads[i][1]
        _moment2_out = beta2 * moments2[i][1] + (1 - beta2) * np.square(
            grads[i][1]
        )

        moments1_out.append(_moment1_out)
        moments2_out.append(_moment2_out)

        lr_t = lr * np.sqrt(1 - beta2_pows[i][1]) / (1 - beta1_pows[i][1])

        if amsgrad:
            _moment2_max = np.maximum(_moment2_out, moments2_max[i][1])
            moments2_max_out.append(_moment2_max)

            params_out.append(
                params[i][1]
                - lr_t
                * (moments1_out[i] / (np.sqrt(moments2_max_out[i]) + epsilon))
            )
        else:
            _moment2_max = np.empty_like(_moment2_out)
            moments2_max_out.append(_moment2_max)

            params_out.append(
                params[i][1]
                - lr_t
                * (moments1_out[i] / (np.sqrt(moments2_out[i]) + epsilon))
            )

    for i in range(num):
        beta1_pows_out.append(beta1_pows[i][1] * beta1)
        beta2_pows_out.append(beta2_pows[i][1] * beta2)

    return (
        params_out,
        moments1_out,
        moments2_out,
        moments2_max_out,
        beta1_pows_out,
        beta2_pows_out,
    )


class TestFusedAdamOp(OpTest):
    def set_amsgrad(self):
        self.amsgrad = False
        # no check `Moment2MaxOut` with amsgrad is False
        self.no_check_set = ['Moments2MaxOut']

    def setUp(self):
        paddle.enable_static()

        '''Test FusedAdam Op with supplied attributes'''
        self.__class__.op_type = "fused_adam"

        num = 10
        inputs_list = [[0] * num] * 6
        learning_rate = 0.004
        beta1 = 0.78
        beta2 = 0.836
        epsilon = 1e-4
        beta1_pow = beta1**10
        beta2_pow = beta2**10
        self.set_amsgrad()

        self.attrs = {
            'epsilon': epsilon,
            'beta1': beta1,
            'beta2': beta2,
            "chunk_size": 32 * 2048,
            "amsgrad": self.amsgrad,
        }

        for i in range(num):
            inputs_list[0][i] = np.random.uniform(-1, 1, (102, 105)).astype(
                "float32"
            )
            inputs_list[1][i] = np.random.uniform(-1, 1, (102, 105)).astype(
                "float32"
            )
            inputs_list[2][i] = np.random.uniform(-1, 1, (102, 105)).astype(
                "float32"
            )
            inputs_list[3][i] = np.random.random((102, 105)).astype("float32")
            inputs_list[4][i] = np.array([beta1_pow]).astype("float32")
            inputs_list[5][i] = np.array([beta2_pow]).astype("float32")

        self.inputs = {
            'Params': [
                ("params" + str(i), inputs_list[0][i]) for i in range(num)
            ],
            'Grads': [
                ("grads" + str(i), inputs_list[1][i]) for i in range(num)
            ],
            'Moments1': [
                ("moments1" + str(i), inputs_list[2][i]) for i in range(num)
            ],
            'Moments2': [
                ("moments2" + str(i), inputs_list[3][i]) for i in range(num)
            ],
            'Moments2Max': [
                ("moments2_max" + str(i), np.zeros_like(inputs_list[0][i]))
                for i in range(num)
            ],
            'LearningRate': np.array([learning_rate]).astype("float32"),
            'Beta1Pows': [
                ("beta1_pows" + str(i), inputs_list[4][i]) for i in range(num)
            ],
            'Beta2Pows': [
                ("beta2_pows" + str(i), inputs_list[5][i]) for i in range(num)
            ],
        }

        (
            params_out,
            moments1_out,
            moments2_out,
            moments2_max_out,
            beta1_pows_out,
            beta2_pows_out,
        ) = fused_adam_step(self.inputs, self.attrs, num)

        self.outputs = {
            'Moments1Out': [
                ("moments1_out" + str(i), moments1_out[i]) for i in range(num)
            ],
            'Moments2Out': [
                ("moments2_out" + str(i), moments2_out[i]) for i in range(num)
            ],
            'Moments2MaxOut': [
                ("moments2_max_out" + str(i), moments2_max_out[i])
                for i in range(num)
            ],
            'ParamsOut': [
                ("params_out" + str(i), params_out[i]) for i in range(num)
            ],
            'Beta1PowsOut': [
                ("beta1_pows_out" + str(i), beta1_pows_out[i])
                for i in range(num)
            ],
            'Beta2PowsOut': [
                ("beta2_pows_out" + str(i), beta2_pows_out[i])
                for i in range(num)
            ],
        }

    def test_check_output(self):
        paddle.enable_static()
        if paddle.is_compiled_with_cuda():
            self.check_output(
                no_check_set=self.no_check_set, check_dygraph=False
            )


class TestFusedAdamOpAMSGrad(TestFusedAdamOp):
    def set_amsgrad(self):
        self.amsgrad = True
        self.no_check_set = None


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
