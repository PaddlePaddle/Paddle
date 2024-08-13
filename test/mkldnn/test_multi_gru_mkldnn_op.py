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

import unittest

import numpy as np
from op_test import OpTest
from test_fusion_gru_op import ACTIVATION, fusion_gru


def multi_gru(
    x,  # T x M
    lod,  # 1 x N
    h0,  # N x D
    wx,  # M x 3D
    wh,  # D x 3D
    bias,  # 1 x 3D
    origin_mode,
    layers,
):
    act_state = ACTIVATION['tanh']
    act_gate = ACTIVATION['sigmoid']
    input = x
    for i in range(0, layers * 2, 2):
        _, _, _, gru1_out = fusion_gru(
            input,
            lod,
            h0[i],
            wx[i],
            wh[i],
            bias[i],
            False,
            origin_mode,
            act_state,
            act_gate,
        )
        _, _, _, gru2_out = fusion_gru(
            input,
            lod,
            h0[i + 1],
            wx[i + 1],
            wh[i + 1],
            bias[i + 1],
            True,
            origin_mode,
            act_state,
            act_gate,
        )
        input = np.concatenate((gru1_out, gru2_out), axis=1)
    return input


class TestMultiGruMkldnnOp(OpTest):
    def set_confs(self):
        pass

    def set_dtype(self):
        pass

    def set_force_fp32_output(self):
        pass

    def setUp(self):
        self.op_type = "multi_gru"
        self.lod = [[2, 4, 3]]
        self.ICs = [3]
        self.OCs = [5]
        self.with_bias = True
        self.layers = 1
        self.origin_mode = False
        self._cpu_only = True
        self.error_margin = 1e-5
        self.set_confs()
        self.dtype = "float32"
        self.set_dtype()
        self.force_fp32_output = False
        self.set_force_fp32_output()

        is_int8 = self.dtype == 'int8'
        scale_data = 63
        shift_data = 64

        T = sum(self.lod[0])
        N = len(self.lod[0])

        self.inputs = {}
        if is_int8:
            x_f32 = np.random.rand(T, self.ICs[0]).astype('float32') * 2 - 1
            x_u8 = np.rint(x_f32 * scale_data + shift_data).astype(np.uint8)
            self.inputs['X'] = (x_u8, self.lod)

        else:
            x_f32 = np.random.rand(T, self.ICs[0]).astype('float32')
            self.inputs['X'] = (x_f32, self.lod)

        wx = []
        wh = []
        bias = []
        h0 = []

        for layer in range(self.layers):
            IC = self.ICs[layer]
            OC = self.OCs[layer]
            for j in range(2):
                wx.append(np.random.rand(IC, 3 * OC).astype('float32'))
                wh.append(np.random.rand(OC, 3 * OC).astype('float32'))
                bias.append(
                    np.random.rand(1, 3 * OC).astype('float32')
                    if self.with_bias
                    else np.zeros((1, 3 * OC), dtype='float32')
                )
                h0.append(np.zeros((N, OC), dtype='float32'))

        self.inputs['WeightX'] = [
            ('wx' + str(i), wx[i]) for i in range(self.layers * 2)
        ]
        self.inputs['WeightH'] = [
            ('wh' + str(i), wh[i]) for i in range(self.layers * 2)
        ]
        if self.with_bias:
            self.inputs['Bias'] = [
                ('b' + str(i), bias[i]) for i in range(self.layers * 2)
            ]

        if is_int8:
            s8_max = 127.0
            scale_weights = []
            for layer in range(self.layers):
                OC = self.OCs[layer]
                for j in range(2):
                    scale_ur = s8_max / np.max(
                        np.abs(
                            np.concatenate(
                                [
                                    wx[2 * layer + j][:, : 2 * OC],
                                    wh[2 * layer + j]
                                    .flatten()[: 2 * OC * OC]
                                    .reshape(OC, 2 * OC),
                                ],
                                axis=0,
                            )
                        ),
                        axis=0,
                    )
                    scale_o = s8_max / np.max(
                        np.abs(
                            np.concatenate(
                                [
                                    wx[2 * layer + j][:, 2 * OC :],
                                    wh[2 * layer + j]
                                    .flatten()[2 * OC * OC :]
                                    .reshape(OC, OC),
                                ],
                                axis=0,
                            )
                        ),
                        axis=0,
                    )

                    scale_weights.append(
                        np.concatenate([scale_ur, scale_o]).astype('float32')
                    )
            self.inputs['Scale_weights'] = [
                ('w_scale' + str(i), scale_weights[i])
                for i in range(self.layers * 2)
            ]
            self.error_margin = 1e-1 if self.force_fp32_output else 1

        hidden_f32 = multi_gru(
            x_f32, self.lod, h0, wx, wh, bias, self.origin_mode, self.layers
        )

        if self.dtype == 'float32' or self.force_fp32_output:
            self.outputs = {'Hidden': (hidden_f32, self.lod)}
        else:
            hidden_u8 = np.rint(hidden_f32 * scale_data + shift_data).astype(
                np.uint8
            )
            self.outputs = {'Hidden': (hidden_u8, self.lod)}

        self.attrs = {
            'activation': 'tanh',
            'gate_activation': 'sigmoid',
            'layers': self.layers,
            'origin_mode': self.origin_mode,
            'use_mkldnn': True,
        }

        if is_int8:
            self.attrs['force_fp32_output'] = self.force_fp32_output
            self.attrs['Scale_data'] = scale_data
            self.attrs['Shift_data'] = shift_data

    def test_check_output(self):
        self.check_output(
            check_dygraph=False, atol=self.error_margin, check_pir_onednn=True
        )


class TestMultiGruMkldnnOpNoBias(TestMultiGruMkldnnOp):
    def set_confs(self):
        self.with_bias = False


class TestMultiGruMkldnnOpLayers2(TestMultiGruMkldnnOp):
    def set_confs(self):
        self.layers = 2
        self.ICs = [2, 6]
        self.OCs = [3, 8]


class TestMultiGruMkldnnOpLayers3(TestMultiGruMkldnnOp):
    def set_confs(self):
        self.layers = 3
        self.ICs = [2, 6, 12]
        self.OCs = [3, 6, 14]


class TestMultiGruMkldnnOpOriginMode(TestMultiGruMkldnnOp):
    def set_confs(self):
        self.origin_mode = True


class TestMultiGruMkldnnInt8Op(TestMultiGruMkldnnOp):
    def set_dtype(self):
        self.dtype = 'int8'


class TestMultiGruMkldnnInt8OpForceFP32Output(TestMultiGruMkldnnInt8Op):
    def set_force_fp32_output(self):
        self.force_fp32_output = True


class TestMultiGruMkldnnInt8OpNoBias(TestMultiGruMkldnnOpNoBias):
    def set_dtype(self):
        self.dtype = 'int8'


class TestMultiGruMkldnnInt8OpNoBiasForceFP32Output(
    TestMultiGruMkldnnInt8OpNoBias
):
    def set_force_fp32_output(self):
        self.force_fp32_output = True


class TestMultiGruMkldnnInt8OpLayers2(TestMultiGruMkldnnOpLayers2):
    def set_dtype(self):
        self.dtype = 'int8'


class TestMultiGruMkldnnInt8OpLayers2ForceFP32Output(
    TestMultiGruMkldnnInt8OpLayers2
):
    def set_force_fp32_output(self):
        self.force_fp32_output = True


class TestMultiGruMkldnnInt8OpLayers3(TestMultiGruMkldnnOpLayers3):
    def set_dtype(self):
        self.dtype = 'int8'


class TestMultiGruMkldnnInt8OpLayers3ForceFP32Output(
    TestMultiGruMkldnnInt8OpLayers3
):
    def set_force_fp32_output(self):
        self.force_fp32_output = True


class TestMultiGruMkldnnInt8OpOriginMode(TestMultiGruMkldnnOpOriginMode):
    def set_dtype(self):
        self.dtype = 'int8'


class TestMultiGruMkldnnInt8OpOriginModeForceFP32Output(
    TestMultiGruMkldnnInt8OpOriginMode
):
    def set_force_fp32_output(self):
        self.force_fp32_output = True


if __name__ == "__main__":
    unittest.main()
