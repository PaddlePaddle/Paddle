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
import paddle
import paddle.nn as nn


class TestLayerPrint(unittest.TestCase):
    def test_layer_str(self):
        module = nn.ELU(0.2)
        self.assertEqual(str(module), 'ELU(alpha=0.2)')

        module = nn.CELU(0.2)
        self.assertEqual(str(module), 'CELU(alpha=0.2)')

        module = nn.GELU(True)
        self.assertEqual(str(module), 'GELU(approximate=True)')

        module = nn.Hardshrink()
        self.assertEqual(str(module), 'Hardshrink(threshold=0.5)')

        module = nn.Hardswish(name="Hardswish")
        self.assertEqual(str(module), 'Hardswish(name=Hardswish)')

        module = nn.Tanh(name="Tanh")
        self.assertEqual(str(module), 'Tanh(name=Tanh)')

        module = nn.Hardtanh(name="Hardtanh")
        self.assertEqual(
            str(module), 'Hardtanh(min=-1.0, max=1.0, name=Hardtanh)')

        module = nn.PReLU(1, 0.25, name="PReLU", data_format="NCHW")
        self.assertEqual(
            str(module),
            'PReLU(num_parameters=1, data_format=NCHW, init=0.25, dtype=float32, name=PReLU)'
        )

        module = nn.ReLU()
        self.assertEqual(str(module), 'ReLU()')

        module = nn.ReLU6()
        self.assertEqual(str(module), 'ReLU6()')

        module = nn.SELU()
        self.assertEqual(
            str(module),
            'SELU(scale=1.0507009873554805, alpha=1.6732632423543772)')

        module = nn.LeakyReLU()
        self.assertEqual(str(module), 'LeakyReLU(negative_slope=0.01)')

        module = nn.Sigmoid()
        self.assertEqual(str(module), 'Sigmoid()')

        module = nn.Hardsigmoid()
        self.assertEqual(str(module), 'Hardsigmoid()')

        module = nn.Softplus()
        self.assertEqual(str(module), 'Softplus(beta=1, threshold=20)')

        module = nn.Softshrink()
        self.assertEqual(str(module), 'Softshrink(threshold=0.5)')

        module = nn.Softsign()
        self.assertEqual(str(module), 'Softsign()')

        module = nn.Swish()
        self.assertEqual(str(module), 'Swish()')

        module = nn.Tanhshrink()
        self.assertEqual(str(module), 'Tanhshrink()')

        module = nn.ThresholdedReLU()
        self.assertEqual(str(module), 'ThresholdedReLU(threshold=1.0)')

        module = nn.LogSigmoid()
        self.assertEqual(str(module), 'LogSigmoid()')

        module = nn.Softmax()
        self.assertEqual(str(module), 'Softmax(axis=-1)')

        module = nn.LogSoftmax()
        self.assertEqual(str(module), 'LogSoftmax(axis=-1)')

        module = nn.Maxout(groups=2)
        self.assertEqual(str(module), 'Maxout(groups=2, axis=1)')

        module = nn.Linear(2, 4, name='linear')
        self.assertEqual(
            str(module),
            'Linear(in_features=2, out_features=4, dtype=float32, name=linear)')

        module = nn.Upsample(size=[12, 12])
        self.assertEqual(
            str(module),
            'Upsample(size=[12, 12], mode=nearest, align_corners=False, align_mode=0, data_format=NCHW)'
        )

        module = nn.UpsamplingNearest2D(size=[12, 12])
        self.assertEqual(
            str(module), 'UpsamplingNearest2D(size=[12, 12], data_format=NCHW)')

        module = nn.UpsamplingBilinear2D(size=[12, 12])
        self.assertEqual(
            str(module),
            'UpsamplingBilinear2D(size=[12, 12], data_format=NCHW)')

        module = nn.Bilinear(in1_features=5, in2_features=4, out_features=1000)
        self.assertEqual(
            str(module),
            'Bilinear(in1_features=5, in2_features=4, out_features=1000, dtype=float32)'
        )

        module = nn.Dropout(p=0.5)
        self.assertEqual(
            str(module), 'Dropout(p=0.5, axis=None, mode=upscale_in_train)')

        module = nn.Dropout2D(p=0.5)
        self.assertEqual(str(module), 'Dropout2D(p=0.5, data_format=NCHW)')

        module = nn.Dropout3D(p=0.5)
        self.assertEqual(str(module), 'Dropout3D(p=0.5, data_format=NCDHW)')

        module = nn.AlphaDropout(p=0.5)
        self.assertEqual(str(module), 'AlphaDropout(p=0.5)')

        module = nn.Pad1D(padding=[1, 2], mode='constant')
        self.assertEqual(
            str(module),
            'Pad1D(padding=[1, 2], mode=constant, value=0.0, data_format=NCL)')

        module = nn.Pad2D(padding=[1, 0, 1, 2], mode='constant')
        self.assertEqual(
            str(module),
            'Pad2D(padding=[1, 0, 1, 2], mode=constant, value=0.0, data_format=NCHW)'
        )

        module = nn.ZeroPad2D(padding=[1, 0, 1, 2])
        self.assertEqual(
            str(module), 'ZeroPad2D(padding=[1, 0, 1, 2], data_format=NCHW)')

        module = nn.Pad3D(padding=[1, 0, 1, 2, 0, 0], mode='constant')
        self.assertEqual(
            str(module),
            'Pad3D(padding=[1, 0, 1, 2, 0, 0], mode=constant, value=0.0, data_format=NCDHW)'
        )

        module = nn.CosineSimilarity(axis=0)
        self.assertEqual(str(module), 'CosineSimilarity(axis=0, eps=1e-08)')

        module = nn.Embedding(10, 3, sparse=True)
        self.assertEqual(str(module), 'Embedding(10, 3, sparse=True)')

        module = nn.Conv1D(3, 2, 3)
        self.assertEqual(
            str(module), 'Conv1D(3, 2, kernel_size=[3], data_format=NCL)')

        module = nn.Conv1DTranspose(2, 1, 2)
        self.assertEqual(
            str(module),
            'Conv1DTranspose(2, 1, kernel_size=[2], data_format=NCL)')

        module = nn.Conv2D(4, 6, (3, 3))
        self.assertEqual(
            str(module), 'Conv2D(4, 6, kernel_size=[3, 3], data_format=NCHW)')

        module = nn.Conv2DTranspose(4, 6, (3, 3))
        self.assertEqual(
            str(module),
            'Conv2DTranspose(4, 6, kernel_size=[3, 3], data_format=NCHW)')

        module = nn.Conv3D(4, 6, (3, 3, 3))
        self.assertEqual(
            str(module),
            'Conv3D(4, 6, kernel_size=[3, 3, 3], data_format=NCDHW)')

        module = nn.Conv3DTranspose(4, 6, (3, 3, 3))
        self.assertEqual(
            str(module),
            'Conv3DTranspose(4, 6, kernel_size=[3, 3, 3], data_format=NCDHW)')

        module = nn.PairwiseDistance()
        self.assertEqual(str(module), 'PairwiseDistance(p=2.0)')

        module = nn.InstanceNorm1D(2)
        self.assertEqual(
            str(module), 'InstanceNorm1D(num_features=2, epsilon=1e-05)')

        module = nn.InstanceNorm2D(2)
        self.assertEqual(
            str(module), 'InstanceNorm2D(num_features=2, epsilon=1e-05)')

        module = nn.InstanceNorm3D(2)
        self.assertEqual(
            str(module), 'InstanceNorm3D(num_features=2, epsilon=1e-05)')

        module = nn.GroupNorm(num_channels=6, num_groups=6)
        self.assertEqual(
            str(module),
            'GroupNorm(num_groups=6, num_channels=6, epsilon=1e-05)')

        module = nn.LayerNorm([2, 2, 3])
        self.assertEqual(
            str(module), 'LayerNorm(normalized_shape=[2, 2, 3], epsilon=1e-05)')

        module = nn.BatchNorm1D(1)
        self.assertEqual(
            str(module),
            'BatchNorm1D(num_features=1, momentum=0.9, epsilon=1e-05, data_format=NCL)'
        )

        module = nn.BatchNorm2D(1)
        self.assertEqual(
            str(module),
            'BatchNorm2D(num_features=1, momentum=0.9, epsilon=1e-05)')

        module = nn.BatchNorm3D(1)
        self.assertEqual(
            str(module),
            'BatchNorm3D(num_features=1, momentum=0.9, epsilon=1e-05, data_format=NCDHW)'
        )

        module = nn.SyncBatchNorm(2)
        self.assertEqual(
            str(module),
            'SyncBatchNorm(num_features=2, momentum=0.9, epsilon=1e-05)')

        module = nn.LocalResponseNorm(size=5)
        self.assertEqual(
            str(module),
            'LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1.0)')

        module = nn.AvgPool1D(kernel_size=2, stride=2, padding=0)
        self.assertEqual(
            str(module), 'AvgPool1D(kernel_size=2, stride=2, padding=0)')

        module = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.assertEqual(
            str(module), 'AvgPool2D(kernel_size=2, stride=2, padding=0)')

        module = nn.AvgPool3D(kernel_size=2, stride=2, padding=0)
        self.assertEqual(
            str(module), 'AvgPool3D(kernel_size=2, stride=2, padding=0)')

        module = nn.MaxPool1D(kernel_size=2, stride=2, padding=0)
        self.assertEqual(
            str(module), 'MaxPool1D(kernel_size=2, stride=2, padding=0)')

        module = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.assertEqual(
            str(module), 'MaxPool2D(kernel_size=2, stride=2, padding=0)')

        module = nn.MaxPool3D(kernel_size=2, stride=2, padding=0)
        self.assertEqual(
            str(module), 'MaxPool3D(kernel_size=2, stride=2, padding=0)')

        module = nn.AdaptiveAvgPool1D(output_size=16)
        self.assertEqual(str(module), 'AdaptiveAvgPool1D(output_size=16)')

        module = nn.AdaptiveAvgPool2D(output_size=3)
        self.assertEqual(str(module), 'AdaptiveAvgPool2D(output_size=3)')

        module = nn.AdaptiveAvgPool3D(output_size=3)
        self.assertEqual(str(module), 'AdaptiveAvgPool3D(output_size=3)')

        module = nn.AdaptiveMaxPool1D(output_size=16, return_mask=True)
        self.assertEqual(
            str(module), 'AdaptiveMaxPool1D(output_size=16, return_mask=True)')

        module = nn.AdaptiveMaxPool2D(output_size=3, return_mask=True)
        self.assertEqual(
            str(module), 'AdaptiveMaxPool2D(output_size=3, return_mask=True)')

        module = nn.AdaptiveMaxPool3D(output_size=3, return_mask=True)
        self.assertEqual(
            str(module), 'AdaptiveMaxPool3D(output_size=3, return_mask=True)')

        module = nn.SimpleRNNCell(16, 32)
        self.assertEqual(str(module), 'SimpleRNNCell(16, 32)')

        module = nn.LSTMCell(16, 32)
        self.assertEqual(str(module), 'LSTMCell(16, 32)')

        module = nn.GRUCell(16, 32)
        self.assertEqual(str(module), 'GRUCell(16, 32)')

        module = nn.PixelShuffle(3)
        self.assertEqual(str(module), 'PixelShuffle(upscale_factor=3)')

        module = nn.SimpleRNN(16, 32, 2)
        self.assertEqual(
            str(module),
            'SimpleRNN(16, 32, num_layers=2\n  (0): RNN(\n    (cell): SimpleRNNCell(16, 32)\n  )\n  (1): RNN(\n    (cell): SimpleRNNCell(32, 32)\n  )\n)'
        )

        module = nn.LSTM(16, 32, 2)
        self.assertEqual(
            str(module),
            'LSTM(16, 32, num_layers=2\n  (0): RNN(\n    (cell): LSTMCell(16, 32)\n  )\n  (1): RNN(\n    (cell): LSTMCell(32, 32)\n  )\n)'
        )

        module = nn.GRU(16, 32, 2)
        self.assertEqual(
            str(module),
            'GRU(16, 32, num_layers=2\n  (0): RNN(\n    (cell): GRUCell(16, 32)\n  )\n  (1): RNN(\n    (cell): GRUCell(32, 32)\n  )\n)'
        )

        module1 = nn.Sequential(
            ('conv1', nn.Conv2D(1, 20, 5)), ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2D(20, 64, 5)), ('relu2', nn.ReLU()))
        self.assertEqual(
            str(module1),
            'Sequential(\n  '\
            '(conv1): Conv2D(1, 20, kernel_size=[5, 5], data_format=NCHW)\n  '\
            '(relu1): ReLU()\n  '\
            '(conv2): Conv2D(20, 64, kernel_size=[5, 5], data_format=NCHW)\n  '\
            '(relu2): ReLU()\n)'
        )

        module2 = nn.Sequential(
            nn.Conv3DTranspose(4, 6, (3, 3, 3)),
            nn.AvgPool3D(
                kernel_size=2, stride=2, padding=0),
            nn.Tanh(name="Tanh"),
            module1,
            nn.Conv3D(4, 6, (3, 3, 3)),
            nn.MaxPool3D(
                kernel_size=2, stride=2, padding=0),
            nn.GELU(True))
        self.assertEqual(
            str(module2),
            'Sequential(\n  '\
            '(0): Conv3DTranspose(4, 6, kernel_size=[3, 3, 3], data_format=NCDHW)\n  '\
            '(1): AvgPool3D(kernel_size=2, stride=2, padding=0)\n  '\
            '(2): Tanh(name=Tanh)\n  '\
            '(3): Sequential(\n    (conv1): Conv2D(1, 20, kernel_size=[5, 5], data_format=NCHW)\n    (relu1): ReLU()\n'\
            '    (conv2): Conv2D(20, 64, kernel_size=[5, 5], data_format=NCHW)\n    (relu2): ReLU()\n  )\n  '\
            '(4): Conv3D(4, 6, kernel_size=[3, 3, 3], data_format=NCDHW)\n  '\
            '(5): MaxPool3D(kernel_size=2, stride=2, padding=0)\n  '\
            '(6): GELU(approximate=True)\n)'
        )


if __name__ == '__main__':
    unittest.main()
