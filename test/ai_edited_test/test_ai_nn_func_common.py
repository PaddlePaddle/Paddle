# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

# [AUTO-GENERATED]
# Target: paddle/nn/functional/common.py
# Coverage target: improve coverage for common functional operations (unfold, interpolate,
#   upsample, bilinear, dropout, dropout1d, dropout2d, dropout3d, alpha_dropout,
#   feature_alpha_dropout, pad, cosine_similarity, linear, label_smooth)
"""
Tests for paddle.nn.functional.common module.
测试 paddle.nn.functional.common 模块的单元测试。
"""

import unittest

import numpy as np

import paddle
from paddle.nn import functional as F


class TestUnfold(unittest.TestCase):
    """Tests for unfold function. / unfold 函数的测试。"""

    def test_unfold_basic(self):
        """Test unfold with basic params. / 测试基本参数的 unfold。"""
        x = paddle.randn([2, 3, 4, 4], dtype='float32')
        out = F.unfold(x, kernel_sizes=[3, 3])
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 3 * 3 * 3)

    def test_unfold_with_stride(self):
        """Test unfold with stride. / 测试带步幅的 unfold。"""
        x = paddle.randn([2, 3, 4, 4], dtype='float32')
        out = F.unfold(x, kernel_sizes=[3, 3], strides=[2, 2])
        self.assertIsNotNone(out)

    def test_unfold_with_padding(self):
        """Test unfold with padding. / 测试带填充的 unfold。"""
        x = paddle.randn([2, 3, 4, 4], dtype='float32')
        out = F.unfold(x, kernel_sizes=[3, 3], paddings=[1, 1])
        self.assertIsNotNone(out)

    def test_unfold_with_dilation(self):
        """Test unfold with dilation. / 测试带膨胀的 unfold。"""
        x = paddle.randn([2, 3, 6, 6], dtype='float32')
        out = F.unfold(x, kernel_sizes=[3, 3], dilations=[2, 2])
        self.assertIsNotNone(out)


class TestInterpolate(unittest.TestCase):
    """Tests for interpolate function. / interpolate 函数的测试。"""

    def setUp(self):
        self.x_2d = paddle.randn([2, 3, 4, 4], dtype='float32')
        self.x_3d = paddle.randn([2, 3, 4, 4, 4], dtype='float32')
        self.x_1d = paddle.randn([2, 3, 8], dtype='float32')

    def test_interpolate_nearest_2d(self):
        """Test interpolate with nearest mode on 2D input. / 测试 nearest 模式的二维 interpolate。"""
        out = F.interpolate(self.x_2d, size=[8, 8], mode='nearest')
        self.assertEqual(out.shape, [2, 3, 8, 8])

    def test_interpolate_bilinear(self):
        """Test interpolate with bilinear mode. / 测试 bilinear 模式的 interpolate。"""
        out = F.interpolate(self.x_2d, size=[8, 8], mode='bilinear')
        self.assertEqual(out.shape, [2, 3, 8, 8])

    def test_interpolate_bicubic(self):
        """Test interpolate with bicubic mode. / 测试 bicubic 模式的 interpolate。"""
        out = F.interpolate(self.x_2d, size=[8, 8], mode='bicubic')
        self.assertEqual(out.shape, [2, 3, 8, 8])

    def test_interpolate_trilinear(self):
        """Test interpolate with trilinear mode on 3D input. / 测试 trilinear 模式的三维 interpolate。"""
        out = F.interpolate(self.x_3d, size=[8, 8, 8], mode='trilinear')
        self.assertEqual(out.shape, [2, 3, 8, 8, 8])

    def test_interpolate_linear(self):
        """Test interpolate with linear mode on 1D input. / 测试 linear 模式的一维 interpolate。"""
        out = F.interpolate(self.x_1d, size=[16], mode='linear')
        self.assertEqual(out.shape, [2, 3, 16])

    def test_interpolate_area(self):
        """Test interpolate with area mode. / 测试 area 模式的 interpolate。"""
        out = F.interpolate(self.x_2d, size=[2, 2], mode='area')
        self.assertEqual(out.shape, [2, 3, 2, 2])

    def test_interpolate_scale_factor(self):
        """Test interpolate with scale_factor. / 测试 scale_factor 的 interpolate。"""
        out = F.interpolate(self.x_2d, scale_factor=2.0, mode='nearest')
        self.assertEqual(out.shape, [2, 3, 8, 8])

    def test_interpolate_nearest_1d(self):
        """Test interpolate linear on 1D (NEAREST not supported for 1D). / 测试一维 linear 的 interpolate（NEAREST 不支持一维）。"""
        out = F.interpolate(self.x_1d, size=[16], mode='linear')
        self.assertEqual(out.shape, [2, 3, 16])

    def test_interpolate_align_corners(self):
        """Test interpolate with align_corners. / 测试 align_corners 的 interpolate。"""
        out = F.interpolate(
            self.x_2d, size=[8, 8], mode='bilinear', align_corners=True
        )
        self.assertEqual(out.shape, [2, 3, 8, 8])

    def test_interpolate_nearest_3d(self):
        """Test interpolate nearest on 3D. / 测试三维 nearest 的 interpolate。"""
        out = F.interpolate(self.x_3d, size=[8, 8, 8], mode='nearest')
        self.assertEqual(out.shape, [2, 3, 8, 8, 8])

    def test_interpolate_nearest_5d_error(self):
        """Test interpolate NEAREST raises for non-4D/5D. / 测试 NEAREST 非4D/5D时抛出 ValueError。"""
        with self.assertRaises(ValueError):
            F.interpolate(self.x_1d, size=[16], mode='NEAREST')

    def test_interpolate_invalid_nearest_3d(self):
        """Test interpolate NEAREST raises for 3D tensor. / 测试 NEAREST 三维张量时抛出 ValueError。"""
        x3d = paddle.randn([2, 3, 4], dtype='float32')
        with self.assertRaises(ValueError):
            F.interpolate(x3d, size=[8], mode='NEAREST')

    def test_interpolate_invalid_size_and_scale(self):
        """Test interpolate raises when both size and scale are None. / 测试 size 和 scale 都为 None 时抛出 ValueError。"""
        with self.assertRaises(ValueError):
            F.interpolate(self.x_2d, mode='bilinear')


class TestUpsample(unittest.TestCase):
    """Tests for upsample function. / upsample 函数的测试。"""

    def test_upsample_nearest(self):
        """Test upsample with nearest mode. / 测试 nearest 模式的 upsample。"""
        x = paddle.randn([2, 3, 4, 4], dtype='float32')
        out = F.upsample(x, size=[8, 8], mode='nearest')
        self.assertEqual(out.shape, [2, 3, 8, 8])

    def test_upsample_bilinear(self):
        """Test upsample with bilinear mode. / 测试 bilinear 模式的 upsample。"""
        x = paddle.randn([2, 3, 4, 4], dtype='float32')
        out = F.upsample(x, scale_factor=2.0, mode='bilinear')
        self.assertEqual(out.shape, [2, 3, 8, 8])


class TestBilinear(unittest.TestCase):
    """Tests for bilinear function. / bilinear 函数的测试。"""

    def test_bilinear_basic(self):
        """Test bilinear function. / 测试 bilinear 函数。"""
        x1 = paddle.randn([2, 5], dtype='float32')
        x2 = paddle.randn([2, 4], dtype='float32')
        weight = paddle.randn([6, 5, 4], dtype='float32')
        bias = paddle.randn([1, 6], dtype='float32')
        out = F.bilinear(x1, x2, weight, bias)
        self.assertEqual(out.shape, [2, 6])

    def test_bilinear_no_bias(self):
        """Test bilinear without bias. / 测试无偏置的 bilinear。"""
        x1 = paddle.randn([2, 5], dtype='float32')
        x2 = paddle.randn([2, 4], dtype='float32')
        weight = paddle.randn([6, 5, 4], dtype='float32')
        out = F.bilinear(x1, x2, weight)
        self.assertEqual(out.shape, [2, 6])


class TestDropout(unittest.TestCase):
    """Tests for dropout function. / dropout 函数的测试。"""

    def test_dropout_train(self):
        """Test dropout in training mode. / 测试训练模式的 dropout。"""
        x = paddle.randn([2, 3, 4], dtype='float32')
        out = F.dropout(x, p=0.5, training=True)
        self.assertEqual(out.shape, [2, 3, 4])

    def test_dropout_eval(self):
        """Test dropout in eval mode. / 测试评估模式的 dropout。"""
        x = paddle.randn([2, 3, 4], dtype='float32')
        out = F.dropout(x, p=0.5, training=False)
        np.testing.assert_allclose(out.numpy(), x.numpy(), rtol=1e-6)

    def test_dropout_upscale_in_train(self):
        """Test dropout with upscale_in_train mode. / 测试 upscale_in_train 模式的 dropout。"""
        x = paddle.randn([2, 3, 4], dtype='float32')
        out = F.dropout(x, p=0.5, mode='upscale_in_train', training=True)
        self.assertEqual(out.shape, [2, 3, 4])

    def test_dropout_downscale_in_infer(self):
        """Test dropout with downscale_in_infer mode. / 测试 downscale_in_infer 模式的 dropout。"""
        x = paddle.randn([2, 3, 4], dtype='float32')
        out = F.dropout(x, p=0.5, mode='downscale_in_infer', training=True)
        self.assertEqual(out.shape, [2, 3, 4])


class TestDropoutND(unittest.TestCase):
    """Tests for dropout1d, dropout2d, dropout3d functions. / dropout1d/2d/3d 函数的测试。"""

    def test_dropout1d_train(self):
        """Test dropout1d in training mode. / 测试训练模式的 dropout1d。"""
        x = paddle.randn([2, 3, 8], dtype='float32')
        out = F.dropout1d(x, p=0.5, training=True)
        self.assertEqual(out.shape, [2, 3, 8])

    def test_dropout1d_eval(self):
        """Test dropout1d in eval mode. / 测试评估模式的 dropout1d。"""
        x = paddle.randn([2, 3, 8], dtype='float32')
        out = F.dropout1d(x, p=0.5, training=False)
        np.testing.assert_allclose(out.numpy(), x.numpy(), rtol=1e-6)

    def test_dropout2d_train(self):
        """Test dropout2d in training mode. / 测试训练模式的 dropout2d。"""
        x = paddle.randn([2, 3, 4, 4], dtype='float32')
        out = F.dropout2d(x, p=0.5, training=True)
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_dropout2d_eval(self):
        """Test dropout2d in eval mode. / 测试评估模式的 dropout2d。"""
        x = paddle.randn([2, 3, 4, 4], dtype='float32')
        out = F.dropout2d(x, p=0.5, training=False)
        np.testing.assert_allclose(out.numpy(), x.numpy(), rtol=1e-6)

    def test_dropout3d_train(self):
        """Test dropout3d in training mode. / 测试训练模式的 dropout3d。"""
        x = paddle.randn([2, 3, 4, 4, 4], dtype='float32')
        out = F.dropout3d(x, p=0.5, training=True)
        self.assertEqual(out.shape, [2, 3, 4, 4, 4])

    def test_dropout3d_eval(self):
        """Test dropout3d in eval mode. / 测试评估模式的 dropout3d。"""
        x = paddle.randn([2, 3, 4, 4, 4], dtype='float32')
        out = F.dropout3d(x, p=0.5, training=False)
        np.testing.assert_allclose(out.numpy(), x.numpy(), rtol=1e-6)


class TestAlphaDropout(unittest.TestCase):
    """Tests for alpha_dropout function. / alpha_dropout 函数的测试。"""

    def test_alpha_dropout_train(self):
        """Test alpha_dropout in training mode. / 测试训练模式的 alpha_dropout。"""
        x = paddle.randn([2, 3, 4], dtype='float32')
        out = F.alpha_dropout(x, p=0.5, training=True)
        self.assertEqual(out.shape, [2, 3, 4])

    def test_alpha_dropout_eval(self):
        """Test alpha_dropout in eval mode. / 测试评估模式的 alpha_dropout。"""
        x = paddle.randn([2, 3, 4], dtype='float32')
        out = F.alpha_dropout(x, p=0.5, training=False)
        np.testing.assert_allclose(out.numpy(), x.numpy(), rtol=1e-6)


class TestFeatureAlphaDropout(unittest.TestCase):
    """Tests for feature_alpha_dropout function. / feature_alpha_dropout 函数的测试。"""

    def test_feature_alpha_dropout_train(self):
        """Test feature_alpha_dropout in training mode. / 测试训练模式的 feature_alpha_dropout。"""
        x = paddle.randn([2, 3, 4, 4], dtype='float32')
        out = F.feature_alpha_dropout(x, p=0.5, training=True)
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_feature_alpha_dropout_eval(self):
        """Test feature_alpha_dropout in eval mode. / 测试评估模式的 feature_alpha_dropout。"""
        x = paddle.randn([2, 3, 4, 4], dtype='float32')
        out = F.feature_alpha_dropout(x, p=0.5, training=False)
        np.testing.assert_allclose(out.numpy(), x.numpy(), rtol=1e-6)


class TestPad(unittest.TestCase):
    """Tests for pad function. / pad 函数的测试。"""

    def test_pad_constant(self):
        """Test pad with constant mode. / 测试 constant 模式的 pad。"""
        x = paddle.randn([2, 3, 4, 4], dtype='float32')
        out = F.pad(x, [1, 1, 1, 1], mode='constant', value=0.0)
        self.assertEqual(out.shape, [2, 3, 6, 6])

    def test_pad_reflect(self):
        """Test pad with reflect mode. / 测试 reflect 模式的 pad。"""
        x = paddle.randn([2, 3, 4, 4], dtype='float32')
        out = F.pad(x, [1, 1, 1, 1], mode='reflect')
        self.assertEqual(out.shape, [2, 3, 6, 6])

    def test_pad_replicate(self):
        """Test pad with replicate mode. / 测试 replicate 模式的 pad。"""
        x = paddle.randn([2, 3, 4, 4], dtype='float32')
        out = F.pad(x, [1, 1, 1, 1], mode='replicate')
        self.assertEqual(out.shape, [2, 3, 6, 6])

    def test_pad_circular(self):
        """Test pad with circular mode. / 测试 circular 模式的 pad。"""
        x = paddle.randn([2, 3, 4, 4], dtype='float32')
        out = F.pad(x, [1, 1, 1, 1], mode='circular')
        self.assertEqual(out.shape, [2, 3, 6, 6])

    def test_pad_1d(self):
        """Test pad with 1D padding. / 测试一维 padding 的 pad。"""
        x = paddle.randn([2, 3, 4], dtype='float32')
        out = F.pad(x, [1, 1])
        self.assertEqual(out.shape, [2, 3, 6])

    def test_pad_3d(self):
        """Test pad with 3D padding. / 测试三维 padding 的 pad。"""
        x = paddle.randn([2, 3, 4, 4, 4], dtype='float32')
        out = F.pad(x, [1, 1, 1, 1, 1, 1], mode='constant')
        self.assertEqual(out.shape, [2, 3, 6, 6, 6])


class TestCosineSimilarity(unittest.TestCase):
    """Tests for cosine_similarity function. / cosine_similarity 函数的测试。"""

    def test_cosine_similarity_basic(self):
        """Test cosine_similarity with basic params. / 测试基本参数的 cosine_similarity。"""
        x1 = paddle.randn([2, 4], dtype='float32')
        x2 = paddle.randn([2, 4], dtype='float32')
        out = F.cosine_similarity(x1, x2, axis=1)
        self.assertEqual(out.shape, [2])

    def test_cosine_similarity_dim0(self):
        """Test cosine_similarity with axis=0. / 测试 axis=0 的 cosine_similarity。"""
        x1 = paddle.randn([3, 4], dtype='float32')
        x2 = paddle.randn([3, 4], dtype='float32')
        out = F.cosine_similarity(x1, x2, axis=0)
        self.assertEqual(out.shape, [4])

    def test_cosine_similarity_epsilon(self):
        """Test cosine_similarity with eps. / 测试带 eps 的 cosine_similarity。"""
        x1 = paddle.randn([2, 4], dtype='float32')
        x2 = paddle.randn([2, 4], dtype='float32')
        out = F.cosine_similarity(x1, x2, axis=1, eps=1e-8)
        self.assertEqual(out.shape, [2])


class TestLinear(unittest.TestCase):
    """Tests for linear function. / linear 函数的测试。"""

    def test_linear_basic(self):
        """Test linear with basic params. / 测试基本参数的 linear。"""
        x = paddle.randn([2, 8], dtype='float32')
        weight = paddle.randn([16, 8], dtype='float32')
        bias = paddle.randn([16], dtype='float32')
        out = F.linear(x, weight.T, bias)
        self.assertEqual(out.shape, [2, 16])

    def test_linear_no_bias(self):
        """Test linear without bias. / 测试无偏置的 linear。"""
        x = paddle.randn([2, 8], dtype='float32')
        weight = paddle.randn([16, 8], dtype='float32')
        out = F.linear(x, weight.T)
        self.assertEqual(out.shape, [2, 16])

    def test_linear_name(self):
        """Test linear with name parameter. / 测试 name 参数的 linear。"""
        x = paddle.randn([2, 8], dtype='float32')
        weight = paddle.randn([16, 8], dtype='float32')
        out = F.linear(x, weight.T, name='test_linear')
        self.assertEqual(out.shape, [2, 16])


class TestLabelSmooth(unittest.TestCase):
    """Tests for label_smooth function. / label_smooth 函数的测试。"""

    def test_label_smooth_basic(self):
        """Test label_smooth with basic params (float32 label). / 测试基本参数（float32标签）的 label_smooth。"""
        label = paddle.to_tensor([0.0, 1.0, 2.0, 3.0], dtype='float32')
        out = F.label_smooth(label, epsilon=0.1)
        self.assertEqual(out.shape, [4])

    def test_label_smooth_one_hot(self):
        """Test label_smooth with prior_dist (soft label). / 测试 prior_dist（软标签）的 label_smooth。"""
        label = paddle.to_tensor([[1, 0, 0], [0, 1, 0]], dtype='float32')
        out = F.label_smooth(label, epsilon=0.1)
        self.assertEqual(out.shape, [2, 3])


class TestFold(unittest.TestCase):
    """Tests for fold function. / fold 函数的测试。"""

    def test_fold_basic(self):
        """Test fold with basic params. / 测试基本参数的 fold。"""
        x = paddle.randn([2, 3 * 3 * 3, 4], dtype='float32')
        out = F.fold(x, output_sizes=[4, 4], kernel_sizes=[3, 3])
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_fold_with_stride(self):
        """Test fold with stride. / 测试带步幅的 fold。"""
        x = paddle.randn([2, 3 * 3 * 3, 4], dtype='float32')
        out = F.fold(
            x, output_sizes=[4, 4], kernel_sizes=[3, 3], strides=[1, 1]
        )
        self.assertEqual(out.shape, [2, 3, 4, 4])


if __name__ == '__main__':
    unittest.main()
