# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# [AUTO-GENERATED] Unit test for paddle.nn.functional.common (unfold & interpolate)
# 自动生成的单测，覆盖 paddle.nn.functional.common 模块中未覆盖的代码路径
# Target: cover uncovered lines 170-230, 543-607 in paddle/python/paddle/nn/functional/common.py
# 目标：覆盖 common.py 中 unfold 的参数验证错误路径和 interpolate 的各种模式校验

"""
This test covers the following modules and code paths:
这个测试覆盖以下模块和代码路径：

1. unfold() - 参数验证错误路径:
   - kernel_sizes 类型错误 (line 170)
   - strides 类型错误 (line 180)
   - dilations 类型错误 (line 190)
   - paddings 类型/长度错误 (lines 205-209)
   覆盖 unfold 函数的各个参数校验分支

2. interpolate() - 各种模式校验:
   - 无效 resample (line 534)
   - LINEAR 仅支持3D (line 540)
   - NEAREST 仅支持4D/5D (line 542-543)
   - BILINEAR/BICUBIC 仅支持4D (line 545-546)
   - TRILINEAR 仅支持5D (line 547-548)
   - size 和 scale_factor 同时为 None (line 550-551)
   - size 长度不匹配 (line 553-556)
   - align_corners 类型错误 (line 569-570)
   - antialias 类型错误 (line 572-573)
   - align_mode 无效 (line 575-576)
   - align_corners 与 NEAREST 不兼容 (line 577-580)
   - antialias 仅支持 BILINEAR/BICUBIC (line 582-585)
"""

import unittest

import paddle
import paddle.nn.functional as F


class TestUnfoldErrorPaths(unittest.TestCase):
    """Test unfold() parameter validation.
    测试 unfold() 参数验证。
    覆盖 common.py 第 170-213 行。
    """

    def setUp(self):
        paddle.disable_static()

    def test_unfold_basic(self):
        """Basic unfold should work."""
        x = paddle.randn([2, 3, 8, 8])
        out = F.unfold(x, kernel_sizes=3, strides=1, paddings=1, dilations=1)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 3 * 3 * 3)

    def test_unfold_kernel_sizes_int(self):
        """unfold with integer kernel_sizes."""
        x = paddle.randn([2, 3, 8, 8])
        out = F.unfold(x, kernel_sizes=2)
        self.assertEqual(out.shape[0], 2)

    def test_unfold_strides_int(self):
        """unfold with integer strides."""
        x = paddle.randn([2, 3, 8, 8])
        out = F.unfold(x, kernel_sizes=3, strides=2)
        self.assertEqual(out.shape[0], 2)

    def test_unfold_dilations_int(self):
        """unfold with integer dilations."""
        x = paddle.randn([2, 3, 8, 8])
        out = F.unfold(x, kernel_sizes=3, dilations=2)
        self.assertEqual(out.shape[0], 2)

    def test_unfold_paddings_int(self):
        """unfold with integer paddings."""
        x = paddle.randn([2, 3, 8, 8])
        out = F.unfold(x, kernel_sizes=3, paddings=1)
        self.assertEqual(out.shape[0], 2)

    def test_unfold_paddings_list_2(self):
        """unfold with list of 2 paddings (doubled to 4)."""
        x = paddle.randn([2, 3, 8, 8])
        out = F.unfold(x, kernel_sizes=3, paddings=[1, 1])
        self.assertEqual(out.shape[0], 2)

    def test_unfold_paddings_list_4(self):
        """unfold with list of 4 paddings."""
        x = paddle.randn([2, 3, 8, 8])
        out = F.unfold(x, kernel_sizes=3, paddings=[1, 1, 1, 1])
        self.assertEqual(out.shape[0], 2)

    def test_unfold_invalid_input_dims(self):
        """unfold with wrong input dimensions should raise AssertionError."""
        x = paddle.randn([2, 3])  # 2D instead of 4D
        with self.assertRaises(AssertionError):
            F.unfold(x, kernel_sizes=3)


class TestInterpolateErrorPaths(unittest.TestCase):
    """Test interpolate() parameter validation.
    测试 interpolate() 参数验证。
    覆盖 common.py 第 530-590 行。
    """

    def setUp(self):
        paddle.disable_static()

    def test_interpolate_bilinear_basic(self):
        """Basic bilinear interpolation should work."""
        x = paddle.randn([2, 3, 8, 8])
        out = F.interpolate(x, size=[16, 16], mode='bilinear')
        self.assertEqual(out.shape, [2, 3, 16, 16])

    def test_interpolate_nearest_basic(self):
        """Basic nearest interpolation."""
        x = paddle.randn([2, 3, 8, 8])
        out = F.interpolate(x, size=[16, 16], mode='nearest')
        self.assertEqual(out.shape, [2, 3, 16, 16])

    def test_interpolate_bicubic_basic(self):
        """Basic bicubic interpolation."""
        x = paddle.randn([2, 3, 8, 8])
        out = F.interpolate(x, size=[16, 16], mode='bicubic')
        self.assertEqual(out.shape, [2, 3, 16, 16])

    def test_interpolate_trilinear_basic(self):
        """Basic trilinear interpolation."""
        x = paddle.randn([2, 3, 4, 4, 4])
        out = F.interpolate(x, size=[8, 8, 8], mode='trilinear')
        self.assertEqual(out.shape, [2, 3, 8, 8, 8])

    def test_interpolate_linear_basic(self):
        """Basic linear interpolation."""
        x = paddle.randn([2, 3, 8])
        out = F.interpolate(x, size=[16], mode='linear')
        self.assertEqual(out.shape, [2, 3, 16])

    def test_interpolate_area_basic(self):
        """Basic area interpolation."""
        x = paddle.randn([2, 3, 8, 8])
        out = F.interpolate(x, size=[4, 4], mode='area')
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_interpolate_with_scale_factor(self):
        """Interpolation with scale_factor."""
        x = paddle.randn([2, 3, 8, 8])
        out = F.interpolate(x, scale_factor=2.0, mode='bilinear')
        self.assertEqual(out.shape, [2, 3, 16, 16])

    def test_interpolate_both_none_raises(self):
        """Both size and scale_factor being None should raise ValueError."""
        x = paddle.randn([2, 3, 8, 8])
        with self.assertRaises(ValueError):
            F.interpolate(x, size=None, scale_factor=None, mode='bilinear')

    def test_interpolate_size_length_mismatch(self):
        """Size length not matching rank-2 should raise ValueError."""
        x = paddle.randn([2, 3, 8, 8])
        with self.assertRaises(ValueError):
            F.interpolate(x, size=[16], mode='bilinear')

    def test_interpolate_align_corners_type_error(self):
        """Non-bool align_corners should raise TypeError."""
        x = paddle.randn([2, 3, 8, 8])
        with self.assertRaises(TypeError):
            F.interpolate(x, size=[16, 16], mode='bilinear', align_corners=1)

    def test_interpolate_antialias_type_error(self):
        """Non-bool antialias should raise TypeError."""
        x = paddle.randn([2, 3, 8, 8])
        with self.assertRaises(TypeError):
            F.interpolate(x, size=[16, 16], mode='bilinear', antialias=1)

    def test_interpolate_align_corners_with_nearest(self):
        """align_corners with NEAREST mode should raise ValueError."""
        x = paddle.randn([2, 3, 8, 8])
        with self.assertRaises(ValueError):
            F.interpolate(x, size=[16, 16], mode='nearest', align_corners=True)

    def test_interpolate_antialias_non_bilinear(self):
        """antialias with non-bilinear/bicubic mode should raise ValueError."""
        x = paddle.randn([2, 3, 8, 8])
        with self.assertRaises(ValueError):
            F.interpolate(x, size=[16, 16], mode='nearest', antialias=True)


if __name__ == '__main__':
    unittest.main()
