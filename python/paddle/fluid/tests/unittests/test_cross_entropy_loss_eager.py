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

from __future__ import print_function

from test_cross_entropy_loss import CrossEntropyLoss
from paddle.fluid.framework import _test_eager_guard


class CrossEntropyLossEager(CrossEntropyLoss):
    def test_cross_entropy_loss_soft_1d_eager(self):
        with _test_eager_guard():
            self.test_cross_entropy_loss_soft_1d()

    def test_cross_entropy_loss_soft_1d_weight_eager(self):
        with _test_eager_guard():
            self.test_cross_entropy_loss_soft_1d_weight()

    def test_cross_entropy_loss_soft_1d_mean_eager(self):
        with _test_eager_guard():
            self.test_cross_entropy_loss_soft_1d_mean()

    def test_cross_entropy_loss_soft_1d_weight_mean_eager(self):
        with _test_eager_guard():
            self.test_cross_entropy_loss_soft_1d_weight_mean()

    def test_cross_entropy_loss_soft_2d_eager(self):
        with _test_eager_guard():
            self.test_cross_entropy_loss_soft_2d()

    def test_cross_entropy_loss_soft_2d_weight_mean_eager(self):
        with _test_eager_guard():
            self.test_cross_entropy_loss_soft_2d_weight_mean()

    def test_cross_entropy_loss_1d_with_mean_ignore_eager(self):
        with _test_eager_guard():
            self.test_cross_entropy_loss_1d_with_mean_ignore()

    def test_cross_entropy_loss_1d_with_mean_ignore_negative_eager(self):
        with _test_eager_guard():
            self.test_cross_entropy_loss_1d_with_mean_ignore_negative()

    def test_cross_entropy_loss_1d_with_weight_mean_ignore_eager(self):
        with _test_eager_guard():
            self.test_cross_entropy_loss_1d_with_weight_mean_ignore()

    def test_cross_entropy_loss_1d_with_weight_mean_ignore_exceedlabel_eager(
            self):
        with _test_eager_guard():
            self.test_cross_entropy_loss_1d_with_weight_mean_ignore_exceedlabel(
            )

    def test_cross_entropy_loss_1d_with_weight_mean_eager(self):
        with _test_eager_guard():
            self.test_cross_entropy_loss_1d_with_weight_mean()

    def test_cross_entropy_loss_1d_with_weight_sum_eager(self):
        with _test_eager_guard():
            self.test_cross_entropy_loss_1d_with_weight_sum()

    def test_cross_entropy_loss_1d_with_weight_none_eager(self):
        with _test_eager_guard():
            self.test_cross_entropy_loss_1d_with_weight_none()

    def test_cross_entropy_loss_1d_with_weight_none_func_eager(self):
        with _test_eager_guard():
            self.test_cross_entropy_loss_1d_with_weight_none_func()

    def test_cross_entropy_loss_1d_mean_eager(self):
        with _test_eager_guard():
            self.test_cross_entropy_loss_1d_mean()

    def test_cross_entropy_loss_1d_sum_eager(self):
        with _test_eager_guard():
            self.test_cross_entropy_loss_1d_sum()

    def test_cross_entropy_loss_1d_none_eager(self):
        with _test_eager_guard():
            self.test_cross_entropy_loss_1d_none()

    def test_cross_entropy_loss_2d_with_weight_none_eager(self):
        with _test_eager_guard():
            self.test_cross_entropy_loss_2d_with_weight_none()

    def test_cross_entropy_loss_2d_with_weight_axis_change_mean_eager(self):
        with _test_eager_guard():
            self.test_cross_entropy_loss_2d_with_weight_axis_change_mean()

    def test_cross_entropy_loss_2d_with_weight_mean_ignore_exceedlabel_eager(
            self):
        with _test_eager_guard():
            self.test_cross_entropy_loss_2d_with_weight_mean_ignore_exceedlabel(
            )

    def test_cross_entropy_loss_2d_with_weight_mean_eager(self):
        with _test_eager_guard():
            self.test_cross_entropy_loss_2d_with_weight_mean()

    def test_cross_entropy_loss_2d_with_weight_sum_eager(self):
        with _test_eager_guard():
            self.test_cross_entropy_loss_2d_with_weight_sum()

    def test_cross_entropy_loss_2d_none_eager(self):
        with _test_eager_guard():
            self.test_cross_entropy_loss_2d_none()

    def test_cross_entropy_loss_2d_mean_eager(self):
        with _test_eager_guard():
            self.test_cross_entropy_loss_2d_mean()

    def test_cross_entropy_loss_2d_sum_eager(self):
        with _test_eager_guard():
            self.test_cross_entropy_loss_2d_sum()


if __name__ == "__main__":
    unittest.main()
