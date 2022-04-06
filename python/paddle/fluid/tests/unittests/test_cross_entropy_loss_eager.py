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

import test_cross_entropy_loss
from paddle.fluid.framework import _test_eager_guard


class CrossEntropyLossEager(CrossEntropyLoss):
    def test_soft_1d_dygraph_final_state_api(self):
        with _test_eager_guard():
            self.test_cross_entropy_loss_soft_1d()
            self.test_cross_entropy_loss_soft_1d_weight()
            self.test_cross_entropy_loss_soft_1d_mean()
            self.test_cross_entropy_loss_soft_1d_weight_mean()

    # put all testcases in one test will be failed
    def test_soft_2d_dygraph_final_state_api(self):
        with _test_eager_guard():
            self.test_cross_entropy_loss_soft_2d()
            self.test_cross_entropy_loss_soft_2d_weight_mean()

    def test_other_dygraph_final_state_api(self):
        with _test_eager_guard():
            self.test_cross_entropy_loss_1d_with_mean_ignore()
            self.test_cross_entropy_loss_1d_with_mean_ignore_negative()
            self.test_cross_entropy_loss_1d_with_weight_mean_ignore()
            self.test_cross_entropy_loss_1d_with_weight_mean_ignore_exceedlabel(
            )
            self.test_cross_entropy_loss_1d_with_weight_mean()
            self.test_cross_entropy_loss_1d_with_weight_sum()
            self.test_cross_entropy_loss_1d_with_weight_none()
            self.test_cross_entropy_loss_1d_with_weight_none_func()
            self.test_cross_entropy_loss_1d_mean()
            self.test_cross_entropy_loss_1d_sum()
            self.test_cross_entropy_loss_1d_none()
            self.test_cross_entropy_loss_2d_with_weight_none()
            self.test_cross_entropy_loss_2d_with_weight_axis_change_mean()
            self.test_cross_entropy_loss_2d_with_weight_mean_ignore_exceedlabel(
            )
            self.test_cross_entropy_loss_2d_with_weight_mean()
            self.test_cross_entropy_loss_2d_with_weight_sum()
            self.test_cross_entropy_loss_2d_none()
            self.test_cross_entropy_loss_2d_mean()
            self.test_cross_entropy_loss_2d_sum()


if __name__ == "__main__":
    unittest.main()
