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
import unittest

import paddle


class TestOptimizerAlias(unittest.TestCase):
    def setUp(self):
        self.api_map = [
            (
                paddle.optimizer.Adadelta,
                paddle.optim.Adadelta,
                paddle.optim.adadelta.Adadelta,
            ),
            (
                paddle.optimizer.Adagrad,
                paddle.optim.Adagrad,
                paddle.optim.adagrad.Adagrad,
            ),
            (paddle.optimizer.Adam, paddle.optim.Adam, paddle.optim.adam.Adam),
            (
                paddle.optimizer.Adamax,
                paddle.optim.Adamax,
                paddle.optim.adamax.Adamax,
            ),
            (
                paddle.optimizer.AdamW,
                paddle.optim.AdamW,
                paddle.optim.adamw.AdamW,
            ),
            (paddle.optimizer.ASGD, paddle.optim.ASGD, paddle.optim.asgd.ASGD),
            (
                paddle.optimizer.LBFGS,
                paddle.optim.LBFGS,
                paddle.optim.lbfgs.LBFGS,
            ),
            (paddle.optimizer.Muon, paddle.optim.Muon, paddle.optim.muon.Muon),
            (
                paddle.optimizer.NAdam,
                paddle.optim.NAdam,
                paddle.optim.nadam.NAdam,
            ),
            (
                paddle.optimizer.Optimizer,
                paddle.optim.Optimizer,
                paddle.optim.optimizer.Optimizer,
            ),
            (
                paddle.optimizer.RAdam,
                paddle.optim.RAdam,
                paddle.optim.radam.RAdam,
            ),
            (
                paddle.optimizer.RMSProp,
                paddle.optim.RMSProp,
                paddle.optim.rmsprop.RMSProp,
            ),
            (
                paddle.optimizer.Rprop,
                paddle.optim.Rprop,
                paddle.optim.rprop.Rprop,
            ),
            (paddle.optimizer.SGD, paddle.optim.SGD, paddle.optim.sgd.SGD),
        ]

    def test_compatibility(self):
        for pairs in self.api_map:
            self.assertTrue(pairs[0] is pairs[1])
            self.assertTrue(pairs[0] is pairs[2])


if __name__ == "__main__":
    unittest.main()
