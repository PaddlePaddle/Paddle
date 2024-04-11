# Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

import os
import shutil
import unittest

import numpy as np

from paddle.cinn.auto_schedule.cost_model import CostModel


class TestCostModel(unittest.TestCase):
    def test_cost_model_init(self):
        with self.assertRaises(ValueError):
            cost_model = CostModel(2)

        cost_model = CostModel()

    def test_basic_functions(self):
        samples = [np.random.randn(5, 6) for i in range(16)]
        labels = [1.0] * 16

        cost_model = CostModel()
        cost_model.train(samples, labels)
        pred = cost_model.predict(samples)

        path = "./test_cost_model.save_model"
        cost_model.save(path)

        load_cost_model = CostModel()
        load_cost_model.load(path)

        load_pred = load_cost_model.predict(samples)

        if os.path.exists(path):
            if os.path.isfile(path):
                os.remove(path)
            else:
                shutil.rmtree(path)

        np.testing.assert_almost_equal(pred, load_pred)


if __name__ == "__main__":
    unittest.main()
