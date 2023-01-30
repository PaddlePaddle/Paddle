# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

<<<<<<< HEAD
import config
import mock_data as mock
import numpy as np
import parameterize

import paddle

=======
import numpy as np
import paddle
import scipy.stats

import config
import mock_data as mock
import parameterize

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
np.random.seed(2022)


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'dist'),
<<<<<<< HEAD
    [
        (
            'test-mock-exp',
            mock.Exponential(
                rate=paddle.rand([100, 200, 99], dtype=config.DEFAULT_DTYPE)
            ),
        )
    ],
)
class TestExponentialFamily(unittest.TestCase):
=======
    [('test-mock-exp',
      mock.Exponential(
          rate=paddle.rand([100, 200, 99], dtype=config.DEFAULT_DTYPE)))])
class TestExponentialFamily(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_entropy(self):
        np.testing.assert_allclose(
            self.dist.entropy(),
            paddle.distribution.ExponentialFamily.entropy(self.dist),
            rtol=config.RTOL.get(config.DEFAULT_DTYPE),
<<<<<<< HEAD
            atol=config.ATOL.get(config.DEFAULT_DTYPE),
        )
=======
            atol=config.ATOL.get(config.DEFAULT_DTYPE))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (config.TEST_CASE_NAME, 'dist'),
<<<<<<< HEAD
    [
        ('test-dummy', mock.DummyExpFamily(0.5, 0.5)),
        (
            'test-dirichlet',
            paddle.distribution.Dirichlet(
                paddle.to_tensor(parameterize.xrand())
            ),
        ),
        (
            'test-beta',
            paddle.distribution.Beta(
                paddle.to_tensor(parameterize.xrand()),
                paddle.to_tensor(parameterize.xrand()),
            ),
        ),
    ],
)
class TestExponentialFamilyException(unittest.TestCase):
=======
    [('test-dummy', mock.DummyExpFamily(0.5, 0.5)),
     ('test-dirichlet',
      paddle.distribution.Dirichlet(paddle.to_tensor(parameterize.xrand()))),
     ('test-beta',
      paddle.distribution.Beta(paddle.to_tensor(parameterize.xrand()),
                               paddle.to_tensor(parameterize.xrand())))])
class TestExponentialFamilyException(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_entropy_exception(self):
        with self.assertRaises(NotImplementedError):
            paddle.distribution.ExponentialFamily.entropy(self.dist)


if __name__ == '__main__':
    unittest.main()
