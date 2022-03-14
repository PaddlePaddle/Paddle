#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import paddle
import paddle.static
from paddle.fluid.tests.unittests.ipu.test_conv_op_ipu import TestBase


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestStride(TestBase):
    def set_op_attrs(self):
        super().set_op_attrs()
        self.attrs['stride'] = [2, 3]


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestDilation(TestBase):
    def set_op_attrs(self):
        super().set_op_attrs()
        self.attrs['dilation'] = [2, 2]


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestGroups(TestBase):
    def set_op_attrs(self):
        super().set_op_attrs()
        self.attrs['groups'] = 3


if __name__ == "__main__":
    unittest.main()
