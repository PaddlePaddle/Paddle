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

import tempfile
import unittest

from paddle.fluid.tests.unittests.ipu.test_save_load_ipu import TestBase, IPUOpTest


@unittest.skipIf(IPUOpTest.use_ipumodel(), "skip for ipumodel")
class TestSGDFP16(TestBase):
    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['steps'] = 100
        self.attrs['save_at_step'] = 20
        self.attrs['is_training'] = True
        self.attrs['opt_type'] = 'sgd'
        self.attrs['enable_fp16'] = True
        self.attrs['model_path'] = tempfile.TemporaryDirectory()


@unittest.skipIf(IPUOpTest.use_ipumodel(), "skip for ipumodel")
class TestAdamFP16(TestBase):
    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['steps'] = 100
        self.attrs['save_at_step'] = 20
        self.attrs['is_training'] = True
        self.attrs['opt_type'] = 'adam'
        self.attrs['enable_fp16'] = True
        self.attrs['model_path'] = tempfile.TemporaryDirectory()


@unittest.skipIf(IPUOpTest.use_ipumodel(), "skip for ipumodel")
class TestLambFP16(TestBase):
    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['steps'] = 100
        self.attrs['save_at_step'] = 20
        self.attrs['is_training'] = True
        self.attrs['opt_type'] = 'lamb'
        self.attrs['enable_fp16'] = True
        self.attrs['model_path'] = tempfile.TemporaryDirectory()


if __name__ == "__main__":
    unittest.main()
