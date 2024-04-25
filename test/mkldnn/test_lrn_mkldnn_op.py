#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import unittest

sys.path.append("../deprecated/legacy_test")
from test_lrn_op import TestLRNOp


class TestLRNMKLDNNOp(TestLRNOp):
    def get_attrs(self):
        attrs = TestLRNOp.get_attrs(self)
        attrs['use_mkldnn'] = True
        return attrs

    def test_check_output(self):
        # We cannot validate MidOut as LRN REF has different meaning in it
        # TODO(wangzhongpu): support onednn op in dygraph mode
        self.check_output(
            atol=0.002,
            no_check_set=['MidOut'],
            check_dygraph=False,
            check_pir_onednn=True,
        )

    def test_check_grad_normal(self):
        # TODO(wangzhongpu): support onednn op in dygraph mode
        self.check_grad(
            ['X'], 'Out', max_relative_error=0.01, check_dygraph=False
        )


class TestLRNMKLDNNOpWithIsTest(TestLRNMKLDNNOp):
    def get_attrs(self):
        attrs = TestLRNMKLDNNOp.get_attrs(self)
        attrs['is_test'] = True
        return attrs

    def test_check_grad_normal(self):
        def check_raise_is_test():
            try:
                self.check_grad(
                    ['X'], 'Out', max_relative_error=0.01, check_dygraph=False
                )
            except Exception as e:
                t = "is_test attribute should be set to False in training phase."
                if t in str(e):
                    raise AttributeError

        self.assertRaises(AttributeError, check_raise_is_test)


class TestLRNMKLDNNOpNHWC(TestLRNMKLDNNOp):
    def init_test_case(self):
        self.data_format = 'NHWC'


if __name__ == "__main__":
    from paddle import enable_static

    enable_static()
    unittest.main()
