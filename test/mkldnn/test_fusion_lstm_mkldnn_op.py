#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from test_fusion_lstm_op import TestFusionLSTMOp


class TestFusionLSTMONEDNNOp(TestFusionLSTMOp):
    def set_conf(self):
        self.use_mkldnn = True
        self.check_pir_onednn = True

    def test_check_output(self):
        for use_seq in {True, False}:
            self.attrs['use_seq'] = use_seq
            self.check_output(
                check_dygraph=False,
                no_check_set=["Cell"],
                check_pir_onednn=True,
            )


class TestFusionLSTMONEDNNOpReverse(TestFusionLSTMONEDNNOp):
    def set_conf(self):
        self.is_reverse = True
        self.use_mkldnn = True


class TestFusionLSTMONEDNNOpInitReverse(TestFusionLSTMONEDNNOp):
    def set_conf(self):
        self.has_initial_state = True
        self.is_reverse = True
        self.use_mkldnn = True


class TestFusionLSTMONEDNNOpMD1(TestFusionLSTMONEDNNOp):
    def set_conf(self):
        self.M = 36
        self.D = 8
        self.use_mkldnn = True


class TestFusionLSTMONEDNNOpMD2(TestFusionLSTMONEDNNOp):
    def set_conf(self):
        self.M = 8
        self.D = 8
        self.use_mkldnn = True


class TestFusionLSTMONEDNNOpMD3(TestFusionLSTMONEDNNOp):
    def set_conf(self):
        self.M = 15
        self.D = 3
        self.use_mkldnn = True


class TestFusionLSTMONEDNNOpBS1(TestFusionLSTMONEDNNOp):
    def set_conf(self):
        self.lod = [[3]]
        self.D = 16
        self.use_mkldnn = True


class TestFusionLSTMONEDNNOpPeepholesInit(TestFusionLSTMONEDNNOp):
    def set_conf(self):
        self.use_peepholes = True
        self.has_initial_state = True
        self.use_mkldnn = True


if __name__ == '__main__':
    from paddle import enable_static

    enable_static()
    unittest.main()
