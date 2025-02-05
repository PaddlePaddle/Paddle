#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from test_fusion_gru_op import TestFusionGRUOp


class TestFusionGRUMKLDNNOp(TestFusionGRUOp):
    def set_confs(self):
        self.use_mkldnn = True
        self.check_pir_onednn = True


class TestFusionGRUMKLDNNOpNoInitial(TestFusionGRUOp):
    def set_confs(self):
        self.with_h0 = False
        self.use_mkldnn = True
        self.check_pir_onednn = True


class TestFusionGRUMKLDNNOpNoBias(TestFusionGRUOp):
    def set_confs(self):
        self.with_bias = False
        self.use_mkldnn = True
        self.check_pir_onednn = True


class TestFusionGRUMKLDNNOpReverse(TestFusionGRUOp):
    def set_confs(self):
        self.is_reverse = True
        self.use_mkldnn = True
        self.check_pir_onednn = True


class TestFusionGRUMKLDNNOpOriginMode(TestFusionGRUOp):
    def set_confs(self):
        self.origin_mode = True
        self.use_mkldnn = True
        self.check_pir_onednn = True


class TestFusionGRUMKLDNNOpMD1(TestFusionGRUOp):
    def set_confs(self):
        self.M = 36
        self.D = 8
        self.use_mkldnn = True
        self.check_pir_onednn = True


class TestFusionGRUMKLDNNOpMD2(TestFusionGRUOp):
    def set_confs(self):
        self.M = 8
        self.D = 8
        self.use_mkldnn = True
        self.check_pir_onednn = True


class TestFusionGRUMKLDNNOpMD3(TestFusionGRUOp):
    def set_confs(self):
        self.M = 17
        self.D = 15
        self.use_mkldnn = True
        self.check_pir_onednn = True


class TestFusionGRUMKLDNNOpBS1(TestFusionGRUOp):
    def set_confs(self):
        self.lod = [[3]]
        self.D = 16
        self.use_mkldnn = True
        self.check_pir_onednn = True


if __name__ == "__main__":
    from paddle import enable_static

    enable_static()
    unittest.main()
