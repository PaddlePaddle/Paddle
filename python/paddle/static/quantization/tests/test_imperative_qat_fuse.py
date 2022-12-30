#   copyright (c) 2018 paddlepaddle authors. all rights reserved.
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.

import os
import unittest

from test_imperative_qat import TestImperativeQat

import paddle
from paddle.framework import core, set_flags

paddle.enable_static()

os.environ["CPU_NUM"] = "1"
if core.is_compiled_with_cuda():
    set_flags({"FLAGS_cudnn_deterministic": True})


class TestImperativeQatfuseBN(TestImperativeQat):
    def set_vars(self):
        self.weight_quantize_type = 'abs_max'
        self.activation_quantize_type = 'moving_average_abs_max'
        self.diff_threshold = 0.03125
        self.onnx_format = False
        self.fuse_conv_bn = True


if __name__ == '__main__':
    unittest.main()
