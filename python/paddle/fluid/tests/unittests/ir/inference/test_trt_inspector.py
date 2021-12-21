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

import sys
import os
import threading
import time
import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.core import PassVersionChecker
from paddle.fluid.core import AnalysisConfig


class LogCapture(object):
    def __init__(self, stream):
        self.escape_char = "\b"
        self.origstream = stream
        self.origstreamfd = self.origstream.fileno()
        self.capturedtext = ""
        self.pipe_out, self.pipe_in = os.pipe()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()

    def start(self):
        self.capturedtext = ""
        self.streamfd = os.dup(self.origstreamfd)
        os.dup2(self.pipe_in, self.origstreamfd)

    def stop(self):
        self.origstream.write(self.escape_char)
        self.origstream.flush()
        self.readOutput()
        os.close(self.pipe_in)
        os.close(self.pipe_out)
        os.dup2(self.streamfd, self.origstreamfd)
        os.close(self.streamfd)

    def readOutput(self):
        while True:
            char = os.read(self.pipe_out, 1).decode(self.origstream.encoding)
            if not char or self.escape_char in char:
                break
            self.capturedtext += char


class TensorRTInspectorTest(InferencePassTest):
    def setUp(self):
        self.set_params()
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name="data", shape=[24, 24], dtype="float32")
            matmul_out = fluid.layers.matmul(
                x=data,
                y=data,
                transpose_x=self.transpose_x,
                transpose_y=self.transpose_y,
                alpha=self.alpha)
            out = fluid.layers.batch_norm(matmul_out, is_test=True)

        self.feeds = {"data": np.ones([24, 24]).astype("float32"), }
        self.enable_trt = True
        self.trt_parameters = InferencePassTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False,
            True)
        self.fetch_list = [out]

    def set_params(self):
        self.transpose_x = True
        self.transpose_y = True
        self.alpha = 2.0

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            capture = LogCapture(sys.stderr)
            capture.start()
            self.check_output_with_option(use_gpu)
            capture.stop()
            lines = capture.capturedtext.splitlines()
            start_line = end_line = -1
            for i in range(len(lines)):
                if 'TensorRTEngine Inspector Entry' in lines[i]:
                    start_line = i
                if 'Bindings:' in lines[i]:
                    end_line = i
                    break
            self.assertTrue(start_line != -1 and end_line != -1)
            self.assertTrue(end_line - start_line > 5)
            self.assertTrue('matmul' in lines[start_line + 2])
            self.assertTrue('LayerType: Scale' in lines[start_line + 3])
            self.assertTrue('batchnorm_add_scale' in lines[start_line + 4])


if __name__ == "__main__":
    unittest.main()
