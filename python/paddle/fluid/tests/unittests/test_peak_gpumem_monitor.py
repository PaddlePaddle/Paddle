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

from __future__ import print_function

import unittest
import os
os.environ['FLAGS_benchmark'] = 'True'

import numpy
import paddle.fluid.core as core
from paddle.fluid.executor import Executor
from paddle.fluid.layers import mul, data


class TestPeakMemoryMonitoring(unittest.TestCase):
    def test_mul(self):

        a = data(name='a', shape=[784], dtype='float32')
        b = data(
            name='b',
            shape=[784, 100],
            dtype='float32',
            append_batch_size=False)
        out = mul(x=a, y=b)

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)

            a_np = numpy.random.random((100, 784)).astype('float32')
            b_np = numpy.random.random((784, 100)).astype('float32')
            self.assertEqual(0, core.get_mem_usage(0))
            exe = Executor(place)
            outs = exe.run(feed={'a': a_np, 'b': b_np}, fetch_list=[out])
            out = outs[0]
            #disable this assert since ctest will ignore the os.environ setting 
            #self.assertGreater(core.get_mem_usage(0), 0)

            raised = False
            try:
                core.print_mem_usage()
            except:
                raised = True
            self.assertFalse(raised, 'Exception raised')


if __name__ == '__main__':
    unittest.main()
