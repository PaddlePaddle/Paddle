#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
import unittest
import numpy as np
import paddle.v2.fluid as fluid
import paddle.v2.fluid.profiler as profiler
import paddle.v2.fluid.layers as layers
import os


class TestProfiler(unittest.TestCase):
    def test_nvprof(self):
        if not fluid.core.is_compile_gpu():
            return
        epoc = 8
        dshape = [4, 3, 28, 28]
        data = layers.data(name='data', shape=[3, 28, 28], dtype='float32')
        conv = layers.conv2d(data, 20, 3, stride=[1, 1], padding=[1, 1])

        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        output_file = 'cuda_profiler.txt'
        with profiler.cuda_profiler(output_file, 'csv') as nvprof:
            for i in range(epoc):
                input = np.random.random(dshape).astype('float32')
                exe.run(fluid.default_main_program(), feed={'data': input})
        os.remove(output_file)


if __name__ == '__main__':
    unittest.main()
