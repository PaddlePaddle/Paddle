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
