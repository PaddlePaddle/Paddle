import paddle.v2.fluid.profiler as profiler
import paddle.v2.fluid.layers as layers
import numpy as np

place = core.GPUPlace(0)
exe = Executor(place)

epoc = 8
dshape = [4, 3, 28, 28]
data = layers.data(name='data', shape=dshape, dtype='float32')
conv = layers.conv2d(data, 20, 3, stride=[1, 1], padding=[1, 1])

input = core.LoDTensor()
with profiler("cuda_profiler.txt") as nvprof:
    for i in range(epoc):
        input.set(np.random.random(dshape).astype("float32"), place)
        exe.run(framework.default_main_program(), feed={'data': data})
