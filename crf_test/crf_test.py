import paddle.fluid as fluid
import numpy as np
cc=np.random.rand(12,10).astype('float32')
train_program = fluid.Program()
startup_program = fluid.Program()

with fluid.program_guard(train_program, startup_program):
    emission = fluid.layers.data(name='emission', shape=[10], dtype='float32', lod_level=1)
    target = fluid.layers.data(name='target', shape=[1], dtype='int', lod_level=1)
    emi_length = fluid.layers.data(name='EmissionLength', shape=[1], dtype='int', lod_level=1)
    label_length = fluid.layers.data(name='LabelLength', shape=[1], dtype='int', lod_level=1)
    crf_cost = fluid.layers.linear_chain_crf(
    input=emission,
    label=target,
    EmissionLength=emi_length,
    LabelLength=label_length,
    param_attr=fluid.ParamAttr(
        name='crfw',
        learning_rate=0.01))
startup_program.random_seed=1
train_program.random_seed=1

use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(startup_program)

a = fluid.create_lod_tensor(cc, [[3,3,4,2]], place)
b = fluid.create_lod_tensor(np.array([[1],[1],[2],[3],[1],[1],[1],[3],[1],[1],[1],[1]]),[[3,3,4,2]] , place)
c = fluid.create_lod_tensor(np.array([[1],[1],[2],[3],[1],[1],[1],[3],[5],[1],[5],[5]]),[[3,3,4,2]] , place)
d = fluid.create_lod_tensor(np.array([[1],[1],[2],[3],[1],[1],[1],[3],[1],[1],[1],[1]]),[[3,3,4,2]] , place)

feed1 = {'emission':a,
        'target':b,
        'EmissionLength':c,
        'LabelLength':d}
loss = exe.run(train_program,feed=feed1, fetch_list=[crf_cost])
print(loss)