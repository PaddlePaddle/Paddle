import numpy
import numpy as np
import paddle.fluid as fluid

a = fluid.layers.data(name="a", shape=[100], dtype='float32')

a1 = fluid.layers.fc(input=a, size=500, act=None, bias_attr=False)


cpu = fluid.core.CPUPlace()
exe = fluid.Executor(cpu)

with open('startup_program.pb', 'wb') as f:
    f.write(fluid.default_startup_program().desc.serialize_to_string())
exe.run(fluid.default_startup_program())

data_1 = np.array(numpy.random.random([100, 100]), dtype='float32')

with open('main_program.pb', 'wb') as f:
    f.write(fluid.default_main_program().desc.serialize_to_string())

outs = exe.run(feed={'a':data_1, }, fetch_list=[a1.name])

fluid.io.save_inference_model("./model2", [a.name], [a1], exe)

print(numpy.array(outs))

