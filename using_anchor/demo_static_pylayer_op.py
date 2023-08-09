import numpy as np

import paddle
import paddle.static as static
from paddle.static.nn import static_pylayer

paddle.enable_static()

def forward_fn(x):
    y = paddle.tanh(x)
    return y

def backward_fn(dy):
    dx = dy * 2
    return dx

train_program = static.Program()
start_program = static.Program()

place = paddle.CPUPlace()
exe = paddle.static.Executor(place)
with static.program_guard(train_program, start_program):
    data = paddle.static.data(name="X", shape=[None, 5], dtype="float32")
    data.stop_gradient = False
    ret = static_pylayer.do_static_pylayer(forward_fn, [data], backward_fn)
    loss = paddle.mean(ret)
    sgd_opt = paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)
    print(static.default_main_program())

exe = paddle.static.Executor(place)
exe.run(start_program)
x = np.random.randn(10, 5).astype(np.float32)
loss, loss_g, x_g, y, y_g = exe.run(train_program, feed={"X":x}, fetch_list = [loss.name, loss.name + '@GRAD', data.name + '@GRAD',
                                                                                   ret.name, ret.name + '@GRAD'])
print("x = ")
print(x)
print("x_g = ")
print(x_g)
print("loss = ")
print(loss)
print("loss_g = ")
print(loss_g)
print("y = ")
print(y)
print("y_g = ")
print(y_g)

# to validate
numpy_ret = np.mean(x)
print("numpy_ret = ")
print(numpy_ret)

np.allclose(y, numpy_ret)