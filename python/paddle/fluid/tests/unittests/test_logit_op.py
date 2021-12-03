import unittest
import numpy as np
from op_test import OpTest
import paddle
np.random.seed(10)

def logit(x, eps):
    for i in range(np.size(x)):
        x[i] = max(min(x[i], 1.0 - eps), eps)
        x[i] = np.log(x[i]/( 1.0 - x[i]))
    return x

def logit_grad(x, eps):
    for i in range(np.size(x)):
        if x[i] < eps or x[i] > 1.0 - eps:
            x[i] = 0
        else:
            x[i] = 1.0 / (x[i] * (1.0 - x[i]))
    dout = np.full_like(x, fill_value=1. / x.size)
    dx = dout * x
    return dx

class TestLogitOp(OpTest):
  def setUp(self):
    self.op_type = 'logit'
    self.dtype = 'float64'
    self.shape = [5]
    self.eps = 1e-6
    self.set_attrs()
    x = np.random.uniform(0.1, 1., self.shape).astype(self.dtype)
    out = logit(x, self.eps)
    self.x_grad = logit_grad(x, self.eps)
    self.inputs = {'X': x}
    self.outputs = {'Out': out}
    self.attrs = {'eps': self.eps}

  def set_attrs(self):
    pass

  def test_check_output(self):
    self.check_output()

  def test_check_grad(self):
    self.check_grad(['X'], ['Out'], user_defined_grads=[self.x_grad])

class TestLogitAPI(unittest.TestCase):
  def setUp(self):
    self.x = np.random.uniform(-1., 1., 5).astype(np.float32)
    self.place = paddle.CUDAPlace(0) \
        if paddle.fluid.core.is_compiled_with_cuda() \
        else paddle.CPUPlace()

  def check_api(self, eps=1e-6):
    ref_out = logit(x, eps)
    # test static api
    with paddle.static.program_guard(paddle.static.Program()):
        x = paddle.fluid.data(name='x', shape=self.x_shape)
        y = paddle.logit(x)
        exe = paddle.static.Executor(self.place)
        out = exe.run(feed={'x': self.x}, fetch_list=[y])
    self.assertTrue(np.allclose(out[0], ref_out))
    # test dygrapg api
    paddle.disable_static()
    x = paddle.to_tensor(self.x)
    y = paddle.logit(x)
    self.assertTrue(np.allclose(y.numpy(), ref_out))
    paddle.enable_static()

if __name__ == "__main__":
    unittest.main()
