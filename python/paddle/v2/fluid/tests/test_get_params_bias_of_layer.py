import paddle.v2.fluid as fluid
import decorators
import unittest
import numpy


class TestGetParamGradFromLayer(unittest.TestCase):
    @decorators.prog_scope()
    def test_fc(self):
        dat = fluid.layers.data("img", shape=[784])
        hidden = fluid.layers.fc(dat, size=100, act='tanh')
        loss = fluid.layers.mean(x=hidden)
        fluid.backward.append_backward(loss)
        cpu = fluid.CPUPlace()
        exe = fluid.Executor(cpu)
        exe.run(fluid.default_startup_program())
        p, b, p_g = exe.run(
            fluid.default_main_program(),
            feed={
                'img': numpy.random.random(size=(100, 784)).astype('float32')
            },
            fetch_list=[hidden.param(), hidden.bias(), hidden.param().grad()])
        self.assertFalse(numpy.isnan(p).any())
        self.assertFalse(numpy.isnan(b).any())
        self.assertFalse(numpy.isnan(p_g).any())


if __name__ == '__main__':
    unittest.main()
