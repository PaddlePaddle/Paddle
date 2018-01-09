import unittest

import decorators
import paddle.v2.fluid as fluid
import numpy


class ParallelOpTest(unittest.TestCase):
    @decorators.prog_scope()
    def test_fc(self):
        img = fluid.layers.data(name='img', shape=[784])
        hidden = fluid.layers.fc(input=img, size=200, act='tanh')
        hidden = fluid.layers.fc(input=hidden, size=200, act='tanh')
        prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
        loss = fluid.layers.cross_entropy(
            input=prediction,
            label=fluid.layers.data(
                name='label', shape=[1], dtype='int64'))
        loss = fluid.layers.mean(x=loss)
        new_prog, outs = fluid.transpilers.transpile_to_multi_devices(
            fluid.default_main_program(),
            input_vars=['img', 'label'],
            output_vars=[loss])

        with fluid.program_guard(main_program=new_prog):
            loss = fluid.layers.mean(x=outs[0])
            sgd = fluid.optimizer.SGD(learning_rate=1e-3)
            sgd.minimize(loss)

        cpu = fluid.CPUPlace()
        exe = fluid.Executor(cpu)
        exe.run(fluid.default_startup_program())
        img_np = numpy.random.random(size=(32, 784)).astype('float32').reshape(
            (-1, 784))
        label_np = numpy.random.randint(
            low=0, high=9, size=32).astype('int64').reshape((-1, 1))
        loss_np = exe.run(new_prog,
                          feed={'img': img_np,
                                'label': label_np},
                          fetch_list=[loss])[0]
        print loss_np


if __name__ == '__main__':
    unittest.main()
