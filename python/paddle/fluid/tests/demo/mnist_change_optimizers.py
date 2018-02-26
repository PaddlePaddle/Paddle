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

import paddle.v2 as paddle
import paddle.fluid as fluid


class Network(object):
    def __init__(self):
        self.main_program = fluid.Program()
        self.startup_program = fluid.Program()
        self.loss = None
        self.metrics = []
        self.data = []

    def network(self):
        raise NotImplementedError()

    def optimizer(self):
        raise NotImplementedError()

    def build(self):
        if self.loss is not None:
            raise ValueError("build can only be invoked once.")

        with fluid.program_guard(self.main_program, self.startup_program):
            with fluid.unique_name.guard():
                self.network()
                if self.loss is None:
                    raise ValueError(
                        "network method should change self.loss to actual loss")
                optimizer = self.optimizer()
                optimizer.minimize(self.loss)
        return self.main_program, self.startup_program


class MNISTNetwork(Network):
    def __init__(self):
        super(MNISTNetwork, self).__init__()

    def network(self):
        img = fluid.layers.data(name='image', shape=[784])
        self.data.append(img)
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        self.data.append(label)

        hidden = fluid.layers.fc(input=img, size=200, act='tanh')
        prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
        loss = fluid.layers.cross_entropy(input=prediction, label=label)
        avg_loss = fluid.layers.mean(loss)
        acc = fluid.layers.accuracy(input=prediction, label=label)
        self.loss = avg_loss
        self.metrics.append(('accuracy', acc))


class MNISTNetworkWithAdam(MNISTNetwork):
    def __init__(self):
        super(MNISTNetworkWithAdam, self).__init__()

    def optimizer(self):
        return fluid.optimizer.Adam()


class MNISTNetworkWithSGD(MNISTNetwork):
    def __init__(self):
        super(MNISTNetworkWithSGD, self).__init__()

    def optimizer(self):
        return fluid.optimizer.SGD(1e-4)


class Trainer(object):
    def __init__(self, network, place):
        if not isinstance(network, Network):
            raise TypeError()
        network.build()
        self.network = network
        self.exe = fluid.Executor(place)
        self.feeder = fluid.DataFeeder(feed_list=self.network.data, place=place)

    def random_initialize(self):
        self.exe.run(self.network.startup_program)

    def load_params(self, dirname='model'):
        fluid.io.load_params(
            self.exe, main_program=self.network.main_program, dirname=dirname)

    def save_params(self, dirname='model'):
        fluid.io.save_params(
            self.exe, main_program=self.network.main_program, dirname=dirname)

    def train(self,
              reader_creator,
              break_callback=None,
              num_passes=10,
              log_interval=100,
              msg_prefix=''):
        for pass_id in range(num_passes):
            for batch_id, data_batch in enumerate(reader_creator()):
                need_log = (batch_id + 1) % log_interval == 0

                if need_log:
                    fetch_list = [self.network.loss] + [
                        metric[1] for metric in self.network.metrics
                    ]
                else:
                    fetch_list = []

                results = self.exe.run(self.network.main_program,
                                       feed=self.feeder.feed(data_batch),
                                       fetch_list=fetch_list)

                if need_log:
                    msg = msg_prefix + "Pass {0}, Batch {1}, Loss {2:2.4}".format(
                        str(pass_id), str(batch_id), float(results[0]))
                    for i, metric in enumerate(self.network.metrics):
                        msg += ', {0} {1:2.4}'.format(metric[0],
                                                      float(results[i + 1]))
                    print msg

                    if break_callback is not None and break_callback(*results):
                        return


def main():
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=500),
        batch_size=256)
    adam_scope = fluid.core.Scope()
    with fluid.scope_guard(adam_scope):
        train_adam = Trainer(MNISTNetworkWithAdam(), fluid.CUDAPlace(0))
        train_adam.random_initialize()
        train_adam.train(
            reader_creator=train_reader,
            break_callback=lambda loss, acc: acc > 0.95)
        train_adam.save_params()
        print 'Adam train complete'

    del adam_scope

    sgd_scope = fluid.core.Scope()
    with fluid.scope_guard(sgd_scope):
        print 'fine tune by SGD'
        train_sgd = Trainer(MNISTNetworkWithSGD(), fluid.CUDAPlace(0))
        train_sgd.random_initialize()
        train_sgd.load_params()
        train_sgd.train(
            reader_creator=train_reader,
            break_callback=lambda loss, acc: acc > 0.98)
        train_sgd.save_params()
        print 'SGD fine tuning complete'


if __name__ == '__main__':
    main()
