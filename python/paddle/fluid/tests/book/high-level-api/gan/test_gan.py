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

import errno
import math
import os

import matplotlib
import numpy

import paddle
import paddle.fluid as fluid

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

NOISE_SIZE = 100
NUM_PASS = 1000
NUM_REAL_IMGS_IN_BATCH = 121
NUM_TRAIN_TIMES_OF_DG = 3
LEARNING_RATE = 2e-5


def train_discriminator():
    noise = fluid.layers.data(name='noise', shape=[NOISE_SIZE], dtype='float32')
    real_img = fluid.layers.data(
        name='img', shape=[784], dtype='float32', stop_gradient=False)
    label = fluid.layers.data(name='label', shape=[1], dtype='float32')
    hidden = fluid.layers.fc(input=noise,
                             size=200,
                             act='relu',
                             param_attr='G.w1',
                             bias_attr='G.b1')
    g_img = fluid.layers.fc(input=hidden,
                            size=28 * 28,
                            act='tanh',
                            param_attr='G.w2',
                            bias_attr='G.b2')
    img = fluid.layers.concat(input=[real_img, g_img], axis=0)
    hidden = fluid.layers.fc(input=img,
                             size=200,
                             act='relu',
                             param_attr='D.w1',
                             bias_attr='D.b1')
    logit = fluid.layers.fc(input=hidden,
                            size=1,
                            act=None,
                            param_attr='D.w2',
                            bias_attr='D.b2')
    dg_loss = fluid.layers.mean(
        fluid.layers.sigmoid_cross_entropy_with_logits(
            x=logit, label=label))
    opt = fluid.optimizer.Adam(learning_rate=LEARNING_RATE)
    opt.minimize(loss=dg_loss, parameter_list=['D.w1', 'D.b1', 'D.w2', 'D.b2'])

    return dg_loss


def train_generator():
    noise = fluid.layers.data(name='noise', shape=[NOISE_SIZE], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='float32')
    hidden = fluid.layers.fc(input=noise,
                             size=200,
                             act='relu',
                             param_attr='G.w1',
                             bias_attr='G.b1')
    g_img = fluid.layers.fc(input=hidden,
                            size=28 * 28,
                            act='tanh',
                            param_attr='G.w2',
                            bias_attr='G.b2')
    hidden = fluid.layers.fc(input=g_img,
                             size=200,
                             act='relu',
                             param_attr='D.w1',
                             bias_attr='D.b1')
    logit = fluid.layers.fc(input=hidden,
                            size=1,
                            act=None,
                            param_attr='D.w2',
                            bias_attr='D.b2')
    dg_loss = fluid.layers.mean(
        fluid.layers.sigmoid_cross_entropy_with_logits(
            x=logit, label=label))
    opt = fluid.optimizer.Adam(learning_rate=LEARNING_RATE)
    opt.minimize(loss=dg_loss, parameter_list=['G.w1', 'G.b1', 'G.w2', 'G.b2'])

    return dg_loss


class GANTrainer(object):
    def __init__(self, place):
        startup_program = fluid.Program()
        self.d_program = fluid.Program()
        self.g_program = fluid.Program()

        with fluid.program_guard(self.d_program, startup_program):
            self.d_loss = train_discriminator()

        with fluid.program_guard(self.g_program, startup_program):
            self.g_loss = train_generator()

        self.exe = fluid.Executor(place)
        self.exe.run(startup_program)

    def train(self):
        num_true = NUM_REAL_IMGS_IN_BATCH
        train_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.mnist.train(), buf_size=60000),
            batch_size=num_true)

        for pass_id in range(NUM_PASS):
            for batch_id, data in enumerate(train_reader()):
                num_true = len(data)
                n = numpy.random.uniform(
                    low=-1.0, high=1.0,
                    size=[num_true * NOISE_SIZE]).astype('float32').reshape(
                        [num_true, NOISE_SIZE])
                print(self.exe.run(self.g_program,
                                   feed={
                                       'noise': n,
                                       'label': numpy.ones(
                                           shape=[num_true, 1], dtype='float32')
                                   },
                                   fetch_list=[self.g_loss]))

                real_data = numpy.array(map(lambda x: x[0], data)).astype(
                    'float32')
                real_data = real_data.reshape(num_true, 784)
                total_label = numpy.concatenate([
                    numpy.ones(
                        shape=[real_data.shape[0], 1], dtype='float32'),
                    numpy.zeros(
                        shape=[real_data.shape[0], 1], dtype='float32')
                ])
                print(self.exe.run(
                    self.d_program,
                    feed={'img': real_data,
                          'label': total_label,
                          'noise': n},
                    fetch_list=[self.d_loss]))
                # # generate image each batch
            # fig = plot(generated_img)
            # plt.savefig(
            #     'out/{0}.png'.format(str(pass_id).zfill(3)), bbox_inches='tight')
            # plt.close(fig)
            # pass


def main():
    trainer = GANTrainer(place=fluid.CPUPlace())
    trainer.train()


if __name__ == '__main__':
    main()
