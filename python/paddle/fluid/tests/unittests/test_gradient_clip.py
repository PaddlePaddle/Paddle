#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import unittest
import numpy as np
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid


def bow_net(data,
            label,
            dict_dim,
            emb_dim=128,
            hid_dim=128,
            hid_dim2=96,
            class_dim=2):
    """
    BOW net
    This model is from https://github.com/PaddlePaddle/models:
    fluid/PaddleNLP/text_classification/nets.py
    """
    emb = fluid.layers.embedding(
        input=data, is_sparse=True, size=[dict_dim, emb_dim])
    bow = fluid.layers.sequence_pool(input=emb, pool_type='sum')
    bow_tanh = fluid.layers.tanh(bow)
    fc_1 = fluid.layers.fc(input=bow_tanh, size=hid_dim, act="tanh")
    fc_2 = fluid.layers.fc(input=fc_1, size=hid_dim2, act="tanh")
    prediction = fluid.layers.fc(input=[fc_2], size=class_dim, act="softmax")
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    return avg_cost


class TestGradientClip(unittest.TestCase):
    def setUp(self):
        self.word_dict = paddle.dataset.imdb.word_dict()
        self.BATCH_SIZE = 2
        self.train_data = paddle.batch(
            paddle.dataset.imdb.train(self.word_dict),
            batch_size=self.BATCH_SIZE)

    def get_places(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        return places

    def check_operators(self, place):
        CLIP = 1

        prog = fluid.framework.Program()
        startup_program = fluid.framework.Program()
        with fluid.program_guard(
                main_program=prog, startup_program=startup_program):
            image = fluid.layers.data(name='x', shape=[784], dtype='float32')
            label = fluid.layers.data(name='y', shape=[1], dtype='int64')

            hidden1 = fluid.layers.fc(input=image, size=128, act='relu')
            hidden2 = fluid.layers.fc(input=hidden1, size=64, act='relu')
            predict = fluid.layers.fc(input=hidden2, size=10, act='softmax')

            cost = fluid.layers.cross_entropy(input=predict, label=label)
            avg_cost = fluid.layers.mean(cost)

        prog_clip = prog.clone()
        avg_cost_clip = prog_clip.block(0).var(avg_cost.name)

        p_g = fluid.backward.append_backward(loss=avg_cost)
        p_g_clip = fluid.backward.append_backward(loss=avg_cost_clip)

        with fluid.program_guard(
                main_program=prog_clip, startup_program=startup_program):
            fluid.clip.set_gradient_clip(
                fluid.clip.GradientClipByGlobalNorm(clip_norm=CLIP))
            p_g_clip = fluid.clip.append_gradient_clip_ops(p_g_clip)

        grad_list = [elem[1] for elem in p_g]
        grad_clip_list = [elem[1] for elem in p_g_clip]

        train_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.mnist.train(), buf_size=8192),
            batch_size=128)

        exe = fluid.Executor(place)
        feeder = fluid.DataFeeder(feed_list=[image, label], place=place)
        exe.run(startup_program)

        count = 0
        for data in train_reader():
            count += 1
            if count > 5:
                break
            out = exe.run(prog, feed=feeder.feed(data), fetch_list=grad_list)
            out_clip = exe.run(prog_clip,
                               feed=feeder.feed(data),
                               fetch_list=grad_clip_list)
            global_norm = 0
            for v in out:
                global_norm += np.sum(np.power(v, 2))
            global_norm = np.sqrt(global_norm)

            global_norm_clip = 0
            for v in out_clip:
                global_norm_clip += np.sum(np.power(v, 2))
            global_norm_clip = np.sqrt(global_norm_clip)

            assert np.isclose(
                a=global_norm_clip, b=np.minimum(global_norm, CLIP), rtol=5e-3)

    def check_sparse_gradient_clip(self, place):
        prog = fluid.framework.Program()
        startup_program = fluid.framework.Program()
        with fluid.program_guard(
                main_program=prog, startup_program=startup_program):
            data = fluid.layers.data(
                name="words", shape=[1], dtype="int64", lod_level=1)
            label = fluid.layers.data(name="label", shape=[1], dtype="int64")
            cost = bow_net(data, label, len(self.word_dict))

            fluid.clip.set_gradient_clip(
                clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=5.0))

            sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01)
            sgd_optimizer.minimize(cost)

        exe = fluid.Executor(place)
        feeder = fluid.DataFeeder(feed_list=[data, label], place=place)
        exe.run(startup_program)

        data = next(self.train_data())
        val = exe.run(prog, feed=feeder.feed(data), fetch_list=[cost])[0]
        self.assertEqual((1, ), val.shape)
        print(val)
        self.assertFalse(np.isnan(val))

    def test_operators(self):
        self.check_operators(core.CPUPlace())

    def test_sparse_gradient_clip(self):
        for place in self.get_places():
            self.check_sparse_gradient_clip(place)


if __name__ == '__main__':
    unittest.main()
