# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
import six
from fake_reader import fake_imdb_reader
from paddle.fluid.clip import _allow_pure_fp16_global_norm_clip

paddle.enable_static()


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
        self.word_dict_len = 5147
        self.BATCH_SIZE = 2
        reader = fake_imdb_reader(self.word_dict_len, self.BATCH_SIZE * 100)
        self.train_data = paddle.batch(reader, batch_size=self.BATCH_SIZE)
        self.clip_gradient = lambda x: None
        self.init()

    def init(self):
        pass

    def get_places(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        return places

    def check_clip_result(self, out, out_clip):
        pass

    def check_gradient_clip(self, place, dtype='float32'):
        prog = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(
                main_program=prog, startup_program=startup_program):
            image = fluid.data(name="a", shape=[-1, 784], dtype='float32')
            label = fluid.data(name="b", shape=[-1, 1], dtype='int64')
            if dtype != 'float32':
                image_cast = paddle.cast(image, dtype)
                hidden = fluid.layers.fc(input=image_cast, size=32, act='relu')
            else:
                hidden = fluid.layers.fc(input=image, size=32, act='relu')
            predict = fluid.layers.fc(input=hidden, size=10, act='softmax')

            cost = fluid.layers.cross_entropy(input=predict, label=label)
            avg_cost = fluid.layers.mean(cost)

        prog_clip = prog.clone()
        avg_cost_clip = prog_clip.block(0).var(avg_cost.name)

        p_g = fluid.backward.append_backward(loss=avg_cost)
        p_g_clip = fluid.backward.append_backward(loss=avg_cost_clip)

        p_g = sorted(p_g, key=lambda x: x[0].name)
        p_g_clip = sorted(p_g_clip, key=lambda x: x[0].name)
        with fluid.program_guard(
                main_program=prog_clip, startup_program=startup_program):
            p_g_clip = self.clip_gradient(p_g_clip)

        grad_list = [elem[1] for elem in p_g]
        grad_clip_list = [elem[1] for elem in p_g_clip]

        train_reader = paddle.batch(paddle.dataset.mnist.train(), batch_size=3)
        exe = fluid.Executor(place)
        feeder = fluid.DataFeeder(feed_list=[image, label], place=place)
        exe.run(startup_program)

        data = next(train_reader())
        out = exe.run(prog, feed=feeder.feed(data), fetch_list=grad_list)
        out_clip = exe.run(prog_clip,
                           feed=feeder.feed(data),
                           fetch_list=grad_clip_list)
        self.check_clip_result(out, out_clip)

    def check_sparse_gradient_clip(self, place):
        prog = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(
                main_program=prog, startup_program=startup_program):
            data = fluid.data(
                name="words", shape=[-1, 1], dtype="int64", lod_level=1)
            label = fluid.data(name="label", shape=[-1, 1], dtype="int64")
            cost = bow_net(data, label, self.word_dict_len)

            self.backward_and_optimize(cost)

        exe = fluid.Executor(place)
        feeder = fluid.DataFeeder(feed_list=[data, label], place=place)
        exe.run(startup_program)

        data = next(self.train_data())
        val = exe.run(prog, feed=feeder.feed(data), fetch_list=[cost])[0]
        self.assertEqual((1, ), val.shape)
        self.assertFalse(np.isnan(val))

    def backward_and_optimize(self, cost):
        pass


class TestGradientClipByGlobalNorm(TestGradientClip):
    def init(self):
        self.clip_norm = 0.2

    def check_clip_result(self, out, out_clip):
        global_norm = 0
        for v in out:
            global_norm += np.sum(np.square(v))
        global_norm = np.sqrt(global_norm)
        scale = self.clip_norm / np.maximum(self.clip_norm, global_norm)
        res = []
        for i in range(len(out)):
            out[i] = scale * out[i]

        for u, v in zip(out, out_clip):
            self.assertTrue(
                np.allclose(
                    a=u, b=v, rtol=1e-5, atol=1e-8),
                "gradient clip by global norm has wrong results!, \nu={}\nv={}\ndiff={}".
                format(u, v, u - v))

    # test whether the ouput is right when use 'set_gradient_clip'
    def test_old_gradient_clip(self):
        def func(params_grads):
            clip = fluid.clip.GradientClipByGlobalNorm(clip_norm=self.clip_norm)
            fluid.clip.set_gradient_clip(clip)
            return fluid.clip.append_gradient_clip_ops(params_grads)

        self.clip_gradient = func
        self.check_gradient_clip(fluid.CPUPlace())

    # test whether the ouput is right when use grad_clip
    def test_new_gradient_clip(self):
        def func(params_grads):
            clip = fluid.clip.GradientClipByGlobalNorm(clip_norm=self.clip_norm)
            return clip(params_grads)

        self.clip_gradient = func
        self.check_gradient_clip(fluid.CPUPlace())

    # test whether the ouput is right when use grad_clip under float64
    def test_new_gradient_clip_fp64(self):
        def func(params_grads):
            clip = fluid.clip.GradientClipByGlobalNorm(clip_norm=self.clip_norm)
            return clip(params_grads)

        self.clip_gradient = func
        self.check_gradient_clip(fluid.CPUPlace(), "float64")

    # invoke 'set_gradient_clip' in a wrong order
    def test_wrong_API_order(self):
        def backward_func(cost):
            clip = fluid.clip.GradientClipByGlobalNorm(clip_norm=5.0)
            fluid.clip.set_gradient_clip(clip)
            sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01,
                                                grad_clip=clip)
            # if 'set_gradient_clip' and 'optimize(grad_clip)' together, 'set_gradient_clip' will be ineffective
            sgd_optimizer.minimize(cost)
            # 'set_gradient_clip' must before 'minimize', otherwise, 'set_gradient_clip' will be ineffective
            fluid.clip.set_gradient_clip(clip)

        self.backward_and_optimize = backward_func
        for place in self.get_places():
            self.check_sparse_gradient_clip(place)

    # raise typeError
    def test_tpyeError(self):
        # the type of optimizer(grad_clip=) must be an instance of GradientClipBase's derived class
        with self.assertRaises(TypeError):
            sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.1,
                                                grad_clip="test")

    # if grad is None or not need clip
    def test_none_grad_fp32(self):
        ops = self._test_none_grad_helper("float32")
        self.assertListEqual(ops, [
            'squared_l2_norm', 'squared_l2_norm', 'sum', 'sqrt',
            'fill_constant', 'elementwise_max', 'elementwise_div',
            'elementwise_mul', 'elementwise_mul'
        ])

    def test_none_grad_fp16(self):
        ops = self._test_none_grad_helper("float16")
        self.assertListEqual(ops, [
            'square', 'reduce_sum', 'square', 'reduce_sum', 'sum', 'cast',
            'sqrt', 'fill_constant', 'elementwise_max', 'elementwise_div',
            'cast', 'elementwise_mul', 'cast', 'elementwise_mul'
        ])

    def _test_none_grad_helper(self, dtype):
        prog = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(
                main_program=prog, startup_program=startup_program):
            clip = fluid.clip.GradientClipByGlobalNorm(self.clip_norm)
            x = fluid.default_main_program().global_block().create_parameter(
                name="x", shape=[2, 3], dtype=dtype)
            y = fluid.default_main_program().global_block().create_parameter(
                name="y", shape=[2, 3], dtype=dtype)

            # (x, None) should not be returned
            params_grads = [(x, None), (x, y), (y, x)]
            params_grads = clip(params_grads)
            self.assertTrue(
                len(params_grads) == 2,
                "ClipByGlobalNorm: when grad is None, it shouldn't be returned by gradient clip!"
            )

            ops = [op.type for op in x.block.ops]
        return ops


class TestGradientClipByNorm(TestGradientClip):
    def init(self):
        self.clip_norm = 0.2

    def check_clip_result(self, out, out_clip):
        for u, v in zip(out, out_clip):
            norm = np.sqrt(np.sum(np.power(u, 2)))
            scale = self.clip_norm / np.maximum(self.clip_norm, norm)
            u = u * scale
            self.assertTrue(
                np.allclose(
                    a=u, b=v, rtol=1e-5, atol=1e-8),
                "gradient clip by norm has wrong results!")

    # test whether the ouput is right when use grad_clip
    def test_gradient_clip(self):
        def func(params_grads):
            clip = fluid.clip.GradientClipByNorm(clip_norm=self.clip_norm)
            return clip(params_grads)

        self.clip_gradient = func
        self.check_gradient_clip(fluid.CPUPlace())

    # if grad is None or not need clip
    def test_none_grad(self):
        clip = fluid.clip.GradientClipByNorm(self.clip_norm)
        x = fluid.default_main_program().global_block().create_parameter(
            name="x", shape=[2, 3], dtype="float32", need_clip=False)
        y = fluid.default_main_program().global_block().create_parameter(
            name="y", shape=[2, 3], dtype="float32", need_clip=False)

        # (x, None) should not be returned
        params_grads = [(x, None), (x, y)]
        params_grads = clip(params_grads)
        self.assertTrue(
            len(clip(params_grads)) == 1,
            "ClipGradByNorm: when grad is None, it shouldn't be returned by gradient clip!"
        )
        self.assertTrue(
            params_grads[0][1].name == 'y',
            "ClipGradByNorm: grad should not be clipped when filtered out!")


class TestGradientClipByValue(TestGradientClip):
    def init(self):
        self.max = 0.2
        self.min = 0.1

    def check_clip_result(self, out, out_clip):
        for i, v in enumerate(out):
            out[i] = np.clip(v, self.min, self.max)
        for u, v in zip(out, out_clip):
            u = np.clip(u, self.min, self.max)
            self.assertTrue(
                np.allclose(
                    a=u, b=v, rtol=1e-6, atol=1e-8),
                "gradient clip by value has wrong results!")

    # test whether the ouput is right when use grad_clip
    def test_gradient_clip(self):
        def func(params_grads):
            clip = fluid.clip.GradientClipByValue(max=self.max, min=self.min)
            return clip(params_grads)

        self.clip_gradient = func
        self.check_gradient_clip(fluid.CPUPlace())

    # if grad is None or not need clip
    def test_none_grad(self):
        clip = fluid.clip.GradientClipByValue(self.max, self.min)
        x = fluid.default_main_program().global_block().create_parameter(
            name="x", shape=[2, 3], dtype="float32", need_clip=False)
        y = fluid.default_main_program().global_block().create_parameter(
            name="y", shape=[2, 3], dtype="float32", need_clip=False)

        # (x, None) should not be returned
        params_grads = [(x, None), (x, y)]
        params_grads = clip(params_grads)
        self.assertTrue(
            len(clip(params_grads)) == 1,
            "ClipGradByValue: when grad is None, it shouldn't be returned by gradient clip!"
        )
        self.assertTrue(
            params_grads[0][1].name == 'y',
            "ClipGradByValue: grad should not be clipped when filtered out!")


class TestDygraphGradientClip(unittest.TestCase):
    def test_gradient_clip(self):
        with fluid.dygraph.guard():
            linear = fluid.dygraph.Linear(5, 5)
            inputs = fluid.layers.uniform_random(
                [16, 5], min=-10, max=10).astype('float32')
            out = linear(fluid.dygraph.to_variable(inputs))
            loss = fluid.layers.reduce_mean(out)
            loss.backward()
            sgd_optimizer = fluid.optimizer.SGD(
                learning_rate=0.0,
                parameter_list=linear.parameters(),
                grad_clip=fluid.clip.GradientClipByGlobalNorm(0.1))
            self.check_clip_result(loss, sgd_optimizer)

    def check_clip_result(self, loss, optimizer):
        pass


class TestDygraphGradientClipByGlobalNorm(TestDygraphGradientClip):
    def setUp(self):
        self.clip_norm = 0.8
        self.clip1 = fluid.clip.GradientClipByGlobalNorm(
            clip_norm=self.clip_norm)
        self.clip2 = fluid.clip.GradientClipByGlobalNorm(
            clip_norm=self.clip_norm)

    def check_clip_result(self, loss, optimizer):
        # if grad is None
        x = fluid.dygraph.to_variable(
            np.array([2, 3]).astype("float32"), name="x")
        y = fluid.dygraph.to_variable(
            np.array([3, 4]).astype("float32"), name="y")
        assert len(self.clip1([(x, x), (x, y), (x, None)])) == 2
        # get params and grads from network
        opt, params_grads = optimizer.minimize(loss)
        _, grads = zip(*params_grads)
        params_grads = self.clip2(params_grads)
        _, grads_clip = zip(*params_grads)

        global_norm = 0
        for u in grads:
            u = u.numpy()
            global_norm += np.sum(np.power(u, 2))
        global_norm = np.sqrt(global_norm)

        global_norm_clip = 0
        for v in grads_clip:
            v = v.numpy()
            global_norm_clip += np.sum(np.power(v, 2))
        global_norm_clip = np.sqrt(global_norm_clip)

        a = np.minimum(global_norm, self.clip_norm)
        b = global_norm_clip
        self.assertTrue(
            np.isclose(
                a=a, b=b, rtol=1e-6, atol=1e-8),
            "gradient clip by global norm has wrong results, expetcd:%f, but recieved:%f"
            % (a, b))


class TestDygraphGradientClipByNorm(TestDygraphGradientClip):
    def setUp(self):
        self.clip_norm = 0.8
        self.clip = fluid.clip.GradientClipByNorm(clip_norm=self.clip_norm)

    def check_clip_result(self, loss, optimizer):
        # if grad is None
        x = fluid.dygraph.to_variable(np.array([2, 3]).astype("float32"))
        assert len(self.clip([(x, None)])) == 0
        # get params and grads from network
        self.clip([(fluid.dygraph.to_variable(np.array([2, 3])), None)])
        opt, params_grads = optimizer.minimize(loss)
        _, grads = zip(*params_grads)
        params_grads = self.clip(params_grads)
        _, grads_clip = zip(*params_grads)

        for u, v in zip(grads, grads_clip):
            u = u.numpy()
            v = v.numpy()
            a = np.sqrt(np.sum(np.power(u, 2)))
            a = np.minimum(a, self.clip_norm)
            b = np.sqrt(np.sum(np.power(v, 2)))
            self.assertTrue(
                np.isclose(
                    a=a, b=b, rtol=1e-6, atol=1e-8),
                "gradient clip by norm has wrong results, expetcd:%f, but recieved:%f"
                % (a, b))


class TestDygraphGradientClipByValue(TestDygraphGradientClip):
    def setUp(self):
        self.max = 0.2
        self.min = 0.1
        self.clip = fluid.clip.GradientClipByValue(max=self.max, min=self.min)

    def check_clip_result(self, loss, optimizer):
        # if grad is None
        x = fluid.dygraph.to_variable(np.array([2, 3]).astype("float32"))
        assert len(self.clip([(x, None)])) == 0
        # get params and grads from network
        opt, params_grads = optimizer.minimize(loss)
        _, grads = zip(*params_grads)
        params_grads = self.clip(params_grads)
        _, grads_clip = zip(*params_grads)
        for u, v in zip(grads, grads_clip):
            u = np.clip(u.numpy(), self.min, self.max)
            v = v.numpy()
            self.assertTrue(
                np.allclose(
                    a=u, b=v, rtol=1e-6, atol=1e-8),
                "gradient clip by value has wrong results!")


class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = paddle.nn.Linear(5, 5)
        self.batch_norm = paddle.nn.BatchNorm(5)

    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        return x


class TestDygraphGradientClipFP16(unittest.TestCase):
    def test_gradient_clip(self):
        if fluid.core.is_compiled_with_cuda():
            with fluid.dygraph.guard():
                paddle.seed(10)
                model = SimpleNet()
                sgd_optimizer = paddle.optimizer.SGD(
                    learning_rate=0.0, parameters=model.parameters())
                model, sgd_optimizer = paddle.amp.decorate(
                    models=model, optimizers=sgd_optimizer, level='O2')
                scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
                inputs = fluid.layers.uniform_random(
                    [1, 5], min=-10, max=10).astype('float32')
                with paddle.amp.auto_cast(level='O2'):
                    out = model(fluid.dygraph.to_variable(inputs))
                    loss = fluid.layers.reduce_mean(out)
                scaled = scaler.scale(loss)
                scaled.backward()
                scaler.unscale_(sgd_optimizer)
                # before clip
                params_grads = []
                for param in model.parameters():
                    if param.stop_gradient:
                        continue
                    if param._grad_ivar() is not None:
                        params_grads.append((param, param._grad_ivar()))
                _, grads = zip(*params_grads)
                # clip grads
                clip = fluid.clip.GradientClipByGlobalNorm(clip_norm=0.8)
                params_grads = clip(params_grads)
                _, grads_clip = zip(*params_grads)
                # param update                      
                scaler.step(sgd_optimizer)
                scaler.update()

                global_norm = 0
                for u in grads:
                    u = u.numpy()
                    global_norm += np.sum(np.power(u, 2))
                global_norm = np.sqrt(global_norm)
                global_norm_clip = 0
                for v in grads_clip:
                    v = v.numpy()
                    global_norm_clip += np.sum(np.power(v, 2))
                global_norm_clip = np.sqrt(global_norm_clip)

                a = np.minimum(global_norm, 0.8)
                b = global_norm_clip
                self.assertTrue(
                    np.isclose(
                        a=a, b=b, rtol=1e-3, atol=1e-8),
                    "gradient clip by global norm has wrong results, expetcd:%f, but recieved:%f"
                    % (a, b))


class TestDygraphGradientClipFP64(unittest.TestCase):
    def test_gradient_clip(self):
        with fluid.dygraph.guard():
            inputs = fluid.layers.uniform_random(
                [16, 5], min=-10, max=10).astype('float64')
            linear = fluid.dygraph.Linear(5, 5, dtype="float64")
            out = linear(fluid.dygraph.to_variable(inputs))
            loss = fluid.layers.reduce_mean(out)
            loss.backward()
            # before clip
            params_grads = []
            for param in linear.parameters():
                if param.stop_gradient:
                    continue
                if param._grad_ivar() is not None:
                    params_grads.append((param, param._grad_ivar()))
            _, grads = zip(*params_grads)
            # clip grads
            clip = fluid.clip.GradientClipByGlobalNorm(clip_norm=0.1)
            params_grads = clip(params_grads)
            _, grads_clip = zip(*params_grads)

            global_norm = 0
            for u in grads:
                u = u.numpy()
                global_norm += np.sum(np.power(u, 2))
            global_norm = np.sqrt(global_norm)

            global_norm_clip = 0
            for v in grads_clip:
                v = v.numpy()
                print(v)
                global_norm_clip += np.sum(np.power(v, 2))
            global_norm_clip = np.sqrt(global_norm_clip)
            print(global_norm_clip)

            a = np.minimum(global_norm, 0.1)
            b = global_norm_clip

            self.assertTrue(
                np.isclose(
                    a=a, b=b, rtol=1e-6, atol=1e-8),
                "gradient clip by global norm has wrong results, expetcd:%f, but recieved:%f"
                % (a, b))


class TestPureFP16ClipGradByGlobalNorm(unittest.TestCase):
    def check_main(self, expected_has_cast_op):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, startup_prog):
            names = ["p0", "p1"]
            shapes = [[2, 3], [4, 5]]

            param_and_grads = []
            main_block = main_prog.global_block()
            for name, shape in zip(names, shapes):
                p = main_block.create_parameter(
                    name=name, shape=shape, dtype='float16')
                g = main_block.create_parameter(
                    name=p.name + '@GRAD', shape=p.shape, dtype=p.dtype)
                param_and_grads.append((p, g))

            clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
            clip(param_and_grads)
            actual_has_cast = any(op.type == 'cast' for op in main_block.ops)
            self.assertEqual(actual_has_cast, expected_has_cast_op)

    def test_main(self):
        self.check_main(True)
        _allow_pure_fp16_global_norm_clip(True)
        self.check_main(False)
        _allow_pure_fp16_global_norm_clip(False)
        self.check_main(True)


if __name__ == '__main__':
    unittest.main()
