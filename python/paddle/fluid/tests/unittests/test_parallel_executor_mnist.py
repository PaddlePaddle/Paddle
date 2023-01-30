# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

<<<<<<< HEAD
import os
import unittest

import numpy as np
from parallel_executor_test_base import DeviceType, TestParallelExecutorBase

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core


def simple_fc_net(use_feed):
    img = paddle.static.data(name='image', shape=[-1, 784], dtype='float32')
    label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64')
    hidden = img
    for _ in range(4):
        hidden = paddle.static.nn.fc(
            hidden,
            size=200,
            activation='tanh',
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=1.0)
            ),
        )
    prediction = paddle.static.nn.fc(hidden, size=10, activation='softmax')
    loss = paddle.nn.functional.cross_entropy(
        input=prediction, label=label, reduction='none', use_softmax=False
    )
=======
from __future__ import print_function

import unittest

import numpy as np
import paddle.fluid.core as core
import paddle
import os
import paddle.fluid as fluid
from parallel_executor_test_base import TestParallelExecutorBase, DeviceType
from parallel_executor_test_base import DeviceType


def simple_fc_net(use_feed):
    img = fluid.layers.data(name='image', shape=[784], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    hidden = img
    for _ in range(4):
        hidden = fluid.layers.fc(
            hidden,
            size=200,
            act='tanh',
            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
                value=1.0)))
    prediction = fluid.layers.fc(hidden, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    loss = paddle.mean(loss)
    return loss


def fc_with_batchnorm(use_feed):
<<<<<<< HEAD
    img = paddle.static.data(name='image', shape=[-1, 784], dtype='float32')
    label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64')
=======
    img = fluid.layers.data(name='image', shape=[784], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    hidden = img
    for _ in range(1):
        with fluid.name_scope("hidden"):
<<<<<<< HEAD
            hidden = paddle.static.nn.fc(
                hidden,
                size=200,
                activation='tanh',
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Constant(value=1.0)
                ),
            )

            hidden = paddle.static.nn.batch_norm(input=hidden)
    with fluid.name_scope("fc_layer"):
        prediction = paddle.static.nn.fc(hidden, size=10, activation='softmax')
    with fluid.name_scope("loss"):
        loss = paddle.nn.functional.cross_entropy(
            input=prediction, label=label, reduction='none', use_softmax=False
        )
=======
            hidden = fluid.layers.fc(
                hidden,
                size=200,
                act='tanh',
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Constant(value=1.0)))

            hidden = fluid.layers.batch_norm(input=hidden)
    with fluid.name_scope("fc_layer"):
        prediction = fluid.layers.fc(hidden, size=10, act='softmax')
    with fluid.name_scope("loss"):
        loss = fluid.layers.cross_entropy(input=prediction, label=label)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        loss = paddle.mean(loss)
    return loss


def init_data():
    np.random.seed(5)
    img = np.random.random(size=[32, 784]).astype(np.float32)
    label = np.ones(shape=[32, 1], dtype='int64')
    return img, label


class TestMNIST(TestParallelExecutorBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    @classmethod
    def setUpClass(cls):
        os.environ['CPU_NUM'] = str(4)

<<<<<<< HEAD
    def _compare_reduce_and_allreduce(
        self, model, use_device, delta1=1e-6, delta2=1e-4
    ):
=======
    def _compare_reduce_and_allreduce(self,
                                      model,
                                      use_device,
                                      delta1=1e-6,
                                      delta2=1e-4):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        if use_device == DeviceType.CUDA and not core.is_compiled_with_cuda():
            return

        if use_device == DeviceType.XPU and not core.is_compiled_with_xpu():
            return

        img, label = init_data()

<<<<<<< HEAD
        (
            all_reduce_first_loss,
            all_reduce_last_loss,
            _,
        ) = self.check_network_convergence(
            model,
            feed_dict={"image": img, "label": label},
            use_device=use_device,
            use_reduce=False,
        )

        reduce_first_loss, reduce_last_loss, _ = self.check_network_convergence(
            model,
            feed_dict={"image": img, "label": label},
            use_device=use_device,
            use_reduce=True,
        )
=======
        all_reduce_first_loss, all_reduce_last_loss, _ = self.check_network_convergence(
            model,
            feed_dict={
                "image": img,
                "label": label
            },
            use_device=use_device,
            use_reduce=False)

        reduce_first_loss, reduce_last_loss, _ = self.check_network_convergence(
            model,
            feed_dict={
                "image": img,
                "label": label
            },
            use_device=use_device,
            use_reduce=True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        for loss in zip(all_reduce_first_loss, reduce_first_loss):
            self.assertAlmostEqual(loss[0], loss[1], delta=delta1)
        for loss in zip(all_reduce_last_loss, reduce_last_loss):
            self.assertAlmostEqual(loss[0], loss[1], delta=delta2)

    # simple_fc
    def check_simple_fc_convergence(self, use_device, use_reduce=False):
        if use_device == DeviceType.CUDA and not core.is_compiled_with_cuda():
            return

        if use_device == DeviceType.XPU and not core.is_compiled_with_xpu():
            return

        img, label = init_data()

<<<<<<< HEAD
        self.check_network_convergence(
            simple_fc_net,
            feed_dict={"image": img, "label": label},
            use_device=use_device,
            use_reduce=use_reduce,
        )
=======
        self.check_network_convergence(simple_fc_net,
                                       feed_dict={
                                           "image": img,
                                           "label": label
                                       },
                                       use_device=use_device,
                                       use_reduce=use_reduce)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_simple_fc(self):
        # use_device
        self.check_simple_fc_convergence(DeviceType.CUDA)
        self.check_simple_fc_convergence(DeviceType.CPU)
        self.check_simple_fc_convergence(DeviceType.XPU)

    def test_simple_fc_with_new_strategy(self):
        # use_device, use_reduce
        # NOTE: the computation result of nccl_reduce is non-deterministic,
        # related issue: https://github.com/NVIDIA/nccl/issues/157
<<<<<<< HEAD
        self._compare_reduce_and_allreduce(
            simple_fc_net, DeviceType.CUDA, 1e-5, 1e-2
        )
        self._compare_reduce_and_allreduce(
            simple_fc_net, DeviceType.CPU, 1e-5, 1e-2
        )
=======
        self._compare_reduce_and_allreduce(simple_fc_net, DeviceType.CUDA, 1e-5,
                                           1e-2)
        self._compare_reduce_and_allreduce(simple_fc_net, DeviceType.CPU, 1e-5,
                                           1e-2)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def check_simple_fc_parallel_accuracy(self, use_device):
        if use_device == DeviceType.CUDA and not core.is_compiled_with_cuda():
            return

        img, label = init_data()

        single_first_loss, single_last_loss, _ = self.check_network_convergence(
            method=simple_fc_net,
<<<<<<< HEAD
            feed_dict={"image": img, "label": label},
            use_device=use_device,
            use_parallel_executor=False,
        )
        (
            parallel_first_loss,
            parallel_last_loss,
            _,
        ) = self.check_network_convergence(
            method=simple_fc_net,
            feed_dict={"image": img, "label": label},
            use_device=use_device,
            use_parallel_executor=True,
        )

        self.assertAlmostEqual(
=======
            feed_dict={
                "image": img,
                "label": label
            },
            use_device=use_device,
            use_parallel_executor=False)
        parallel_first_loss, parallel_last_loss, _ = self.check_network_convergence(
            method=simple_fc_net,
            feed_dict={
                "image": img,
                "label": label
            },
            use_device=use_device,
            use_parallel_executor=True)

        self.assertAlmostEquals(
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            np.mean(parallel_first_loss),
            single_first_loss,
            delta=1e-6,
        )
<<<<<<< HEAD
        self.assertAlmostEqual(
            np.mean(parallel_last_loss), single_last_loss, delta=1e-6
        )
=======
        self.assertAlmostEquals(np.mean(parallel_last_loss),
                                single_last_loss,
                                delta=1e-6)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_simple_fc_parallel_accuracy(self):
        self.check_simple_fc_parallel_accuracy(DeviceType.CUDA)
        self.check_simple_fc_parallel_accuracy(DeviceType.CPU)

    def check_batchnorm_fc_convergence(self, use_device, use_fast_executor):
        if use_device == DeviceType.CUDA and not core.is_compiled_with_cuda():
            return
        if use_device == DeviceType.XPU and not core.is_compiled_with_xpu():
            return
        img, label = init_data()

<<<<<<< HEAD
        self.check_network_convergence(
            fc_with_batchnorm,
            feed_dict={"image": img, "label": label},
            use_device=use_device,
            use_fast_executor=use_fast_executor,
        )
=======
        self.check_network_convergence(fc_with_batchnorm,
                                       feed_dict={
                                           "image": img,
                                           "label": label
                                       },
                                       use_device=use_device,
                                       use_fast_executor=use_fast_executor)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_batchnorm_fc(self):
        for use_device in (DeviceType.CPU, DeviceType.CUDA):
            for use_fast_executor in (False, True):
<<<<<<< HEAD
                self.check_batchnorm_fc_convergence(
                    use_device, use_fast_executor
                )
=======
                self.check_batchnorm_fc_convergence(use_device,
                                                    use_fast_executor)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_batchnorm_fc_with_new_strategy(self):
        # NOTE: the computation result of nccl_reduce is non-deterministic,
        # related issue: https://github.com/NVIDIA/nccl/issues/157
<<<<<<< HEAD
        self._compare_reduce_and_allreduce(
            fc_with_batchnorm, DeviceType.CUDA, 1e-5, 1e-2
        )
        self._compare_reduce_and_allreduce(
            fc_with_batchnorm, DeviceType.CPU, 1e-5, 1e-2
        )


class TestMNISTNoReduce(unittest.TestCase):
=======
        self._compare_reduce_and_allreduce(fc_with_batchnorm, DeviceType.CUDA,
                                           1e-5, 1e-2)
        self._compare_reduce_and_allreduce(fc_with_batchnorm, DeviceType.CPU,
                                           1e-5, 1e-2)


class TestMNISTNoReduce(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def run_program(self, device_type):
        if device_type == DeviceType.CUDA:
            if not paddle.is_compiled_with_cuda():
                return
            places = paddle.static.cuda_places()
        else:
            self.assertEqual(device_type, DeviceType.CPU)
            places = paddle.static.cpu_places(4)

        paddle.seed(10)
        with paddle.fluid.unique_name.guard():
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                loss = simple_fc_net(use_feed=True)
                optimizer = paddle.optimizer.SGD(learning_rate=0.0)
                optimizer.minimize(loss)

        grads = [p.name + '@GRAD' for p in main.all_parameters()]
        no_reduce = paddle.static.BuildStrategy.ReduceStrategy._NoReduce

        build_strategy = paddle.static.BuildStrategy()
        build_strategy.reduce_strategy = no_reduce
        main_multi_place = paddle.static.CompiledProgram(
<<<<<<< HEAD
            main
        ).with_data_parallel(
            loss_name=loss.name, build_strategy=build_strategy, places=places
        )
=======
            main).with_data_parallel(loss_name=loss.name,
                                     build_strategy=build_strategy,
                                     places=places)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        build_strategy = paddle.static.BuildStrategy()
        build_strategy.reduce_strategy = no_reduce
        main_single_place = paddle.static.CompiledProgram(
<<<<<<< HEAD
            main.clone()
        ).with_data_parallel(
            loss_name=loss.name, build_strategy=build_strategy, places=places[0]
        )
=======
            main.clone()).with_data_parallel(loss_name=loss.name,
                                             build_strategy=build_strategy,
                                             places=places[0])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        image, label = init_data()
        feed = {'image': image, 'label': label}
        exe = paddle.static.Executor(places[0])
        scope = paddle.static.Scope()
        with paddle.static.scope_guard(scope):
            exe.run(startup)
<<<<<<< HEAD
            grads_multi_place = exe.run(
                main_multi_place, feed=feed, fetch_list=[grads]
            )
=======
            grads_multi_place = exe.run(main_multi_place,
                                        feed=feed,
                                        fetch_list=[grads])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            feeds = self.split_feed(feed, len(places))
            grads_single_place = [list() for _ in range(len(grads))]
            for f in feeds:
                gs = exe.run(main_single_place, feed=f, fetch_list=[grads])
                for i, g in enumerate(gs):
                    grads_single_place[i].append(g)

            for i in range(len(grads)):
<<<<<<< HEAD
                grads_single_place[i] = np.concatenate(
                    grads_single_place[i], axis=0
                ) / len(places)
=======
                grads_single_place[i] = np.concatenate(grads_single_place[i],
                                                       axis=0) / len(places)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.assertEqual(len(grads_multi_place), len(grads_single_place))
        for g1, g2 in zip(grads_multi_place, grads_single_place):
            np.testing.assert_allclose(g1, g2, rtol=1e-05)

    def split_feed(self, feed, n):
        image = feed['image']
        label = feed['label']
        self.assertEqual(image.shape[0] % n, 0)
        self.assertEqual(label.shape[0] % n, 0)
        images = np.split(image, n)
        labels = np.split(label, n)
        return [{'image': images[i], 'label': labels[i]} for i in range(n)]

    def test_main(self):
        self.run_program(DeviceType.CUDA)
        self.run_program(DeviceType.CPU)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
