# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.fluid as fluid
from paddle.device.cuda.graphs import CUDAGraph
import unittest
import numpy as np
import os
import pathlib
import shutil
from paddle.fluid.dygraph.base import switch_to_static_graph
from simple_nets import simple_fc_net_with_inputs


def can_use_cuda_graph():
    return paddle.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm()


class TestCUDAGraph(unittest.TestCase):
    def setUp(self):
        if can_use_cuda_graph():
            paddle.set_flags({
                'FLAGS_allocator_strategy': 'auto_growth',
                'FLAGS_sync_nccl_allreduce': False,
                'FLAGS_cudnn_deterministic': True,
                'FLAGS_use_stream_safe_cuda_allocator': False,
            })

    def random_tensor(self, shape):
        return paddle.to_tensor(
            np.random.randint(
                low=0, high=10, size=shape).astype("float32"))

    @switch_to_static_graph
    def test_cuda_graph_static_graph(self):
        if not can_use_cuda_graph():
            return

        seed = 100
        loss_cuda_graph = self.cuda_graph_static_graph_main(
            seed, use_cuda_graph=True)
        loss_no_cuda_graph = self.cuda_graph_static_graph_main(
            seed, use_cuda_graph=False)
        self.assertEqual(loss_cuda_graph, loss_no_cuda_graph)

    def cuda_graph_static_graph_main(self, seed, use_cuda_graph):
        batch_size = 1
        class_num = 10
        image_shape = [batch_size, 784]
        label_shape = [batch_size, 1]

        paddle.seed(seed)
        np.random.seed(seed)
        startup = paddle.static.Program()
        main = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            image = paddle.static.data(
                name="image", shape=image_shape, dtype='float32')
            label = paddle.static.data(
                name="label", shape=label_shape, dtype='int64')
            image.persistable = True
            label.persistable = True
            loss = simple_fc_net_with_inputs(image, label, class_num)
            loss.persistable = True
            lr = paddle.optimizer.lr.PiecewiseDecay(
                boundaries=[2, 3, 4], values=[0.01, 0.02, 0.03, 0.04])
            optimizer = paddle.optimizer.SGD(learning_rate=lr)
            optimizer.minimize(loss)
        place = paddle.CUDAPlace(0)
        exe = paddle.static.Executor(place)
        scope = paddle.static.Scope()
        with paddle.static.scope_guard(scope):
            exe.run(startup)
            build_strategy = paddle.static.BuildStrategy()
            build_strategy.allow_cuda_graph_capture = True
            build_strategy.fix_op_run_order = True
            build_strategy.fuse_all_optimizer_ops = True
            compiled_program = paddle.static.CompiledProgram(
                main).with_data_parallel(
                    loss_name=loss.name,
                    build_strategy=build_strategy,
                    places=place)
            image_t = scope.var(image.name).get_tensor()
            label_t = scope.var(label.name).get_tensor()
            loss_t = scope.var(loss.name).get_tensor()
            lr_var = main.global_block().var(lr._var_name)
            self.assertTrue(lr_var.persistable)
            lr_t = scope.var(lr_var.name).get_tensor()
            cuda_graph = None
            for batch_id in range(20):
                image_t.set(
                    np.random.rand(*image_shape).astype('float32'), place)
                label_t.set(np.random.randint(
                    low=0, high=class_num, size=label_shape, dtype='int64'),
                            place)

                if batch_id == 1 and use_cuda_graph:
                    cuda_graph = CUDAGraph(place, mode="global")
                    cuda_graph.capture_begin()
                    exe.run(compiled_program)
                    cuda_graph.capture_end()

                if cuda_graph:
                    lr_t.set(np.array([lr()], dtype='float32'), place)
                    cuda_graph.replay()
                else:
                    exe.run(compiled_program)
                lr.step()
            if cuda_graph:
                cuda_graph.reset()
        return np.array(loss_t)

    def test_cuda_graph_dynamic_graph(self):
        if not can_use_cuda_graph():
            return

        shape = [2, 3]
        x = self.random_tensor(shape)
        z = self.random_tensor(shape)

        g = CUDAGraph()
        g.capture_begin()
        y = x + 10
        z.add_(x)
        g.capture_end()

        for _ in range(10):
            z_np_init = z.numpy()
            x_new = self.random_tensor(shape)
            x.copy_(x_new, False)
            g.replay()
            x_np = x_new.numpy()
            y_np = y.numpy()
            z_np = z.numpy()
            self.assertTrue((y_np - x_np == 10).all())
            self.assertTrue((z_np - z_np_init == x_np).all())

        g.reset()

    def test_concat_and_split(self):
        if not can_use_cuda_graph():
            return

        concat_num = 100
        xs = []
        xs_np = []

        for i in range(concat_num):
            x_np = np.random.random(size=[1]).astype(np.float32)
            xs.append(paddle.to_tensor(x_np))
            xs_np.append(x_np)

        graph = CUDAGraph()
        graph.capture_begin()
        y = paddle.concat(xs)
        zs = paddle.split(y, len(xs))
        graph.capture_end()
        graph.replay()

        y_np = y.numpy()
        y_np_expected = np.concatenate(xs_np)
        self.assertTrue(np.array_equal(y_np, y_np_expected))
        self.assertEqual(len(zs), len(xs_np))
        for i, z in enumerate(zs):
            self.assertTrue(np.array_equal(z.numpy(), xs_np[i]))

        output_dir = 'cuda_graph_dot_{}'.format(os.getpid())
        try:
            graph.print_to_dot_files(pathlib.Path(output_dir))
            graph.reset()
            shutil.rmtree(output_dir)
        except Exception as e:
            msg = str(e)
            sub_msg = "The print_to_dot_files() method is only supported when CUDA version >= 11.3"
            self.assertTrue(sub_msg in msg)
        finally:
            graph.reset()

    def test_dataloader(self):
        if not can_use_cuda_graph():
            return

        class AutoIncDataset(paddle.io.Dataset):
            def __init__(self, n, dtype):
                self.n = n
                self.dtype = dtype

            def __len__(self):
                return self.n

            def __getitem__(self, idx):
                return np.array([idx]).astype(self.dtype)

        n = 100
        dtype = 'int64'
        dataset = AutoIncDataset(n, dtype)
        data_loader = paddle.io.DataLoader(
            dataset, batch_size=1, num_workers=2, use_buffer_reader=True)
        x = None
        y = None

        graph = None
        for i, data in enumerate(data_loader):
            if graph is None:
                x = data
                x = x.cuda()
                graph = CUDAGraph()
                graph.capture_begin()
                y = x * x
                graph.capture_end()
            else:
                x.copy_(data, False)
                x = x.cuda()

            graph.replay()
            actual_x = np.array([[i]]).astype(dtype)
            actual_y = np.array([[i * i]]).astype(dtype)
            self.assertTrue(np.array_equal(actual_x, x.numpy()))
            self.assertTrue(np.array_equal(actual_y, y.numpy()))


if __name__ == "__main__":
    unittest.main()
