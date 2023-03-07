# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.device.cuda.graphs import CUDAGraph

# from test_standalone_executor import build_program


paddle.enable_static()


def can_use_cuda_graph():
    return paddle.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm()


def func1(data):
    weight = paddle.randn([64, 64], name='weight')  # gpu
    matmul_out = paddle.matmul(data, weight, name='matmul_out')  # gpus
    bias = paddle.ones([4, 64], dtype='float32', name='bias')
    add_out = paddle.add(matmul_out, bias, name='add_out')
    return add_out, bias


def simple_fc_net_with_inputs(img, label, class_num=10):
    hidden = img
    for _ in range(2):
        hidden = paddle.static.nn.fc(
            hidden,
            size=100,
            activation='relu',
            bias_attr=fluid.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1.0)
            ),
        )
    prediction = paddle.static.nn.fc(
        hidden, size=class_num, activation='softmax'
    )
    loss = paddle.nn.functional.cross_entropy(
        input=prediction, label=label, reduction='none', use_softmax=False
    )
    loss = paddle.mean(loss)
    return loss


def build_program(batch_size, class_num):
    main = paddle.static.Program()
    startup = paddle.static.Program()

    image_shape = [batch_size, 784]
    label_shape = [batch_size, 1]
    with paddle.static.program_guard(main, startup):
        image = paddle.static.data(
            name="image", shape=image_shape, dtype='float32'
        )
        label = paddle.static.data(
            name="label", shape=label_shape, dtype='int64'
        )
        image.persistable = True
        label.persistable = True
        loss = simple_fc_net_with_inputs(image, label, class_num)
        loss.persistable = True
        lr = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[2, 3, 4], values=[0.01, 0.02, 0.03, 0.04]
        )
        optimizer = paddle.optimizer.SGD(learning_rate=lr)
        optimizer.minimize(loss)
    return main, startup, image, label, loss, lr

"""
def build_program():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()

    with paddle.static.program_guard(main_program, startup_program):
        with paddle.static.device_guard('gpu'):
            data = paddle.ones([4, 64], dtype='float32', name='data')
            # data = paddle.static.Print(data)

        # data -> [memcpy_h2d] -> data' -> [matmul] -> out ->[add] -> add_out
        with paddle.static.device_guard('gpu'):
            weight = paddle.randn([64, 64], name='weight')  # gpu
            weight.persistable = True
            # weight = paddle.static.Print(weight)
            matmul_out = paddle.matmul(data, weight, name='matmul_out')  # gpus
            bias = paddle.ones([4, 64], dtype='float32', name='bias')
            add_out = paddle.add(matmul_out, bias, name='add_out')
            # add_out, bias = wrap_cuda_graph(func1)(data)

        # add_out -> [memcpy_d2h] -> add_out' -> [sub] -> sub_out -> [tanh] -> tanh_out
        with paddle.static.device_guard('gpu'):
            sub_out = paddle.subtract(add_out, data, name='sub_out')
            tanh_out = paddle.tanh(sub_out, name='tanh_out')

        with paddle.static.device_guard('gpu'):
            bias_1 = paddle.add(bias, sub_out, name='bias_1')
            out_before = paddle.tanh(bias_1, name='out_before')
            out_last = paddle.subtract(tanh_out, data, name='out_last')

            out = paddle.add(out_before, out_last, name='out')
            mean = paddle.mean(out, name='mean_out')
            # mean = paddle.static.Print(mean)
            mean.persistable = True

    return main_program, startup_program, mean, weight
"""

class TestCustomStream(unittest.TestCase):
    """
    fill_constant(cpu)     gaussian_random
      |     |      |              |
      |     | matmul_v2(s1) fill_constant
      |     |      |              |    |
      |     |     elementwise_add(s1)  |
      |     |           |              |
      |  elementwise_sub(cpu)          |
      |     |           |              |
      |  tanh(cpu)     elementwise_add(s2)
      |     |                  |
    elementwise_sub(s1)      tanh(s2)
                 |             |
                elementwise_add(s2)
                        |
                  reduce_mean(s2)
    """

    def setUp(self):
        self.steps = 3
        if can_use_cuda_graph():
            paddle.set_flags(
                {
                    'FLAGS_allocator_strategy': 'auto_growth',
                    'FLAGS_sync_nccl_allreduce': False,
                    'FLAGS_cudnn_deterministic': True,
                    'FLAGS_use_stream_safe_cuda_allocator': True,
                }
            )

    def set_custom_stream(self, prog):
        op_index_for_stream1 = [2, 4, 9]
        op_index_for_stream2 = [7, 8, 10, 11]
        ops = prog.global_block().ops
        print("yoki: ops: ", ops)
        for op_index in op_index_for_stream1:
            ops[op_index].dist_attr.execution_stream = "s1"
            ops[op_index].dist_attr.stream_priority = 0
            print("yoki s1: ops[", op_index, "]: ", ops[op_index].type)
        for op_index in op_index_for_stream2:
            ops[op_index].dist_attr.execution_stream = "s2"
            ops[op_index].dist_attr.stream_priority = -1
            print("yoki s2: ops[", op_index, "]: ", ops[op_index].type)
        print("yoki: ops2: ", ops)

    def run_program(self, use_cuda_graph=False, apply_custom_stream=False):
        # paddle.seed(2022)
        seed = 100

        batch_size = 1
        class_num = 10
        image_shape = [batch_size, 784]
        label_shape = [batch_size, 1]

        paddle.seed(seed)
        np.random.seed(seed)

        main, startup, image, label, loss, lr = build_program(
            batch_size, class_num
        )

        # if apply_custom_stream:
        #     self.set_custom_stream(main)
        op_index_for_stream1 = [2, 4, 9]
        op_index_for_stream2 = [7, 8, 10, 11]
        ops = main.global_block().ops
        print("yoki: ops: ", ops)
        for op_index in op_index_for_stream1:
            ops[op_index].dist_attr.execution_stream = "s1"
            ops[op_index].dist_attr.stream_priority = 0
            print("yoki s1: ops[", op_index, "]: ", ops[op_index].type)
        for op_index in op_index_for_stream2:
            ops[op_index].dist_attr.execution_stream = "s2"
            ops[op_index].dist_attr.stream_priority = -1
            print("yoki s2: ops[", op_index, "]: ", ops[op_index].type)
        print("yoki: ops2: ", ops)

        place = paddle.CUDAPlace(0)
        exe = paddle.static.Executor(place)
        scope = paddle.static.Scope()
        with paddle.static.scope_guard(scope):
            exe.run(startup)
            # build_strategy = paddle.static.BuildStrategy()
            # build_strategy.allow_cuda_graph_capture = True
            # build_strategy.fix_op_run_order = True
            # build_strategy.fuse_all_optimizer_ops = True
            # compiled_program = paddle.static.CompiledProgram(
            #     main, build_strategy=build_strategy
            # )
            image_t = scope.var(image.name).get_tensor()
            label_t = scope.var(label.name).get_tensor()
            loss_t = scope.var(loss.name).get_tensor()
            lr_var = main.global_block().var(lr._var_name)
            self.assertTrue(lr_var.persistable)
            lr_t = scope.var(lr_var.name).get_tensor()
            cuda_graph = None
            outs = []
            paddle.set_flags({"FLAGS_new_executor_use_cuda_graph": True})
            for batch_id in range(20):
                use_feed_data = False
                image_np = np.random.rand(*image_shape).astype('float32')
                label_np = np.random.randint(
                    low=0, high=class_num, size=label_shape, dtype='int64'
                )
                if not use_feed_data:
                    image_t.set(image_np, place)
                    label_t.set(label_np, place)

                if batch_id == 1 and use_cuda_graph:
                    cuda_graph = CUDAGraph(place, mode="global")
                    cuda_graph.capture_begin()
                    # exe.run(compiled_program)
                    exe.run(main)
                    cuda_graph.capture_end()

                if cuda_graph:
                    lr_t.set(np.array([lr()], dtype='float32'), place)
                    cuda_graph.replay()
                else:
                    # if use_feed_data:
                    #     exe.run(
                    #         compiled_program,
                    #         feed={'image': image_np, 'label': label_np},
                    #     )
                    # else:
                    #     exe.run(compiled_program)
                    exe.run(main)
                outs.append(np.array(loss_t))
                print("yoki: ", np.array(loss_t))
                lr.step()
            if cuda_graph:
                cuda_graph.reset()
        return outs
        """
        # main_program, startup_program, loss, weight = build_program()
        self.assertEqual(len(startup_program.global_block().ops), 0)

        if apply_custom_stream:
            self.set_custom_stream(main_program)

        # if use_cuda_graph:
        #     section_programs = cuda_graph_transform(main_program)

        place = paddle.CUDAPlace(0)
        exe = paddle.static.Executor(place)
        scope = paddle.static.Scope()
        with paddle.static.scope_guard(scope):
            with paddle.static.program_guard(main_program, startup_program):
                # build_strategy = paddle.static.BuildStrategy()
                # build_strategy.allow_cuda_graph_capture = True
                # # build_strategy.fix_op_run_order = True
                # # build_strategy.fuse_all_optimizer_ops = True
                # compiled_program = paddle.static.CompiledProgram(
                #     main_program
                # ).with_data_parallel(
                #     loss_name=loss.name, build_strategy=build_strategy, places=place
                # )
                loss_t = scope.var(loss.name).get_tensor()
                weight_t = scope.var(weight.name).get_tensor()

                cuda_graph = None
                outs = []
                paddle.set_flags({"FLAGS_new_executor_use_cuda_graph": True})
                for i in range(self.steps):
                    if i == 1 and use_cuda_graph:
                        # paddle.set_flags({"FLAGS_new_executor_use_cuda_graph": True})
                        cuda_graph = CUDAGraph(place, mode="global")
                        cuda_graph.capture_begin(use_multi_stream=True)
                        exe.run(main_program)
                        cuda_graph.capture_end()

                    if cuda_graph:
                        cuda_graph.replay()
                    else:
                        exe.run(main_program)
                    # exe.run(main_program)

                    outs.append(np.array(loss_t))
                    print("yoki: ", np.array(weight_t))
                    print("yoki: ", np.array(loss_t))
                if cuda_graph:
                    cuda_graph.reset()
        return outs
        """

    def test_result(self):
        if not can_use_cuda_graph():
            return

        outs_baselines = self.run_program(use_cuda_graph=False, apply_custom_stream=False)
        # outs_stream = self.run_program(use_cuda_graph=False, apply_custom_stream=True)
        # outs_cuda_graph = self.run_program(use_cuda_graph=True, apply_custom_stream=False)
        outs_cuda_graph_stream = self.run_program(use_cuda_graph=True, apply_custom_stream=True)
        for bl, out in zip(outs_baselines, outs_cuda_graph_stream):
            self.assertEqual(bl[0], out[0])


if __name__ == "__main__":
    unittest.main()
