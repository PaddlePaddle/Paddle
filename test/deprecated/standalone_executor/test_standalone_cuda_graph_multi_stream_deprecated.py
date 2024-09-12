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


import sys
import unittest

import numpy as np

sys.path.append("../../legacy_test")
from test_cuda_graph_static_mode import build_program

import paddle
from paddle.device.cuda.graphs import CUDAGraph

paddle.enable_static()


def can_use_cuda_graph():
    return paddle.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm()


@unittest.skipIf(
    not paddle.is_compiled_with_cuda() or float(paddle.version.cuda()) < 11.0,
    "only support cuda >= 11.0",
)
class TestCustomStream(unittest.TestCase):
    def setUp(self):
        self.steps = 10
        if can_use_cuda_graph():
            paddle.set_flags(
                {
                    'FLAGS_allocator_strategy': 'auto_growth',
                    'FLAGS_sync_nccl_allreduce': False,
                    'FLAGS_cudnn_deterministic': True,
                    'FLAGS_use_stream_safe_cuda_allocator': True,
                    'FLAGS_new_executor_use_cuda_graph': True,
                }
            )

    def set_custom_stream(self, prog):
        op_index_for_stream1 = [2, 4, 9]
        op_index_for_stream2 = [7, 8, 10, 11]
        ops = prog.global_block().ops
        for op_index in op_index_for_stream1:
            ops[op_index].dist_attr.execution_stream = "s1"
            ops[op_index].dist_attr.stream_priority = 0
        for op_index in op_index_for_stream2:
            ops[op_index].dist_attr.execution_stream = "s2"
            ops[op_index].dist_attr.stream_priority = -1

    def run_program(self, use_cuda_graph=False, apply_custom_stream=False):
        seed = 100

        batch_size = 1
        class_num = 10
        image_shape = [batch_size, 784]
        label_shape = [batch_size, 1]

        paddle.seed(seed)
        np.random.seed(seed)
        startup = paddle.static.Program()
        main = paddle.static.Program()
        image, label, loss, lr = build_program(
            main, startup, batch_size, class_num
        )

        if apply_custom_stream:
            self.set_custom_stream(main)

        place = paddle.CUDAPlace(0)
        exe = paddle.static.Executor(place)
        scope = paddle.static.Scope()
        with paddle.static.scope_guard(scope):
            exe.run(startup)
            image_t = scope.var(image.name).get_tensor()
            label_t = scope.var(label.name).get_tensor()
            loss_t = scope.var(loss.name).get_tensor()
            lr_var = main.global_block().var(lr._var_name)
            self.assertTrue(lr_var.persistable)
            lr_t = scope.var(lr_var.name).get_tensor()
            cuda_graph = None
            outs = []
            for batch_id in range(20):
                image_np = np.random.rand(*image_shape).astype('float32')
                label_np = np.random.randint(
                    low=0, high=class_num, size=label_shape, dtype='int64'
                )
                image_t.set(image_np, place)
                label_t.set(label_np, place)

                if batch_id == 1 and use_cuda_graph:
                    cuda_graph = CUDAGraph(place, mode="global")
                    cuda_graph.capture_begin()
                    exe.run(main)
                    cuda_graph.capture_end()

                if cuda_graph:
                    lr_t.set(np.array([lr()], dtype='float32'), place)
                    cuda_graph.replay()
                else:
                    exe.run(main)
                outs.append(np.array(loss_t))
                lr.step()
            if cuda_graph:
                cuda_graph.reset()
        return outs

    def test_result(self):
        if not can_use_cuda_graph():
            return

        outs = []
        for use_cuda_graph in [False, True]:
            for apply_custom_stream in [False, True]:
                out = self.run_program(use_cuda_graph, apply_custom_stream)
                outs.append(out)

        for out in outs:
            for baseline, result in zip(outs[0], out):
                self.assertEqual(baseline, result)


if __name__ == "__main__":
    unittest.main()
