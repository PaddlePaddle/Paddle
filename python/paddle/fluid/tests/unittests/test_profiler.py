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

import os
import tempfile
import unittest

import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.layers as layers
import paddle.fluid.profiler as profiler
import paddle.fluid.proto.profiler.profiler_pb2 as profiler_pb2
import paddle.utils as utils
from paddle.utils.flops import flops


class TestProfiler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ['CPU_NUM'] = str(4)

    def build_program(self, compile_program=True):
        startup_program = fluid.Program()
        main_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            image = fluid.layers.data(name='x', shape=[784], dtype='float32')
            hidden1 = fluid.layers.fc(input=image, size=64, act='relu')
            i = layers.zeros(shape=[1], dtype='int64')
            counter = fluid.layers.zeros(
                shape=[1], dtype='int64', force_cpu=True
            )
            until = layers.fill_constant([1], dtype='int64', value=10)
            data_arr = paddle.tensor.array_write(hidden1, i)
            cond = paddle.less_than(x=counter, y=until)
            while_op = paddle.static.nn.control_flow.While(cond=cond)
            with while_op.block():
                hidden_n = fluid.layers.fc(input=hidden1, size=64, act='relu')
                paddle.tensor.array_write(hidden_n, i, data_arr)
                paddle.increment(x=counter, value=1)
                paddle.assign(paddle.less_than(x=counter, y=until), cond)

            hidden_n = paddle.tensor.array_read(data_arr, i)
            hidden2 = fluid.layers.fc(input=hidden_n, size=64, act='relu')
            predict = fluid.layers.fc(input=hidden2, size=10, act='softmax')
            label = fluid.layers.data(name='y', shape=[1], dtype='int64')
            cost = paddle.nn.functional.cross_entropy(
                input=predict, label=label, reduction='none', use_softmax=False
            )
            avg_cost = paddle.mean(cost)
            batch_size = paddle.tensor.create_tensor(dtype='int64')
            batch_acc = paddle.static.accuracy(
                input=predict, label=label, total=batch_size
            )

        optimizer = fluid.optimizer.Momentum(learning_rate=0.001, momentum=0.9)
        opts = optimizer.minimize(avg_cost, startup_program=startup_program)

        if compile_program:
            # TODO(luotao): profiler tool may have bug with multi-thread parallel executor.
            # https://github.com/PaddlePaddle/Paddle/pull/25200#issuecomment-650483092
            exec_strategy = fluid.ExecutionStrategy()
            exec_strategy.num_threads = 1
            train_program = fluid.compiler.CompiledProgram(
                main_program
            ).with_data_parallel(
                loss_name=avg_cost.name, exec_strategy=exec_strategy
            )
        else:
            train_program = main_program
        return train_program, startup_program, avg_cost, batch_size, batch_acc

    def get_profile_path(self):
        profile_path = os.path.join(tempfile.gettempdir(), "profile")
        open(profile_path, "w").write("")
        return profile_path

    def check_profile_result(self, profile_path):
        data = open(profile_path, 'rb').read()
        if len(data) > 0:
            profile_pb = profiler_pb2.Profile()
            profile_pb.ParseFromString(data)
            self.assertGreater(len(profile_pb.events), 0)
            for event in profile_pb.events:
                if event.type == profiler_pb2.Event.GPUKernel:
                    if not event.detail_info and not event.name.startswith(
                        "MEM"
                    ):
                        raise Exception(
                            "Kernel %s missing event. Has this kernel been recorded by RecordEvent?"
                            % event.name
                        )
                elif event.type == profiler_pb2.Event.CPU and (
                    event.name.startswith("Driver API")
                    or event.name.startswith("Runtime API")
                ):
                    print("Warning: unregister", event.name)

    def run_iter(self, exe, main_program, fetch_list):
        x = np.random.random((32, 784)).astype("float32")
        y = np.random.randint(0, 10, (32, 1)).astype("int64")
        outs = exe.run(
            main_program, feed={'x': x, 'y': y}, fetch_list=fetch_list
        )

    def net_profiler(
        self,
        exe,
        state,
        tracer_option,
        batch_range=None,
        use_parallel_executor=False,
        use_new_api=False,
    ):
        (
            main_program,
            startup_program,
            avg_cost,
            batch_size,
            batch_acc,
        ) = self.build_program(compile_program=use_parallel_executor)
        exe.run(startup_program)

        profile_path = self.get_profile_path()
        if not use_new_api:
            with profiler.profiler(state, 'total', profile_path, tracer_option):
                for iter in range(10):
                    if iter == 2:
                        profiler.reset_profiler()
                    self.run_iter(
                        exe, main_program, [avg_cost, batch_acc, batch_size]
                    )
        else:
            options = utils.ProfilerOptions(
                options={
                    'state': state,
                    'sorted_key': 'total',
                    'tracer_level': tracer_option,
                    'batch_range': [0, 10]
                    if batch_range is None
                    else batch_range,
                    'profile_path': profile_path,
                }
            )
            with utils.Profiler(enabled=True, options=options) as prof:
                for iter in range(10):
                    self.run_iter(
                        exe, main_program, [avg_cost, batch_acc, batch_size]
                    )
                    utils.get_profiler().record_step()
                    if batch_range is None and iter == 2:
                        utils.get_profiler().reset()
        # TODO(luotao): check why nccl kernel in profile result.
        # https://github.com/PaddlePaddle/Paddle/pull/25200#issuecomment-650483092
        # self.check_profile_result(profile_path)

    def test_cpu_profiler(self):
        exe = fluid.Executor(fluid.CPUPlace())
        for use_new_api in [False, True]:
            self.net_profiler(
                exe,
                'CPU',
                "Default",
                batch_range=[5, 10],
                use_new_api=use_new_api,
            )

    @unittest.skipIf(
        not core.is_compiled_with_cuda(), "profiler is enabled only with GPU"
    )
    def test_cuda_profiler(self):
        exe = fluid.Executor(fluid.CUDAPlace(0))
        for use_new_api in [False, True]:
            self.net_profiler(
                exe,
                'GPU',
                "OpDetail",
                batch_range=[0, 10],
                use_new_api=use_new_api,
            )

    @unittest.skipIf(
        not core.is_compiled_with_cuda(), "profiler is enabled only with GPU"
    )
    def test_all_profiler(self):
        exe = fluid.Executor(fluid.CUDAPlace(0))
        for use_new_api in [False, True]:
            self.net_profiler(
                exe,
                'All',
                "AllOpDetail",
                batch_range=None,
                use_new_api=use_new_api,
            )


class TestProfilerAPIError(unittest.TestCase):
    def test_errors(self):
        options = utils.ProfilerOptions()
        self.assertIsNone(options['profile_path'])
        self.assertIsNone(options['timeline_path'])

        options = options.with_state('All')
        self.assertTrue(options['state'] == 'All')
        try:
            print(options['test'])
        except ValueError:
            pass

        global_profiler = utils.get_profiler()
        with utils.Profiler(enabled=True) as prof:
            self.assertTrue(utils.get_profiler() == prof)
            self.assertTrue(global_profiler != prof)


class TestFLOPSAPI(unittest.TestCase):
    def test_flops(self):
        self.assertTrue(flops('relu', {'X': [[12, 12]]}, {'output': 4}) == 144)
        self.assertTrue(flops('dropout', {}, {'output': 4}) == 0)
        self.assertTrue(
            flops(
                'transpose2',
                {
                    'X': [[12, 12, 12]],
                },
                {},
            )
            == 0
        )
        self.assertTrue(
            flops(
                'reshape2',
                {
                    'X': [[12, 12, 12]],
                },
                {},
            )
            == 0
        )
        self.assertTrue(
            flops(
                'unsqueeze2',
                {
                    'X': [[12, 12, 12]],
                },
                {},
            )
            == 0
        )
        self.assertTrue(
            flops(
                'layer_norm',
                {'Bias': [[128]], 'Scale': [[128]], 'X': [[32, 128, 28, 28]]},
                {'epsilon': 0.01},
            )
            == 32 * 128 * 28 * 28 * 8
        )
        self.assertTrue(
            flops(
                'elementwise_add', {'X': [[12, 12, 12]], 'Y': [[2, 2, 12]]}, {}
            )
            == 12 * 12 * 12
        )
        self.assertTrue(
            flops('gelu', {'X': [[12, 12, 12]]}, {}) == 5 * 12 * 12 * 12
        )
        self.assertTrue(
            flops(
                'matmul',
                {'X': [[3, 12, 12, 8]], 'Y': [[12, 12, 8]]},
                {'transpose_X': False, 'transpose_Y': True},
            )
            == 3 * 12 * 12 * 12 * 2 * 8
        )
        self.assertTrue(
            flops(
                'matmul_v2',
                {'X': [[3, 12, 12, 8]], 'Y': [[12, 12, 8]]},
                {'trans_x': False, 'trans_y': True},
            )
            == 3 * 12 * 12 * 12 * 2 * 8
        )
        self.assertTrue(
            flops('relu', {'X': [[12, 12, 12]]}, {}) == 12 * 12 * 12
        )
        self.assertTrue(
            flops('softmax', {'X': [[12, 12, 12]]}, {}) == 3 * 12 * 12 * 12
        )
        self.assertTrue(
            flops('c_embedding', {'Ids': [[12, 12]], 'W': [[12, 12, 3]]}, {})
            == 0
        )


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
