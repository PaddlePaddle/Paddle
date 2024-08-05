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

import os

os.environ['FLAGS_use_stream_safe_cuda_allocator'] = "true"
import json
import shutil
import unittest

import numpy as np
from utils import static_guard

import paddle
from paddle.base import core
from paddle.profiler import profiler

paddle.enable_static()


def build_program():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()

    with paddle.static.program_guard(main_program, startup_program):
        with paddle.static.device_guard('cpu'):
            data = paddle.ones([4, 64], dtype='float32', name='data')

        # data -> [memcpy_h2d] -> data' -> [matmul] -> out ->[add] -> add_out
        with paddle.static.device_guard('gpu'):
            weight = paddle.randn([64, 64], name='weight')  # gpu
            matmul_out = paddle.matmul(data, weight, name='matmul_out')  # gpus
            bias = paddle.ones([4, 64], dtype='float32', name='bias')
            add_out = paddle.add(matmul_out, bias, name='add_out')

        # add_out -> [memcpy_d2h] -> add_out' -> [sub] -> sub_out -> [tanh] -> tanh_out
        with paddle.static.device_guard('cpu'):
            sub_out = paddle.subtract(add_out, data, name='sub_out')
            tanh_out = paddle.tanh(sub_out, name='tanh_out')

        with paddle.static.device_guard('gpu'):
            bias_1 = paddle.add(bias, sub_out, name='bias_1')
            out_before = paddle.tanh(bias_1, name='out_before')
            out_last = paddle.subtract(tanh_out, data, name='out_last')

            out = paddle.add(out_before, out_last, name='out')
            mean = paddle.mean(out, name='mean_out')

    return main_program, startup_program, [mean]


class ExecutorStatisticsTestCase(unittest.TestCase):
    def setUp(self):
        self.iter_n = 3
        self.place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        self.perf_path = './perfstat'

    def test_executor_statistics(self):
        self.run_with_statistics(executor='Executor')

    def test_standalone_executor_statistics(self):
        self.run_with_statistics(executor='StandaloneExecutor')

    def run_with_statistics(self, executor=None):
        # random failed, skip this testcase
        return
        if os.getenv("FLAGS_static_executor_perfstat_filepath") is None:
            return
        paddle.seed(2020)
        # note: startup program is empty
        main_program, startup_program, fetch_list = build_program()

        scope = paddle.static.Scope()
        with paddle.static.scope_guard(scope):
            exe = paddle.static.Executor(self.place)
            helper_profiler = profiler.Profiler(
                targets=[profiler.ProfilerTarget.CPU], scheduler=(1, 2)
            )
            helper_profiler.start()
            for i in range(self.iter_n):
                exe.run(main_program, fetch_list=fetch_list)
                helper_profiler.step()
            helper_profiler.stop()

        self.assertTrue(os.path.exists(self.perf_path))
        with open(self.perf_path, 'r') as load_f:
            stat_res = json.load(load_f)
            self.assertTrue(len(stat_res) > 0)

        os.remove(self.perf_path)
        shutil.rmtree('./profiler_log')


class MultiStreamModelTestCase(unittest.TestCase):
    def setUp(self):
        self.iter_n = 2
        self.place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_result(self):
        ground_truths = self.run_test(False)
        res = self.run_test(True)

        for gt, out in zip(ground_truths, res):
            self.assertEqual(gt[0], out[0])

    def run_test(self, use_new_executor=True):
        paddle.seed(2020)
        main_program, startup_program, fetch_list = build_program()

        scope = core.Scope()
        exe = paddle.static.Executor(self.place)
        outs = []
        for i in range(self.iter_n):
            outs.append(
                exe.run(main_program, scope=scope, fetch_list=fetch_list)
            )
        print(outs)
        return outs


class SwitchExecutorInterfaceWithFeed(unittest.TestCase):
    def setUp(self):
        self.place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        self.iter_run = 2

    def build_program(self, is_double=False):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            a = paddle.static.data(name="a", shape=[2, 2], dtype='float32')
            b = paddle.ones([2, 2]) * 2
            t = paddle.static.nn.fc(a, 2)
            c = t + b
            if is_double:
                c = c + c

        return main_program, startup_program, [c]

    def _run(
        self,
        feed,
        use_str=False,
        is_double=False,
        add_wrong_fetch=False,
        use_compiled=False,
    ):
        paddle.seed(2020)

        main_program, startup_program, fetch_vars = self.build_program(
            is_double
        )

        exe = paddle.static.Executor(self.place)
        exe.run(startup_program)

        if use_compiled:
            main_program = paddle.static.CompiledProgram(main_program)

        if (
            use_str and not paddle.framework.in_pir_mode()
        ):  # test for fetch name
            fetch_vars = [x.name for x in fetch_vars]
        if add_wrong_fetch:  # test for wrong fetch type
            fetch_vars.append(1123)
        outs = []
        for i in range(self.iter_run):
            out = exe.run(main_program, feed=feed, fetch_list=fetch_vars)[0]

            outs.append(out)

        return outs

    def run_dygraph(self, feed):
        def run_once(is_double):
            paddle.seed(2020)
            a = feed['a']
            a = paddle.to_tensor(a, dtype='float32')
            b = paddle.ones([2, 2]) * 2
            t = paddle.nn.Linear(2, 2)(a)
            c = t + b
            if is_double:
                c = c + c
            return c.numpy()

        out1 = []
        for i in range(self.iter_run):
            out1.append(run_once(False))
        out2 = []
        for i in range(self.iter_run):
            out2.append(run_once(True))
        return [out1, out2]

    def run_new_executor(self, feed, use_compiled=False):
        # run construct program 1
        out1 = self._run(
            feed, use_str=False, is_double=False, use_compiled=use_compiled
        )
        # run construct program 2 with same executor
        out2 = self._run(
            feed, use_str=True, is_double=True, use_compiled=use_compiled
        )

        return [out1, out2]

    def test_with_feed(self):
        data = np.ones([2, 2], dtype="float32")
        feed = {"a": data, 'fake_input': data}

        with static_guard():
            res = self.run_new_executor(feed)
        with paddle.base.dygraph.guard():
            gt = self.run_dygraph(feed)
        for x, y in zip(gt, res):
            np.testing.assert_array_equal(x, y)

    def test_with_error(self):
        feed = [{'a': np.ones([2, 2], dtype="float32")}]

        with self.assertRaises(TypeError):
            self._run(feed[0], add_wrong_fetch=True)

    def test_empty_program(self):
        program = paddle.static.Program()
        exe = paddle.static.Executor(self.place)
        for i in range(10):
            out = exe.run()  # old executor

        for i in range(10):
            print(i, flush=1)
            out = exe.run(program, feed=None)


class TestException(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CPUPlace()
        self.fetch_vars = None

    def build_program(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            w = paddle.rand([10, 3])
            ids = paddle.static.data(name="id", shape=[5], dtype='int64')
            data = paddle.static.data(name="data", shape=[3], dtype='float32')
            emb = paddle.nn.functional.embedding(
                x=ids, weight=w, sparse=False, name="embedding"
            )
            emb = emb + data

        return main_program, startup_program, emb

    def _run(self, feeds):
        paddle.seed(2020)

        main_program, startup_program, fetch_vars = self.build_program()

        exe = paddle.static.Executor(self.place)
        exe.run(startup_program)

        for feed in feeds:
            out = exe.run(main_program, feed=feed, fetch_list=fetch_vars)
        self.fetch_vars = fetch_vars
        return out

    def run_new_executor(self, feed):
        out = self._run(feed)
        return out

    def test_exception(self):
        feed = [
            {
                'id': np.array([1, 2, 3, 4, 5]).astype(np.int64),
                'data': np.array([1, 2, 3]).astype(np.float32),
            },
            {
                'id': np.array([1, 2, 3, 4, 11]).astype(np.int64),
                'data': np.array([1, 2, 3]).astype(np.float32),
            },
        ]
        self.assertRaises(ValueError, self.run_new_executor, feed)

    def test_nan(self):
        flags = {'FLAGS_check_nan_inf': True, 'FLAGS_benchmark': True}
        paddle.base.set_flags(flags)
        feed = [
            {
                'id': np.array([1, 2, 3, 4, 5]).astype(np.int64),
                'data': np.array([1, 2, 3]).astype(np.float32),
            },
            {
                'id': np.array([1, 2, 3, 4, 5]).astype(np.int64),
                'data': np.array([1, 2, 3]).astype(np.float32),
            },
        ]
        feed[1]['data'][0] = np.nan
        self.assertRaises(RuntimeError, self.run_new_executor, feed)

    def test_scope_find_temp_var(self):
        feed = [
            {
                'id': np.array([1, 2, 3, 4, 5]).astype(np.int64),
                'data': np.array([1, 2, 3]).astype(np.float32),
            },
            {
                'id': np.array([1, 2, 3, 4, 5]).astype(np.int64),
                'data': np.array([2, 2, 2]).astype(np.float32),
            },
        ]
        self.run_new_executor(feed)
        if not paddle.framework.in_pir_mode():
            self.assertIsNone(
                paddle.static.global_scope().find_var(self.fetch_vars.name)
            )


class TestFetchEmptyTensor(unittest.TestCase):
    def test_fetch(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.base.core.is_compiled_with_cuda()
        ):
            places.append(paddle.CPUPlace())
        if paddle.base.core.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        for place in places:
            with paddle.static.program_guard(paddle.static.Program()):
                out = paddle.empty([3, 0])
                exe = paddle.static.Executor(place)
                res = exe.run(fetch_list=[out])
            self.assertEqual(res[0].shape, (3, 0))


class TestInplaceApiWithDataTransform(unittest.TestCase):
    def test_increment(self):
        if paddle.base.core.is_compiled_with_cuda():
            with paddle.base.device_guard("gpu:0"):
                x = paddle.tensor.fill_constant([1], "float32", 0)
            with paddle.base.device_guard("cpu"):
                x = paddle.increment(x)
            exe = paddle.static.Executor(paddle.CUDAPlace(0))
            for i in range(10):
                (a,) = exe.run(
                    paddle.static.default_main_program(), fetch_list=[x]
                )
                self.assertEqual(a[0], 1)


if __name__ == "__main__":
    unittest.main()
