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
import sys
import unittest
import paddle
from paddle.fluid import core
from paddle.fluid.core import StandaloneExecutor

import numpy as np

paddle.enable_static()


class LinearTestCase(unittest.TestCase):
    def setUp(self):
        place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else paddle.CPUPlace()
        self.place = core.Place()
        self.place.set_place(place)

    def build_program(self):
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            a = paddle.static.data(name="a", shape=[2, 2], dtype='float32')
            b = paddle.ones([2, 2]) * 2
            t = paddle.static.nn.fc(a, 2)
            c = t + b

        return startup_program, main_program, c

    def test_interp_base(self):
        startup_program, main_program, c = self.build_program()
        standaloneexecutor = StandaloneExecutor(
            self.place, startup_program.desc, main_program.desc, core.Scope())
        out = standaloneexecutor.run({
            "a": np.ones(
                [2, 2], dtype="float32") * 2
        }, [c.name])
        for i in range(10):
            out = standaloneexecutor.run({
                "a": np.ones(
                    [2, 2], dtype="float32") * i
            }, [c.name])

        for i in range(10):
            out = standaloneexecutor.run({
                "a": np.ones(
                    [2, 2], dtype="float32") * i
            }, ['a', c.name])

    def test_dry_run(self):
        startup_program, main_program, c = self.build_program()
        standaloneexecutor = StandaloneExecutor(
            self.place, startup_program.desc, main_program.desc, core.Scope())
        # test for cost_info
        cost_info = standaloneexecutor.dry_run({
            "a": np.ones(
                [2, 2], dtype="float32")
        })
        self.check_cost_info(cost_info)

    def check_cost_info(self, cost_info):
        IS_WINDOWS = sys.platform.startswith('win')

        if core.is_compiled_with_cuda():
            # # w,bias,b, out, memory block is at least 256 bytes on Linux
            gt = 16 * 4 if IS_WINDOWS else 256 * 4
            self.assertGreater(cost_info.device_memory_bytes(), gt)
        else:
            self.assertEqual(cost_info.device_memory_bytes(), 0)


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


class MultiStreamModelTestCase(unittest.TestCase):
    def setUp(self):
        self.iter_n = 2
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else paddle.CPUPlace()

    def test_result(self):
        ground_truths = self.run_raw_executor()
        res = self.run_new_executor()

        for gt, out in zip(ground_truths, res):
            self.assertEqual(gt[0], out[0])

        res_sequential = self.run_new_executor_sequential()
        for gt, out in zip(ground_truths, res_sequential):
            self.assertEqual(gt[0], out[0])

    def run_raw_executor(self):
        paddle.seed(2020)
        main_program, startup_program, fetch_list = build_program()

        exe = paddle.static.Executor(self.place)
        exe.run(startup_program)

        outs = []
        for i in range(self.iter_n):
            outs.append(exe.run(main_program, fetch_list=fetch_list))

        return outs

    def run_new_executor(self):
        paddle.seed(2020)
        main_program, startup_program, fetch_list = build_program()
        fetch_list = [x.name for x in fetch_list]

        p = core.Place()
        p.set_place(self.place)
        inter_core = StandaloneExecutor(p, startup_program.desc,
                                        main_program.desc, core.Scope())
        outs = []
        for i in range(self.iter_n):
            outs.append(
                np.array(inter_core.run({}, fetch_list)._move_to_list()[0]))
        return outs

    def run_new_executor_sequential(self):
        os.environ['FLAGS_new_executor_sequential_run'] = '1'
        res = self.run_new_executor()
        del os.environ['FLAGS_new_executor_sequential_run']
        return res


class SwitchExecutorInterfaceTestCase(MultiStreamModelTestCase):
    def run_new_executor(self):
        paddle.seed(2020)
        os.environ['FLAGS_USE_STANDALONE_EXECUTOR'] = '1'
        main_program, startup_program, fetch_list = build_program()
        exe = paddle.static.Executor(self.place)
        exe.run(startup_program)

        outs = []
        for i in range(self.iter_n):
            outs.append(exe.run(main_program, fetch_list=fetch_list))

        del os.environ['FLAGS_USE_STANDALONE_EXECUTOR']

        return outs


class SwitchExecutorInterfaceWithFeed(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else paddle.CPUPlace()
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

    def _run(self,
             feed,
             use_str=False,
             is_double=False,
             add_wrong_fetch=False,
             use_compiled=False):
        paddle.seed(2020)

        main_program, startup_program, fetch_vars = self.build_program(
            is_double)

        exe = paddle.static.Executor(self.place)
        exe.run(startup_program)

        if use_compiled:
            main_program = paddle.static.CompiledProgram(
                main_program).with_data_parallel(
                    fetch_vars[0].name, places=[self.place])

        if use_str:  # test for fetch name
            fetch_vars = [x.name for x in fetch_vars]
        if add_wrong_fetch:  # test for wrong fetch type
            fetch_vars.append(1123)
        outs = []
        for i in range(self.iter_run):
            out = exe.run(main_program, feed=feed, fetch_list=fetch_vars)[0]

            outs.append(out)

        return outs

    def run_raw_executor(self, feed, use_compiled=False):
        # run construct program 1
        out1 = self._run(
            feed, use_str=False, is_double=False, use_compiled=use_compiled)
        # run construct program 2 with same executor
        out2 = self._run(
            feed, use_str=True, is_double=True, use_compiled=use_compiled)

        return [out1, out2]

    def run_new_executor(self, feed, use_compiled=False):
        os.environ['FLAGS_USE_STANDALONE_EXECUTOR'] = '1'
        out = self.run_raw_executor(feed, use_compiled=use_compiled)
        del os.environ['FLAGS_USE_STANDALONE_EXECUTOR']
        return out

    def test_with_feed(self):
        data = np.ones([2, 2], dtype="float32")
        feed = {"a": data, 'fake_input': data}

        res = self.run_new_executor(feed)
        gt = self.run_raw_executor(feed)
        for x, y in zip(gt, res):
            self.assertTrue(np.array_equal(x, y))

    def test_with_error(self):
        feed = [{'a': np.ones([2, 2], dtype="float32")}]

        with self.assertRaises(TypeError):
            os.environ['FLAGS_USE_STANDALONE_EXECUTOR'] = '1'
            self._run(feed[0], add_wrong_fetch=True)
            del os.environ['FLAGS_USE_STANDALONE_EXECUTOR']

    def test_compiled_program(self):
        data = np.ones([2, 2], dtype="float32")
        feed = {"a": data}

        res = self.run_new_executor(feed, use_compiled=True)
        gt = self.run_raw_executor(feed, use_compiled=True)
        for x, y in zip(gt, res):
            self.assertTrue(np.array_equal(x, y))


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
                x=ids, weight=w, sparse=False, name="embedding")
            emb = emb + data

        return main_program, startup_program, emb

    def _run(self, feeds):
        paddle.seed(2020)

        main_program, startup_program, fetch_vars = self.build_program()

        exe = paddle.static.Executor(self.place)
        exe.run(startup_program)

        for feed in feeds:
            out = exe.run(main_program, feed=feed, fetch_list=fetch_vars)
        print(main_program)
        self.fetch_vars = fetch_vars
        return out

    def run_new_executor(self, feed):
        os.environ['FLAGS_USE_STANDALONE_EXECUTOR'] = '1'
        out = self._run(feed)
        del os.environ['FLAGS_USE_STANDALONE_EXECUTOR']
        return out

    def test_exception(self):
        feed = [{
            'id': np.array([1, 2, 3, 4, 5]).astype(np.int64),
            'data': np.array([1, 2, 3]).astype(np.float32),
        }, {
            'id': np.array([1, 2, 3, 4, 11]).astype(np.int64),
            'data': np.array([1, 2, 3]).astype(np.float32),
        }]
        self.assertRaises(ValueError, self.run_new_executor, feed)

    def test_nan(self):
        flags = {'FLAGS_check_nan_inf': True, 'FLAGS_benchmark': True}
        paddle.fluid.set_flags(flags)
        feed = [{
            'id': np.array([1, 2, 3, 4, 5]).astype(np.int64),
            'data': np.array([1, 2, 3]).astype(np.float32),
        }, {
            'id': np.array([1, 2, 3, 4, 5]).astype(np.int64),
            'data': np.array([1, 2, 3]).astype(np.float32),
        }]
        feed[1]['data'][0] = np.nan
        self.assertRaises(RuntimeError, self.run_new_executor, feed)

    def test_scope_find_temp_var(self):
        feed = [{
            'id': np.array([1, 2, 3, 4, 5]).astype(np.int64),
            'data': np.array([1, 2, 3]).astype(np.float32),
        }, {
            'id': np.array([1, 2, 3, 4, 5]).astype(np.int64),
            'data': np.array([2, 2, 2]).astype(np.float32),
        }]
        self.run_new_executor(feed)
        self.assertIsNone(paddle.static.global_scope().find_var(
            self.fetch_vars.name))


if __name__ == "__main__":
    unittest.main()
