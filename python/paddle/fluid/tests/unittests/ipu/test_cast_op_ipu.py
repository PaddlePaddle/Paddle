#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np
import paddle
import paddle.static
from paddle.fluid.tests.unittests.ipu.op_test_ipu import IPUOpTest


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestBase(IPUOpTest):
    def setUp(self):
        self.set_atol()
        self.set_training()
        self.set_data_feed()
        self.set_feed_attr()
        self.set_op_attrs()

    def set_atol(self):
        self.atol = 1e-3

    def set_data_feed(self):
        self.feed = {
            "x": np.random.uniform(size=[1, 3, 3, 3]).astype('float32'),
        }

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed.values()]
        self.feed_list = list(self.feed.keys())
        self.feed_dtype = [x.dtype for x in self.feed.values()]

    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['dtype'] = 'float16'

    def _test_base(self, run_ipu=True):
        scope = paddle.static.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = self.SEED
        startup_prog.random_seed = self.SEED

        with paddle.static.scope_guard(scope):
            with paddle.static.program_guard(main_prog, startup_prog):
                x = paddle.static.data(
                    name=self.feed_list[0],
                    shape=self.feed_shape[0],
                    dtype=self.feed_dtype[0])
                out = paddle.cast(x, **self.attrs)
                fetch_list = [out.name]

            if run_ipu:
                place = paddle.IPUPlace()
            else:
                place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(startup_prog)

            if run_ipu:
                feed_list = self.feed_list
                ipu_strategy = paddle.static.IpuStrategy()
                ipu_strategy.set_graph_config(is_training=self.is_training)
                program = paddle.static.IpuCompiledProgram(
                    main_prog,
                    ipu_strategy=ipu_strategy).compile(feed_list, fetch_list)
            else:
                program = main_prog

            result = exe.run(program, feed=self.feed, fetch_list=fetch_list)
            return result[0]

    def test_base(self):
        res0 = self._test_base(True)
        res1 = self._test_base(False)

        self.assertTrue(
            np.allclose(
                res0.flatten(), res1.flatten(), atol=self.atol))

        self.assertTrue(res0.shape == res1.shape)


class TestEnableFp16(TestBase):
    def set_atol(self):
        self.atol = 1e-10

    def set_data_feed(self):
        self.feed = {"x": np.array([1, 200, 3000, 40000]).astype('int32'), }

    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['dtype'] = 'float32'

    def _test_base(self, run_ipu=True):
        scope = paddle.static.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = self.SEED
        startup_prog.random_seed = self.SEED

        with paddle.static.scope_guard(scope):
            with paddle.static.program_guard(main_prog, startup_prog):
                x = paddle.static.data(
                    name=self.feed_list[0],
                    shape=self.feed_shape[0],
                    dtype=self.feed_dtype[0])
                out = paddle.cast(x, **self.attrs)
                fetch_list = [out.name]

            if run_ipu:
                place = paddle.IPUPlace()
            else:
                place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(startup_prog)

            if run_ipu:
                feed_list = self.feed_list
                ipu_strategy = paddle.static.IpuStrategy()
                ipu_strategy.set_graph_config(is_training=self.is_training)
                ipu_strategy.set_precision_config(enable_fp16=True)
                program = paddle.static.IpuCompiledProgram(
                    main_prog,
                    ipu_strategy=ipu_strategy).compile(feed_list, fetch_list)
            else:
                program = main_prog

            result = exe.run(program, feed=self.feed, fetch_list=fetch_list)
            return result[0]


class TestDisableTransferCast(TestEnableFp16):
    def set_atol(self):
        self.atol = 1e-10

    def set_data_feed(self):
        self.feed = {"x": np.array([1, 200, 3000, 40000]).astype('int32'), }

    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['dtype'] = 'float32'

    def _test_base(self, run_ipu=True):
        scope = paddle.static.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = self.SEED
        startup_prog.random_seed = self.SEED

        with paddle.static.scope_guard(scope):
            with paddle.static.program_guard(main_prog, startup_prog):
                x = paddle.static.data(
                    name=self.feed_list[0],
                    shape=self.feed_shape[0],
                    dtype=self.feed_dtype[0])
                out = paddle.cast(x, **self.attrs)
                fetch_list = [out.name]

            if run_ipu:
                place = paddle.IPUPlace()
            else:
                place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(startup_prog)

            if run_ipu:
                feed_list = self.feed_list
                ipu_strategy = paddle.static.IpuStrategy()
                ipu_strategy.set_graph_config(is_training=self.is_training)
                ipu_strategy.set_precision_config(enable_fp16=True)
                ipu_strategy.set_options({"transfer_cast_op": False})
                program = paddle.static.IpuCompiledProgram(
                    main_prog,
                    ipu_strategy=ipu_strategy).compile(feed_list, fetch_list)
            else:
                program = main_prog

            result = exe.run(program, feed=self.feed, fetch_list=fetch_list)
            return result[0]


class TestCase2(TestBase):
    def set_atol(self):
        self.atol = 1e-10

    def set_data_feed(self):
        self.feed = {
            "x": np.random.uniform(size=[1, 3, 3, 3]).astype('float16'),
        }

    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['dtype'] = 'float32'


class TestCase3(TestBase):
    def set_atol(self):
        self.atol = 1e-10

    def set_data_feed(self):
        self.feed = {
            "x": np.random.uniform(size=[1, 3, 3, 3]).astype('float32'),
        }

    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['dtype'] = 'int32'


class TestCase4(TestBase):
    def set_atol(self):
        self.atol = 1e-10

    def set_data_feed(self):
        self.feed = {
            "x": np.random.uniform(size=[1, 3, 3, 3]).astype('int32'),
        }

    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['dtype'] = 'float32'


class TestCase5(TestBase):
    def set_atol(self):
        self.atol = 1e-10

    def set_data_feed(self):
        self.feed = {
            "x": np.random.uniform(size=[1, 3, 3, 3]).astype('float16'),
        }

    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['dtype'] = 'int32'


class TestCase6(TestBase):
    def set_atol(self):
        self.atol = 1e-10

    def set_data_feed(self):
        self.feed = {
            "x": np.random.uniform(size=[1, 3, 3, 3]).astype('int32'),
        }

    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['dtype'] = 'float16'


@unittest.skip('float64 is not supported')
class TestCase2(TestBase):
    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['dtype'] = 'float64'


@unittest.skip('skip float16 to float32')
class TestCase3(TestBase):
    def set_data_feed(self):
        self.feed = {
            "x": np.random.uniform(size=[1, 3, 3, 3]).astype('float16'),
        }

    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['dtype'] = 'float32'


@unittest.skip('int32 to int8 is not supported')
class TestCase4(TestBase):
    def set_atol(self):
        self.atol = 1

    def set_data_feed(self):
        self.feed = {
            "x": np.random.randint(
                low=1, high=100, size=[1, 3, 3, 3]).astype('int32'),
        }

    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['dtype'] = 'int8'


if __name__ == "__main__":
    unittest.main()
