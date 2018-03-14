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

import unittest

import numpy
import paddle.fluid.core as core

from paddle.fluid.executor import Executor
from paddle.fluid.layers import mul, data


class TestExecutor(unittest.TestCase):
    def test_mul(self):
        a = data(name='a', shape=[784], dtype='float32')
        b = data(
            name='b',
            shape=[784, 100],
            dtype='float32',
            append_batch_size=False)
        out = mul(x=a, y=b)
        place = core.CPUPlace()
        a_np = numpy.random.random((100, 784)).astype('float32')
        b_np = numpy.random.random((784, 100)).astype('float32')
        exe = Executor(place)
        outs = exe.run(feed={'a': a_np, 'b': b_np}, fetch_list=[out])
        out = outs[0]
        self.assertEqual((100, 100), out.shape)
        self.assertTrue(numpy.allclose(out, numpy.dot(a_np, b_np)))

    def test_prepare_then_run(self):
        a = data(name='a', shape=[784], dtype='float32')
        b = data(
            name='b',
            shape=[784, 100],
            dtype='float32',
            append_batch_size=False)
        c = data(
            name='c', shape=[100, 10], dtype='float32', append_batch_size=False)
        out = mul(x=a, y=b)
        place = core.CPUPlace()
        a_np = numpy.random.random((100, 784)).astype('float32')
        b_np = numpy.random.random((784, 100)).astype('float32')
        c_np = numpy.random.random((100, 10)).astype('float32')
        exe = Executor(place)
        feed = {'a': a_np, 'b': b_np, 'c': c_np}

        prepared_ctx = exe.prepare(feed=feed, fetch_list=[out])
        for _ in range(2):
            outs = exe.run_prepared_ctx(ctx=prepared_ctx, feed=feed)
            out_np = outs[0]
            self.assertEqual((100, 100), out_np.shape)
            self.assertTrue(numpy.allclose(out_np, numpy.dot(a_np, b_np)))

        new_out = mul(x=out, y=c)
        new_prepared_ctx = exe.prepare(feed=feed, fetch_list=[new_out])

        handle_equal = (prepared_ctx.handle == new_prepared_ctx.handle)
        self.assertFalse(handle_equal, "handle should not be equal")

        for _ in range(2):
            outs = exe.run_prepared_ctx(ctx=new_prepared_ctx, feed=feed)
            out_np = outs[0]
            self.assertEqual((100, 10), out_np.shape)
            self.assertTrue(
                numpy.allclose(out_np, numpy.dot(numpy.dot(a_np, b_np), c_np)))

        for _ in range(2):
            outs = exe.run_prepared_ctx(ctx=prepared_ctx, feed=feed)
            out_np = outs[0]
            self.assertEqual((100, 100), out_np.shape)
            self.assertTrue(numpy.allclose(out_np, numpy.dot(a_np, b_np)))


if __name__ == '__main__':
    unittest.main()
