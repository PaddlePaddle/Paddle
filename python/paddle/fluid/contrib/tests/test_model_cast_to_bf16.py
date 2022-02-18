#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.fluid as fluid
import contextlib
import unittest
import numpy as np
import struct
import paddle.fluid.layers as layers
import paddle.static.amp as amp
from paddle.fluid import core

paddle.enable_static()


def convert_uint16_to_float(in_list):
    if in_list.dtype == np.uint16:
        in_list = np.asarray(in_list)
        out = np.vectorize(
            lambda x: struct.unpack('<f', struct.pack('<I', x << 16))[0],
            otypes=[np.float32])(in_list.flat)
        return np.reshape(out, in_list.shape)
    else:
        return in_list


cutf = convert_uint16_to_float


@unittest.skipIf(not core.supports_bfloat16(),
                 "place does not support BF16 evaluation")
class TestModelCastBF16(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seed = 111

    @classmethod
    def tearDownClass(cls):
        pass

    @contextlib.contextmanager
    def static_graph(self):
        with self.scope_prog_guard():
            paddle.seed(self.seed)
            paddle.framework.random._manual_program_seed(self.seed)
            yield

    @contextlib.contextmanager
    def scope_prog_guard(self):
        prog = fluid.Program()
        startup_prog = fluid.Program()
        scope = fluid.core.Scope()
        with fluid.scope_guard(scope):
            with fluid.program_guard(prog, startup_prog):
                yield

    def get_static_graph_result(self,
                                feed,
                                fetch_list,
                                amp_fun,
                                with_lod=False,
                                startup_prog=None):
        exe = fluid.Executor(core.CPUPlace())
        exe.run(fluid.default_startup_program()
                if startup_prog is None else startup_prog)
        prog = fluid.default_main_program()
        if amp_fun is not None:
            if startup_prog is not None:
                amp_fun(prog, startup_prog)
            else:
                amp_fun(prog)
        return exe.run(prog,
                       feed=feed,
                       fetch_list=fetch_list,
                       return_numpy=(not with_lod))

    def _graph_common(self, _amp_fun, startup_prog=None):
        size = 3
        n = np.ones([size, size], dtype='float32') * 3.2
        nn = np.ones([size, size], dtype='float32') * -2.7

        n_bf16 = amp.bf16.convert_float_to_uint16(n)
        nn_bf16 = amp.bf16.convert_float_to_uint16(nn)

        with self.static_graph():
            t_bf16 = layers.data(
                name='t_bf16', shape=[size, size], dtype=np.uint16)
            tt_bf16 = layers.data(
                name='tt_bf16', shape=[size, size], dtype=np.uint16)
            t = layers.data(name='t', shape=[size, size], dtype='float32')
            tt = layers.data(name='tt', shape=[size, size], dtype='float32')

            ret = layers.elementwise_add(t, tt)
            ret = layers.elementwise_mul(ret, t)
            ret = layers.reshape(ret, [0, 0])

            with amp.bf16.bf16_guard():
                ret_bf16 = layers.elementwise_add(t_bf16, tt_bf16)
                ret_bf16 = layers.elementwise_mul(ret_bf16, t_bf16)
                ret_bf16 = layers.reshape(ret_bf16, [0, 0])

            with amp.bf16.bf16_guard():
                ret_fp32bf16 = layers.elementwise_add(t, tt)
                ret_fp32bf16 = layers.elementwise_mul(ret_fp32bf16, t)
                ret_fp32bf16 = layers.reshape(ret_fp32bf16, [0, 0])

            static_ret_bf16, static_ret, ret_fp32bf16 = self.get_static_graph_result(
                feed={
                    't': n,
                    'tt': nn,
                    't_bf16': n_bf16,
                    'tt_bf16': nn_bf16,
                },
                fetch_list=[ret_bf16, ret, ret_fp32bf16],
                amp_fun=_amp_fun,
                startup_prog=startup_prog)

        self.assertTrue(
            np.allclose(cutf(static_ret_bf16), cutf(static_ret), 1e-2))
        self.assertTrue(
            np.allclose(cutf(static_ret_bf16), cutf(ret_fp32bf16), 1e-2))

        with self.static_graph():
            t = layers.data(name='t', shape=[size, size], dtype='float32')
            tt = layers.data(name='tt', shape=[size, size], dtype='float32')

            with amp.bf16.bf16_guard():
                ret = layers.elementwise_add(t, tt)
                ret = layers.reshape(ret, [0, 0], act='elu')
                ret = layers.elementwise_mul(ret, t)
            ret = layers.elementwise_add(ret, tt)

            static_ret_bf16 = \
                self.get_static_graph_result(
                    feed={'t': n, 'tt': nn},
                    fetch_list=[ret],
                    amp_fun=_amp_fun,
                    startup_prog=startup_prog
                )
        self.assertTrue(
            static_ret_bf16, np.ones(
                [size, size], dtype='float32') * -1.1)

    def test_graph_rewrite(self):
        self._graph_common(lambda prog: amp.bf16.rewrite_program_bf16(
            prog,
            amp.bf16.AutoMixedPrecisionListsBF16(
                custom_bf16_list={'elementwise_add'},
                custom_fp32_varnames={'elementwise_add_0.tmp_0'})
        ))

    def test_graph_cast(self):
        self._graph_common(lambda prog, startup_prog: amp.bf16.cast_model_to_bf16(
            prog,
            startup_prog,
            amp.bf16.AutoMixedPrecisionListsBF16(
                custom_bf16_list={'elementwise_add'},
                custom_fp32_list={'elementwise_mul'}),
            use_bf16_guard=True
        ), startup_prog=fluid.default_startup_program())


if __name__ == '__main__':
    unittest.main()
