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

import paddle.fluid as fluid
import paddle.fluid.layers as layers
import numpy
from multiprocessing import Process
import os, sys
import time


class TestRecvOp(unittest.TestCase):
    def test_send(self):
        # Run init_serv in a thread
        place = fluid.CPUPlace()
        p = Process(target=self.init_serv, args=(place, ))
        p.daemon = True
        p.start()
        time.sleep(1)
        self.init_client(place)
        # FIXME(typhoonzero): find a way to gracefully shutdown the server.
        os.system("kill -9 %d" % p.pid)
        p.join()

    def init_serv(self, place):
        main = fluid.Program()
        with fluid.program_guard(main):
            serv = layers.ListenAndServ(
                "127.0.0.1:6174", ["X"], optimizer_mode=False)
            with serv.do():
                x = layers.data(
                    shape=[32, 32],
                    dtype='float32',
                    name="X",
                    append_batch_size=False)
                fluid.initializer.Constant(value=1.0)(x, main.global_block())
                o = layers.scale(x=x, scale=10.0)
            main.global_block().create_var(
                name=o.name, psersistable=False, dtype=o.dtype, shape=o.shape)
        exe = fluid.Executor(place)
        exe.run(main)

    def init_client(self, place):
        main = fluid.Program()
        with fluid.program_guard(main):
            x = layers.data(
                shape=[32, 32],
                dtype='float32',
                name='X',
                append_batch_size=False)
            fluid.initializer.Constant(value=1.0)(x, main.global_block())
            layers.Send("127.0.0.1:6174", [x], [x])
        exe = fluid.Executor(place)
        exe.run(main)


if __name__ == "__main__":
    unittest.main()
