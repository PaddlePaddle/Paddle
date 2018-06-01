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
import time
import unittest
from multiprocessing import Process

import numpy

import paddle.fluid as fluid
import paddle.fluid.layers as layers


class TestSendOp(unittest.TestCase):
    @unittest.skip(
        "This test is buggy. We cannot use time.sleep to sync processes, the connection may fail in unittest."
    )
    def test_send(self):
        # Run init_serv in a thread
        place = fluid.CPUPlace()
        # NOTE: python thread will not work here due to GIL.
        p = Process(target=self.init_serv, args=(place, ))
        p.daemon = True
        p.start()

        time.sleep(10)
        with open("/tmp/paddle.%d.port" % p.pid, "r") as fn:
            selected_port = int(fn.readlines()[0])
        self.init_client(place, selected_port)

        self.run_local(place)
        self.assertTrue(numpy.allclose(self.local_out, self.dist_out))

        # FIXME(typhoonzero): find a way to gracefully shutdown the server.
        os.system("kill -9 %d" % p.pid)
        p.join()

    def init_serv(self, place):
        main = fluid.Program()

        with fluid.program_guard(main):
            serv = layers.ListenAndServ(
                "127.0.0.1:0", ["X"], optimizer_mode=False)
            with serv.do():
                out_var = main.global_block().create_var(
                    name="scale_0.tmp_0",
                    psersistable=True,
                    dtype="float32",
                    shape=[32, 32])
                x = layers.data(
                    shape=[32, 32],
                    dtype='float32',
                    name="X",
                    append_batch_size=False)
                fluid.initializer.Constant(value=1.0)(x, main.global_block())
                layers.scale(x=x, scale=10.0, out=out_var)

        self.server_exe = fluid.Executor(place)
        self.server_exe.run(main)

    def init_client(self, place, port):
        main = fluid.Program()
        with fluid.program_guard(main):
            x = layers.data(
                shape=[32, 32],
                dtype='float32',
                name='X',
                append_batch_size=False)
            fluid.initializer.Constant(value=2.3)(x, main.global_block())
            get_var = main.global_block().create_var(
                name="scale_0.tmp_0",  # server side var
                dtype="float32",
                persistable=False,
                shape=[32, 32])
            o = layers.Send("127.0.0.1:%d" % port, [x], [get_var])
        exe = fluid.Executor(place)
        self.dist_out = exe.run(main, fetch_list=o)  # o is a list

    def run_local(self, place):
        main = fluid.Program()
        with fluid.program_guard(main):
            x = layers.data(
                shape=[32, 32],
                dtype='float32',
                name='X',
                append_batch_size=False)
            fluid.initializer.Constant(value=2.3)(x, main.global_block())
            o = layers.scale(x=x, scale=10.0)
        exe = fluid.Executor(place)
        self.local_out = exe.run(main, fetch_list=[o])


if __name__ == "__main__":
    unittest.main()
