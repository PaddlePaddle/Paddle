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

<<<<<<< HEAD
import os
import signal
import time
import unittest
from multiprocessing import Process

import numpy as np
from dist_test_utils import remove_ps_flag

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers.ops as ops
from paddle.fluid import core
from paddle.fluid.layers.io import ListenAndServ, Recv, Send

RPC_OP_ROLE_ATTR_NAME = (
    op_role_attr_name
) = core.op_proto_and_checker_maker.kOpRoleAttrName()
=======
from __future__ import print_function

import os
import time
import unittest
from multiprocessing import Process
import signal

import numpy as np

import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.layers.io import ListenAndServ
from paddle.fluid.layers.io import Recv
from paddle.fluid.layers.io import Send
import paddle.fluid.layers.ops as ops
from dist_test_utils import *

from paddle.fluid import core

RPC_OP_ROLE_ATTR_NAME = op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName(
)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
RPC_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.RPC


class TestSendOp(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_send(self):
        remove_ps_flag(os.getpid())
        # Run init_serv in a thread
        place = fluid.CPUPlace()
        # NOTE: python thread will not work here due to GIL.
<<<<<<< HEAD
        p = Process(target=self.init_serv, args=(place,))
=======
        p = Process(target=self.init_serv, args=(place, ))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        p.daemon = True
        p.start()

        self.ps_timeout = 5
        self._wait_ps_ready(p.pid)

        with open("/tmp/paddle.%d.port" % p.pid, "r") as fn:
            selected_port = int(fn.readlines()[0])
        self.init_client(place, selected_port)

        self.run_local(place)
        np.testing.assert_allclose(self.local_out, self.dist_out, rtol=1e-05)

        os.kill(p.pid, signal.SIGINT)
        p.join()

    def _wait_ps_ready(self, pid):
        start_left_time = self.ps_timeout
        sleep_time = 0.5
        while True:
            assert start_left_time >= 0, "wait ps ready failed"
            time.sleep(sleep_time)
            try:
                # the listen_and_serv_op would touch a file which contains the listen port
                # on the /tmp directory until it was ready to process all the RPC call.
                os.stat("/tmp/paddle.%d.port" % pid)
                return
            except os.error:
                start_left_time -= sleep_time

    def init_serv(self, place):
        main = fluid.Program()

        with fluid.program_guard(main):
            serv = ListenAndServ("127.0.0.1:0", ["X"], optimizer_mode=False)
            with serv.do():
<<<<<<< HEAD
                out_var = main.global_block().create_var(
                    name="scale_0.tmp_0",
                    psersistable=True,
                    dtype="float32",
                    shape=[32, 32],
                )
                x = paddle.static.data(
                    shape=[32, 32],
                    dtype='float32',
                    name="X",
                )
=======
                out_var = main.global_block().create_var(name="scale_0.tmp_0",
                                                         psersistable=True,
                                                         dtype="float32",
                                                         shape=[32, 32])
                x = layers.data(shape=[32, 32],
                                dtype='float32',
                                name="X",
                                append_batch_size=False)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                fluid.initializer.Constant(value=1.0)(x, main.global_block())
                ops._scale(x=x, scale=10.0, out=out_var)

        self.server_exe = fluid.Executor(place)
        self.server_exe.run(main)

    def init_client(self, place, port):
        main = fluid.Program()
        with fluid.program_guard(main):
<<<<<<< HEAD
            main.global_block().append_op(
                type="fetch_barrier",
                inputs={},
                outputs={"Out": []},
                attrs={
                    "endpoints": ["127.0.0.1:{0}".format(port)],
                    RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE,
                },
            )

            x = paddle.static.data(shape=[32, 32], dtype='float32', name='X')
=======
            main.global_block().append_op(type="fetch_barrier",
                                          inputs={},
                                          outputs={"Out": []},
                                          attrs={
                                              "endpoints":
                                              ["127.0.0.1:{0}".format(port)],
                                              RPC_OP_ROLE_ATTR_NAME:
                                              RPC_OP_ROLE_ATTR_VALUE
                                          })

            x = layers.data(shape=[32, 32],
                            dtype='float32',
                            name='X',
                            append_batch_size=False)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            x.persistable = True
            fluid.initializer.Constant(value=2.3)(x, main.global_block())

            get_var = main.global_block().create_var(
                name="scale_0.tmp_0",  # server side var
                dtype="float32",
                persistable=False,
<<<<<<< HEAD
                shape=[32, 32],
            )
=======
                shape=[32, 32])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            fluid.initializer.Constant(value=2.3)(get_var, main.global_block())

            # NOTE(zjl): `Send` is async send, which means that the sent
            # variable would be needed even though `Send` op runs.
            # Is it a right design? If I do not set `x.persistable = True`,
            # this unittest would hang in rpc client after x is deleted.
            #
            # BTW, `Send` is not a public API to users. So I set
            # `x.persistable = True` to be a hot fix of this unittest.
            Send("127.0.0.1:%d" % port, [x])
            o = Recv("127.0.0.1:%d" % port, [get_var])

        exe = fluid.Executor(place)
        self.dist_out = exe.run(main, fetch_list=o)  # o is a list

    def run_local(self, place):
        main = fluid.Program()
        with fluid.program_guard(main):
<<<<<<< HEAD
            x = paddle.static.data(shape=[32, 32], dtype='float32', name='X')
            fluid.initializer.Constant(value=2.3)(x, main.global_block())
            o = paddle.scale(x=x, scale=10.0)
=======
            x = layers.data(shape=[32, 32],
                            dtype='float32',
                            name='X',
                            append_batch_size=False)
            fluid.initializer.Constant(value=2.3)(x, main.global_block())
            o = layers.scale(x=x, scale=10.0)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        exe = fluid.Executor(place)
        self.local_out = exe.run(main, fetch_list=[o])


if __name__ == "__main__":
    unittest.main()
