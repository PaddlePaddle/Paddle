# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import argparse
import os
import sys
import signal
import time
import socket
from contextlib import closing
from six import string_types
import math
import paddle
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
import paddle.fluid.unique_name as nameGen
from paddle.fluid import core
import unittest
from multiprocessing import Process
import paddle.fluid.layers as layers
from functools import reduce
from test_collective_base import TestCollectiveRunnerBase,runtime_main

class TestCollectiveBroadcast(TestCollectiveRunnerBase):
    def __init__(self):
        self.global_ring_id=0
    def wait_server_ready(self,endpoints):
        assert not isinstance(endpoints, string_types)
        while True:
            all_ok = True
            not_ready_endpoints = []
            for ep in endpoints:
                ip_port = ep.split(":")
                with closing(socket.socket(socket.AF_INET,
                                       socket.SOCK_STREAM)) as sock:
                    sock.settimeout(2)
                    result = sock.connect_ex((ip_port[0], int(ip_port[1])))
                    if result != 0:
                        all_ok = False
                        not_ready_endpoints.append(ep)
            if not all_ok:
                sys.stderr.write("server not ready, wait 3 sec to retry...\n")
                sys.stderr.write("not ready endpoints:" + str(not_ready_endpoints) +
                             "\n")
                sys.stderr.flush()
                time.sleep(3)
            else:
                break
    #endpoints should be ["ip1:port1","ip2:port2"]
    def initCommunicator(self,program,rank,nranks,wait_port,current_endpoint,endpoints):
        other_endpoints = endpoints[:]
        other_endpoints.remove(current_endpoint)
        if rank == 0 and wait_port:
            self.wait_server_ready(other_endpoints)
        block = program.global_block()
        nccl_id_var = block.create_var(
            name=nameGen.generate('nccl_id'),
            persistable=True,
            type=core.VarDesc.VarType.RAW)

        block.append_op(
            type='c_gen_nccl_id',
            inputs={},
            outputs={'Out': nccl_id_var},
            attrs={
                'rank': rank,
                'endpoint': current_endpoint,
                'other_endpoints': other_endpoints
            })

        block.append_op(
            type='c_comm_init',
            inputs={'X': nccl_id_var},
            outputs={},
            attrs={'nranks': nranks,
                   'rank': rank,
                   'ring_id': self.global_ring_id})

                
    def get_model(self,main_prog,startup_program):
        ring_id = 0
        reduce_type=0
        with fluid.program_guard(main_prog,startup_program):
            tindata=layers.data(name="tindata",shape=[32],dtype='int32')
            toutdata = main_prog.current_block().create_var(
                name="outofbroadcast",
                dtype='int32',
                type=core.VarDesc.VarType.LOD_TENSOR,
                persistable=False,
                stop_gradient=False)
            main_prog.global_block().append_op(
                type="c_broadcast",
                inputs = {'X':tindata },
                attrs = {
                        'ring_id': ring_id,
                        'root': reduce_type
                    },
                 outputs = {'Out': toutdata}
            )
            return toutdata

if __name__ == "__main__":
    runtime_main(TestCollectiveBroadcast)

