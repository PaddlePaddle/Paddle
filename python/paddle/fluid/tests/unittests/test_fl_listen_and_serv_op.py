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
"""test f1 listen and serv_op."""

from __future__ import print_function

import paddle
import paddle.fluid as fluid
from paddle.fluid import Program
import os
import signal
import subprocess
import time
import unittest
from multiprocessing import Process
from op_test import OpTest
import numpy
import urllib
import sys
from dist_test_utils import *

cache_path = os.path.expanduser('~/.cache/paddle/dataset')


def run_trainer(use_cuda, sync_mode, ip, port, trainers, trainer_id):
    ''' 
    This function is run trainer.
    Args:
        use_cuda (bool): whether use cuda.
        sync_mode (nouse): specify sync mode.
        ip (string): the ip address.
        port (string): the port for listening.
        trainers (int): the count of trainer.
        trainer_id (int): the id of trainer.

    Returns:
        None
    '''
    x = fluid.layers.data(name='x', shape=[1], dtype='float32')
    y_predict = fluid.layers.fc(input=x, size=1, act=None)
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')
    # loss function
    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_cost = fluid.layers.mean(cost)
    # optimizer
    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    sgd_optimizer.minimize(avg_cost)
    with open("{}/trainer_recv_program.dms".format(cache_path), "rb") as f:
        trainer_recv_program_desc_str = f.read()
    with open("{}/trainer_main_program.dms".format(cache_path), "rb") as f:
        trainer_main_program_desc_str = f.read()
    with open("{}/trainer_send_program.dms".format(cache_path), "rb") as f:
        trainer_send_program_desc_str = f.read()
    recv_program = Program.parse_from_string(trainer_recv_program_desc_str)
    main_program = Program.parse_from_string(trainer_main_program_desc_str)
    send_program = Program.parse_from_string(trainer_send_program_desc_str)

    trainer_startup_program = fluid.default_startup_program()
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    exe.run(trainer_startup_program)
    for i in range(5):
        exe.run(recv_program)
        exe.run(fluid.default_main_program(),
                feed={
                    "x": numpy.array([1, 2]).astype('float32').reshape(2, 1),
                    "y": numpy.array([2, 3]).astype('float32').reshape(2, 1)
                })
        exe.run(send_program)


def run_pserver(use_cuda, sync_mode, ip, port, trainers, trainer_id):
    ''' 
    This function is run trainer.
    Args:
        use_cuda (bool): whether use cuda.
        sync_mode (nouse): specify sync mode.
        ip (string): the ip address.
        port (string): the port for listening.
        trainers (int): the count of trainer.
        trainer_id (int): the id of trainer.

    Returns:
        None
    '''
    remove_ps_flag(os.getpid())
    x = fluid.layers.data(name='x', shape=[1], dtype='float32')
    y_predict = fluid.layers.fc(input=x, size=1, act=None)
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')
    # loss function
    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_cost = fluid.layers.mean(cost)
    # optimizer
    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    sgd_optimizer.minimize(avg_cost)
    with open("{}/pserver_startup_program.dms".format(cache_path), "rb") as f:
        pserver_startup_program_desc_str = f.read()
    with open("{}/pserver_main_program.dms".format(cache_path), "rb") as f:
        pserver_main_program_desc_str = f.read()

    startup_program = Program.parse_from_string(
        pserver_startup_program_desc_str)
    main_program = Program.parse_from_string(pserver_main_program_desc_str)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_program)
    exe.run(main_program)


class TestFlListenAndServOp(unittest.TestCase):
    """This class is Test Fl Listen And ServOp."""

    def setUp(self):
        """This function si set Up."""
        self.ps_timeout = 5
        self.ip = "127.0.0.1"
        self.port = "6000"
        self.trainers = 2
        self.trainer_id = 0

    def _start_pserver(self, use_cuda, sync_mode, pserver_func):
        """This function is start pserver."""
        p = Process(
            target=pserver_func,
            args=(use_cuda, sync_mode, self.ip, self.port, self.trainers,
                  self.trainer_id))
        p.daemon = True
        p.start()
        return p

    def _start_trainer0(self, use_cuda, sync_mode, pserver_func):
        """This function is start trainer0."""
        p = Process(
            target=pserver_func,
            args=(use_cuda, sync_mode, self.ip, self.port, self.trainers, 0))
        p.daemon = True
        p.start()
        return p

    def _start_trainer1(self, use_cuda, sync_mode, pserver_func):
        """This function is start trainer1."""
        p = Process(
            target=pserver_func,
            args=(use_cuda, sync_mode, self.ip, self.port, self.trainers, 1))
        p.daemon = True
        p.start()
        return p

    def _wait_ps_ready(self, pid):
        """This function is wait ps ready."""
        start_left_time = self.ps_timeout
        sleep_time = 0.5
        while True:
            assert start_left_time >= 0, "wait ps ready failed"
            time.sleep(sleep_time)
            try:
                os.stat("/tmp/paddle.%d.port" % pid)
                return
            except os.error:
                start_left_time -= sleep_time

    def test_rpc_interfaces(self):
        """TODO(Yancey1989): need to make sure the rpc interface correctly."""
        # TODO(Yancey1989): need to make sure the rpc interface correctly.
        pass

    def test_handle_signal_in_serv_op(self):
        """run pserver on CPU in sync mode."""
        # run pserver on CPU in sync mode
        if sys.platform == 'win32' or sys.platform == 'sys.platform':
            pass
        else:
            print(sys.platform)
            file_list = [
                'pserver_startup_program.dms', 'pserver_main_program.dms',
                'trainer_recv_program.dms', 'trainer_main_program.dms',
                'trainer_send_program.dms'
            ]
            if not os.path.exists(cache_path):
                os.makedirs(cache_path)
            prefix = 'wget --no-check-certificate https://paddlefl.bj.bcebos.com/test_fl_listen_and_serv/'
            for f in file_list:
                if not os.path.exists('{}/{}'.format(cache_path, f)):
                    cmd = "wget --no-check-certificate https://paddlefl.bj.bcebos.com/test_fl_listen_and_serv/{} -P {}/".format(
                        f, cache_path)
                    os.system(cmd)
            p1 = self._start_pserver(False, True, run_pserver)
            self._wait_ps_ready(p1.pid)
            time.sleep(5)
            t1 = self._start_trainer0(False, True, run_trainer)
            time.sleep(2)
            t2 = self._start_trainer1(False, True, run_trainer)
            # raise SIGTERM to pserver
            time.sleep(2)
            cmd_del = "rm trainer*dms* pserver*dms*"
            os.system(cmd_del)
            os.kill(p1.pid, signal.SIGINT)
            p1.join()
            os.kill(t1.pid, signal.SIGINT)
            t1.join()
            os.kill(t2.pid, signal.SIGINT)
            t2.join()


if __name__ == '__main__':
    unittest.main()
