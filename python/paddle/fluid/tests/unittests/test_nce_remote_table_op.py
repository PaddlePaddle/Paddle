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

from __future__ import print_function

import os
import signal
import time
import unittest
from multiprocessing import Process

import numpy as np
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.op import Operator
from paddle.fluid.framework import Program, program_guard
from dist_test_utils import *
from paddle.fluid.transpiler.distribute_transpiler import DistributedMode


def nce(input, weight, bias, sample_weight, labels, num_classes,
        num_sample_class):
    samples = []
    sample_labels = []
    batch_size = input.shape[0]
    num_true_class = labels.shape[1]
    for i in range(batch_size):
        w = 1 if sample_weight is None else sample_weight[i]
        for label in labels[i]:
            samples.append((i, label, True, w))
            sample_labels.append(label)
        for num in range(num_sample_class):
            samples.append((i, num, False, w))
            sample_labels.append(num)
    # forward bias
    sample_out = np.zeros(len(samples)).astype(np.float32)
    if bias is not None:
        for i in range(len(samples)):
            sample_out[i] = bias[samples[i][1]]
    # forward weight
    for i in range(len(samples)):
        sample_out[i] += np.dot(input[samples[i][0]], weight[samples[i][1]])

    # forward activation
    sample_out = 1.0 / (1.0 + np.exp(-sample_out))
    # forward cost
    out = np.zeros(batch_size).astype(np.float32)
    b = 1.0 / num_classes * num_sample_class

    for i in range(len(samples)):
        o = sample_out[i]
        cost = -np.log(o / (o + b)) if samples[i][2] else -np.log(b / (o + b))
        out[samples[i][0]] += cost * samples[i][3]
    return (out[:, np.newaxis], np.array(sample_out).reshape(
        batch_size, num_sample_class + num_true_class),
            np.array(sample_labels).reshape(batch_size,
                                            num_sample_class + num_true_class))


def run_pserver(pserver_id, use_cuda, sync_mode):
    remove_ps_flag(os.getpid())
    scope = fluid.core.Scope()
    program = Program()
    with fluid.scope_guard(scope):
        with program_guard(program, startup_program=Program()):
            # create table parameter in scope
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            # create and initialize Param Variable
            param = scope.var('table').get_tensor()

            param_array = np.ones((5, 8)).astype("float32")
            for i in range(len(param_array)):
                param_array[i] *= param_array[i] * i + pserver_id * 10 + 1
            param.set(param_array, place)

            optimize_block = program._create_block(program.global_block().idx)
            program.global_block().append_op(
                type="listen_and_serv",
                inputs={'X': []},
                outputs={},
                attrs={
                    "optimize_blocks": [optimize_block],
                    "endpoint": '127.0.0.1:0',
                    "Fanin": 1,
                    "distributed_mode": DistributedMode.SYNC,
                    "grad_to_block_id": []
                })

            exe = fluid.Executor(place)
            exe.run(program)


class TestListenAndServOp(unittest.TestCase):
    def setUp(self):
        self.ps_timeout = 5

    def _start_pserver(self, pserver_id, use_cuda, sync_mode, pserver_func):
        p = Process(target=pserver_func, args=(pserver_id, use_cuda, sync_mode))
        p.daemon = True
        p.start()
        return p

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

    def _get_pserver_port(self, pid):
        with open("/tmp/paddle.%d.port" % pid, 'r') as f:
            port = int(f.read().strip())
        return port

    def _run_nce_op_two_pserver(self, place, port0, port1):
        scope = fluid.core.Scope()
        program = Program()
        with fluid.scope_guard(scope):
            with program_guard(program, startup_program=Program()):
                x = scope.var('Input').get_tensor()
                x_array = np.random.random((4, 8)).astype("float32")
                x.set(x_array, place)
                # create and initialize Param Variable
                param = scope.var('Weight').get_tensor()
                param_array = np.zeros((5, 8)).astype("float32")
                param.set(param_array, place)

                bias = scope.var('Bias').get_tensor()
                bias_array = np.random.random((5, 1)).astype("float32")
                bias.set(bias_array, place)

                sample_w = scope.var('SampleWeight').get_tensor()
                sample_weight = np.random.random((4, 1)).astype("float32")
                sample_w.set(sample_weight, place)

                label = scope.var('Label').get_tensor()
                label_array = np.array([[0], [1], [4], [3]])
                label.set(label_array, place)

                cost = scope.var('Cost').get_tensor()
                cost_w = np.zeros((4, 1)).astype("float32")
                cost.set(cost_w, place)

                sample_l = scope.var('SampleLogits').get_tensor()
                sample_l_w = np.zeros((4, 3)).astype("float32")
                sample_l.set(sample_l_w, place)

                sample_la = scope.var('SampleLabels').get_tensor()
                sample_la_w = np.zeros((4, 3)).astype("int")
                sample_la.set(sample_la_w, place)

                emaps = ['127.0.0.1:' + str(port0), '127.0.0.1:' + str(port1)]
                table_names = ['table', 'table']
                height_sections = [2, 3]

                # create and run nce operator
                nce_op = Operator(
                    "nce",
                    Input='Input',
                    Weight='Weight',
                    Label='Label',
                    Bias='Bias',
                    Cost='Cost',
                    SampleLogits='SampleLogits',
                    SampleLabels='SampleLabels',
                    SampleWeight='SampleWeight',
                    num_total_classes=5,
                    num_neg_samples=2,
                    custom_neg_classes=list(range(2)),
                    sampler=0,
                    seed=0,
                    is_sparse=True,
                    remote_prefetch=True,
                    epmap=emaps,
                    table_names=table_names,
                    height_sections=height_sections)

                nce_op.run(scope, place)

                # get and compare result
                o_cost = np.array(scope.var('Cost').get_tensor())
                o_logits = np.array(scope.var('SampleLogits').get_tensor())
                o_labels = np.array(scope.var('SampleLabels').get_tensor())

                param_array = np.ones((5, 8)).astype("float32")
                for i in range(2):
                    param_array[i] *= param_array[i] * i + 0 * 10 + 1
                for i in range(2, 5):
                    param_array[i] *= param_array[i] * i + 1 * 10 + 1
                out = nce(x_array, param_array, bias_array, sample_weight,
                          label_array, 5, 2)

                np.testing.assert_almost_equal(o_cost, out[0], decimal=6)
                np.testing.assert_almost_equal(o_logits, out[1], decimal=6)
                np.testing.assert_almost_equal(o_labels, out[2], decimal=6)

    def test_nce_op_remote(self):
        os.environ['PADDLE_ENABLE_REMOTE_PREFETCH'] = "1"
        # run pserver on CPU in sync mode
        p0 = self._start_pserver(0, False, True, run_pserver)
        self._wait_ps_ready(p0.pid)
        port0 = self._get_pserver_port(p0.pid)

        p1 = self._start_pserver(1, False, True, run_pserver)
        self._wait_ps_ready(p1.pid)
        port1 = self._get_pserver_port(p1.pid)

        places = [core.CPUPlace()]

        for place in places:
            self._run_nce_op_two_pserver(place, port0, port1)

        # raise SIGTERM to pserver
        os.kill(p0.pid, signal.SIGINT)
        p0.join()
        os.kill(p1.pid, signal.SIGINT)
        p1.join()


if __name__ == '__main__':
    unittest.main()
