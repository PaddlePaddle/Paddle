# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import unittest
import time
import argparse
import os
import sys

sys.path.append("..")
import subprocess
import traceback
import functools
import pickle
from contextlib import closing
import paddle.fluid as fluid
import paddle.fluid.unique_name as nameGen
from paddle.fluid import core
import paddle

from op_test import OpTest, _set_use_system_allocator

from test_sync_batch_norm_op import create_or_get_tensor

_set_use_system_allocator(False)
paddle.enable_static()

SEED = 10


class TestSyncBatchNormRunnerBase:
    def get_model(
        self,
        main,
        startup,
        place,
        layout,
        seed,
        sync_bn=False,
        only_forward=False,
    ):
        raise NotImplementedError(
            "get model should be implemented by child class."
        )

    def wait_server_ready(self, endpoints):
        assert not isinstance(endpoints, str)
        while True:
            all_ok = True
            not_ready_endpoints = []
            for ep in endpoints:
                ip_port = ep.split(":")
                with closing(
                    socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                ) as sock:
                    sock.settimeout(2)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    if hasattr(socket, 'SO_REUSEPORT'):
                        sock.setsockopt(
                            socket.SOL_SOCKET, socket.SO_REUSEPORT, 1
                        )

                    result = sock.connect_ex((ip_port[0], int(ip_port[1])))
                    if result != 0:
                        all_ok = False
                        not_ready_endpoints.append(ep)
            if not all_ok:
                sys.stderr.write("server not ready, wait 3 sec to retry...\n")
                sys.stderr.write(
                    "not ready endpoints:" + str(not_ready_endpoints) + "\n"
                )
                sys.stderr.flush()
                time.sleep(3)
            else:
                break

    def initCommunicator(
        self, program, rank, nranks, wait_port, current_endpoint, endpoints
    ):
        other_endpoints = endpoints[:]
        other_endpoints.remove(current_endpoint)
        if rank == 0 and wait_port:
            self.wait_server_ready(other_endpoints)
        block = program.global_block()
        cncl_id_var = block.create_var(
            name=nameGen.generate('cncl_id'),
            persistable=True,
            type=core.VarDesc.VarType.RAW,
        )
        block.append_op(
            type='c_gen_cncl_id',
            inputs={},
            outputs={'Out': cncl_id_var},
            attrs={
                'rank': rank,
                'endpoint': current_endpoint,
                'other_endpoints': other_endpoints,
            },
        )
        block.append_op(
            type='c_comm_init',
            inputs={'X': cncl_id_var},
            outputs={},
            attrs={
                'nranks': nranks,
                'rank': rank,
                'ring_id': self.global_ring_id,
            },
        )

    def run_trainer(self, args):
        device_id = int(os.getenv("FLAGS_selected_mlus", "0"))
        place = fluid.MLUPlace(device_id)
        places = [place]

        # Test training
        for place in places:
            for layout in ["NCHW", "NHWC"]:
                self._compare(args, place, layout, False)

        # Test inference
        for place in places:
            for layout in ["NCHW", "NHWC"]:
                self._compare(args, place, layout, True)

        # Test FP16 - @TODO
        self.bn_dtype = np.float16
        self.atol = 3e-3

        # Test training
        for place in places:
            for layout in ["NCHW", "NHWC"]:
                self._compare(args, place, layout, False)

        # Test inference
        for place in places:
            for layout in ["NCHW", "NHWC"]:
                self._compare(args, place, layout, True)

        sys.stdout.buffer.write(
            pickle.dumps(
                'training, inference, fp32, fp16, NCHW, NHWC all passed'
            )
        )

    def _compare(self, args, place, layout, only_forward):
        scope = core.Scope()

        np.random.seed(SEED)
        data = np.random.random(size=self.dshape).astype(self.dtype) * 4.0 - 2
        sys.stderr.write("data: " + str(data) + "\n")
        data = create_or_get_tensor(
            scope, "input", OpTest.np_dtype_to_fluid_dtype(data), place
        )

        bn_fetches = self._cal_single_card(
            args, data, place, layout, only_forward
        )
        fetch_names, sync_bn_fetches = self._cal_multiple_cards(
            args, data, place, layout, only_forward
        )

        sys.stderr.write(
            "len(sync_bn_fetches): " + str(len(sync_bn_fetches)) + "\n"
        )
        for i in range(0, len(sync_bn_fetches)):
            sys.stderr.write("i: " + str(i) + "\n")
            sys.stderr.write("fetch_names[i]): " + fetch_names[i] + "\n")

            bn_val = bn_fetches[i]
            sync_bn_val = sync_bn_fetches[i]
            if sync_bn_val.shape != bn_val.shape:
                sync_bn_val = sync_bn_val[: bn_val.shape[0]]

            # i = 0
            if fetch_names[i] == 'reduce_sum_0.tmp_0':
                # sys.stderr.write("skip reduce_sum_0.tmp_0 (Out of reduce_sum op)" + "\n")
                sys.stderr.write(
                    "reduce_sum_0.tmp_0 (Out of reduce_sum op)" + "\n"
                )
                sys.stderr.write("bn_val: " + str(bn_val) + "\n")
                sys.stderr.write("sync_bn_val: " + str(sync_bn_val) + "\n")

                # continue

            # i = 1
            if fetch_names[i] == 'conv2d_0.tmp_0':
                # sys.stderr.write("skip conv2d_0.tmp_0 (X)" + "\n")
                sys.stderr.write("conv2d_0.tmp_0 (X)" + "\n")
                sys.stderr.write("bn_val: " + str(bn_val) + "\n")
                sys.stderr.write("sync_bn_val: " + str(sync_bn_val) + "\n")

                # continue

            # i = 2
            if fetch_names[i] == 'batch_norm_0.tmp_3':
                # sys.stderr.write("skip batch_norm_0.tmp_3 (Y)" + "\n")
                sys.stderr.write("batch_norm_0.tmp_3 (Y)" + "\n")
                sys.stderr.write("bn_val: " + str(bn_val) + "\n")
                sys.stderr.write("sync_bn_val: " + str(sync_bn_val) + "\n")

                # continue

            # i = 2
            if fetch_names[i] == 'batch_norm_0.tmp_2':
                # sys.stderr.write("skip batch_norm_0.tmp_2 (ReserveSpace of batch_norm)" + "\n")
                sys.stderr.write(
                    "batch_norm_0.tmp_2 (ReserveSpace of batch_norm)" + "\n"
                )
                sys.stderr.write("bn_val: " + str(bn_val) + "\n")
                sys.stderr.write("sync_bn_val: " + str(sync_bn_val) + "\n")

                # continue

            # i = 3
            if fetch_names[i] == 'bn_moving_mean':
                sys.stderr.write("skip bn_moving_mean (MeanOut)" + "\n")
                sys.stderr.write("bn_val: " + str(bn_val) + "\n")
                sys.stderr.write("sync_bn_val: " + str(sync_bn_val) + "\n")

                continue

            # i = 4
            if fetch_names[i] == 'bn_moving_variance':
                sys.stderr.write("skip bn_moving_variance (VarianceOut)" + "\n")
                sys.stderr.write("bn_val: " + str(bn_val) + "\n")
                sys.stderr.write("sync_bn_val: " + str(sync_bn_val) + "\n")

                continue

            # i = 7
            if fetch_names[i] == 'batch_norm_0.tmp_0':
                # sys.stderr.write("skip batch_norm_0.tmp_0 (SavedMean)" + "\n")
                sys.stderr.write("batch_norm_0.tmp_0 (SavedMean)" + "\n")
                sys.stderr.write("bn_val: " + str(bn_val) + "\n")
                sys.stderr.write("sync_bn_val: " + str(sync_bn_val) + "\n")

                # continue

            # i = 8
            if fetch_names[i] == 'batch_norm_0.tmp_1':
                sys.stderr.write(
                    "skip batch_norm_0.tmp_1 (SavedVariance)" + "\n"
                )
                sys.stderr.write("bn_val: " + str(bn_val) + "\n")
                sys.stderr.write("sync_bn_val: " + str(sync_bn_val) + "\n")

                continue

            # i = 9
            if fetch_names[i] == 'bn_scale@GRAD':
                # sys.stderr.write("skip bn_scale@GRAD (Scale@GRAD)" + "\n")
                sys.stderr.write("bn_scale@GRAD (Scale@GRAD)" + "\n")
                sys.stderr.write("bn_val: " + str(bn_val) + "\n")
                sys.stderr.write("sync_bn_val: " + str(sync_bn_val) + "\n")

                # continue

            # i = 10
            if fetch_names[i] == 'bn_bias@GRAD':
                # sys.stderr.write("skip bn_bias@GRAD (Bias@GRAD)" + "\n")
                sys.stderr.write("bn_bias@GRAD (Bias@GRAD)" + "\n")
                sys.stderr.write("bn_val: " + str(bn_val) + "\n")
                sys.stderr.write("sync_bn_val: " + str(sync_bn_val) + "\n")

                # continue

            # i = 11
            if fetch_names[i] == 'batch_norm_0.tmp_3@GRAD':
                # sys.stderr.write("skip batch_norm_0.tmp_3@GRAD (Y@GRAD)" + "\n")
                sys.stderr.write("batch_norm_0.tmp_3@GRAD (Y@GRAD)" + "\n")
                sys.stderr.write("bn_val: " + str(bn_val) + "\n")
                sys.stderr.write("sync_bn_val: " + str(sync_bn_val) + "\n")

                # continue

            # i = 12
            if fetch_names[i] == 'conv2d_0.tmp_0@GRAD':
                # sys.stderr.write("skip conv2d_0.tmp_0@GRAD (X@GRAD)" + "\n")
                sys.stderr.write("conv2d_0.tmp_0@GRAD (X@GRAD)" + "\n")
                sys.stderr.write("bn_val: " + str(bn_val) + "\n")
                sys.stderr.write("sync_bn_val: " + str(sync_bn_val) + "\n")

                # continue

            atol = self.atol
            if fetch_names[i] == 'conv2d_0.tmp_0@GRAD':
                atol = 1e-2

            assert np.allclose(bn_val, sync_bn_val, atol=atol), (
                "Output ("
                + fetch_names[i]
                + ") has diff. \n"
                + "\nBN     "
                + str(bn_val)
                + "\n"
                + "Sync BN "
                + str(sync_bn_val)
            )

    def _cal_single_card(self, args, data, place, layout, only_forward):
        # Single-MLU, N = 32 per MLU
        train_prog = fluid.Program()
        startup_prog = fluid.Program()
        train_prog.global_seed(SEED)
        startup_prog.global_seed(SEED)
        paddle.seed(SEED)

        outs = self.get_model(
            train_prog, startup_prog, place, layout, SEED, False, only_forward
        )

        exe = fluid.Executor(place)
        exe.run(startup_prog)
        fetch_names = [v.name for v in outs] + [
            'bn_moving_mean',
            'bn_moving_variance',
            'bn_scale',
            'bn_bias',
        ]
        if not only_forward:
            others = [
                'batch_norm_0.tmp_0',
                'batch_norm_0.tmp_1',
                'bn_scale@GRAD',
                'bn_bias@GRAD',
                'batch_norm_0.tmp_3@GRAD',
                'conv2d_0.tmp_0@GRAD',
            ]
            fetch_names += others
        bn_fetches = exe.run(
            program=train_prog, feed={'input': data}, fetch_list=fetch_names
        )

        return bn_fetches

    def _cal_multiple_cards(self, args, data, place, layout, only_forward):
        # Multi-MLUs, self.N per MLU
        assert core.get_mlu_device_count() > 1

        train_prog = fluid.Program()
        startup_prog = fluid.Program()
        train_prog.global_seed(SEED)
        startup_prog.global_seed(SEED)
        paddle.seed(SEED)
        sys.stderr.write("train_prog: " + train_prog.to_string(True) + "\n")
        sys.stderr.write("startup_prog: " + startup_prog.to_string(True) + "\n")

        endpoints = args["endpoints"].split(",")
        rank = args["trainerid"]
        current_endpoint = args["currentendpoint"]
        nranks = 2

        self.initCommunicator(
            startup_prog, rank, nranks, True, current_endpoint, endpoints
        )
        # sys.stderr.write("after init, startup_prog: " +
        #                  startup_prog.to_string(True) + "\n")
        train_prog.global_seed(SEED)
        train_prog._sync_with_cpp()
        startup_prog.global_seed(SEED)
        startup_prog._sync_with_cpp()
        paddle.seed(SEED)

        self.rank = rank
        outs = self.get_model(
            train_prog, startup_prog, place, layout, SEED, True, only_forward
        )
        # sys.stderr.write("after get_model, train_prog: " +
        #                  train_prog.to_string(True) + "\n")
        # sys.stderr.write("after get_model, startup_prog: " +
        #                  startup_prog.to_string(True) + "\n")

        ops = train_prog.blocks[0].ops
        for i, op in enumerate(ops):
            if op.type == 'batch_norm':
                sys.stderr.write("i: " + str(i) + "\n")
                sys.stderr.write("op type: " + op.type + "\n")
                op.desc.set_type('sync_batch_norm')
            if op.type == 'batch_norm_grad':
                sys.stderr.write("i: " + str(i) + "\n")
                sys.stderr.write("op type: " + op.type + "\n")
                op.desc.set_type('sync_batch_norm_grad')

        # sys.stderr.write("after update sync_batch_norm, train_prog: " +
        #                  train_prog.to_string(True) + "\n")

        exe = fluid.Executor(place)
        exe.run(startup_prog)
        fetch_names = [v.name for v in outs] + [
            'bn_moving_mean',
            'bn_moving_variance',
            'bn_scale',
            'bn_bias',
        ]
        if not only_forward:
            others = [
                'batch_norm_0.tmp_0',
                'batch_norm_0.tmp_1',
                'bn_scale@GRAD',
                'bn_bias@GRAD',
                'batch_norm_0.tmp_3@GRAD',
                'conv2d_0.tmp_0@GRAD',
            ]
            fetch_names += others
        sync_bn_fetches = exe.run(
            program=train_prog, feed={'input': data}, fetch_list=fetch_names
        )

        return fetch_names, sync_bn_fetches


def runtime_main(test_class, col_type, sub_type):
    args = {}
    model = test_class()
    args["deviceid"] = os.getenv("FLAGS_selected_mlus")
    args["trainerid"] = int(os.getenv("PADDLE_TRAINER_ID"))
    args["trainernum"] = int(os.getenv("PADDLE_TRAINERS_NUM"))
    args["endpoints"] = os.getenv('PADDLE_TRAINER_ENDPOINTS')
    args["currentendpoint"] = os.getenv("PADDLE_CURRENT_ENDPOINT")
    args["col_type"] = col_type
    model.run_trainer(args)


import socket
from contextlib import closing


class TestDistBase(unittest.TestCase):
    def setUp(self):
        self._port_set = set()
        self._trainers = 2
        self._ps_endpoints = "127.0.0.1:%s,127.0.0.1:%s" % (
            self._find_free_port(),
            self._find_free_port(),
        )
        self._python_interp = sys.executable

    def _find_free_port(self):
        def __free_port():
            with closing(
                socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            ) as s:
                s.bind(('', 0))
                return s.getsockname()[1]

        while True:
            port = __free_port()
            if port not in self._port_set:
                self._port_set.add(port)
                return port

    def _run_cluster(self, model_file, envs):
        worker_endpoints = self._ps_endpoints.split(",")
        w0_ep, w1_ep = worker_endpoints
        # print("w0_ep:", w0_ep, " w1_ep:", w1_ep)
        env0 = {
            "FLAGS_selected_mlus": "0",
            "PADDLE_TRAINER_ID": "0",
            "PADDLE_TRAINERS_NUM": "2",
            "PADDLE_TRAINER_ENDPOINTS": self._ps_endpoints,
            "PADDLE_CURRENT_ENDPOINT": w0_ep,
        }

        env1 = {
            "FLAGS_selected_mlus": "1",
            "PADDLE_TRAINER_ID": "1",
            "PADDLE_TRAINERS_NUM": "2",
            "PADDLE_TRAINER_ENDPOINTS": self._ps_endpoints,
            "PADDLE_CURRENT_ENDPOINT": w1_ep,
        }
        # update environment
        env0.update(envs)
        env1.update(envs)

        tr_cmd = "%s %s"
        tr0_cmd = tr_cmd % (self._python_interp, model_file)
        tr1_cmd = tr_cmd % (self._python_interp, model_file)
        tr0_pipe = open("/tmp/tr0_err_%d.log" % os.getpid(), "w")
        tr1_pipe = open("/tmp/tr1_err_%d.log" % os.getpid(), "w")
        print("tr0_cmd: {}, env: {}\n".format(tr0_cmd, env0))
        print("tr1_cmd: {}, env: {}\n".format(tr1_cmd, env1))
        tr0_proc = subprocess.Popen(
            tr0_cmd.strip().split(),
            stdout=subprocess.PIPE,
            stderr=tr0_pipe,
            env=env0,
        )

        tr1_proc = subprocess.Popen(
            tr0_cmd.strip().split(),
            stdout=subprocess.PIPE,
            stderr=tr1_pipe,
            env=env1,
        )

        tr0_out, tr0_err = tr0_proc.communicate()
        tr1_out, tr1_err = tr1_proc.communicate()

        sys.stderr.write('trainer 0 stderr: %s\n' % tr0_err)
        sys.stderr.write('trainer 1 stderr: %s\n' % tr1_err)
        # close trainer file
        tr0_pipe.close()
        tr1_pipe.close()
        with open("/tmp/tr0_err_%d.log" % os.getpid(), "r") as f:
            sys.stderr.write('trainer 0 stderr file: %s\n' % f.read())
        with open("/tmp/tr1_err_%d.log" % os.getpid(), "r") as f:
            sys.stderr.write('trainer 1 stderr file: %s\n' % f.read())
        return (
            pickle.loads(tr0_out),
            pickle.loads(tr1_out),
            tr0_proc.pid,
            tr1_proc.pid,
        )

    def check_with_place(
        self, model_file, col_type, check_error_log=False, need_envs={}
    ):
        required_envs = {
            "FLAGS_fraction_of_gpu_memory_to_use": "0.15",
            "FLAGS_eager_delete_tensor_gb": "0.0",
            "PATH": os.getenv("PATH"),
            "PYTHONPATH": os.getenv("PYTHONPATH", ""),
            "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", ""),
            "LD_PRELOAD": os.getenv("LD_PRELOAD", ""),
            "FLAGS_call_stack_level": "2",
            "GLOG_v": "3",
            "PADDLE_WITH_GLOO": '0',
            "BACKEND": "cncl",
        }
        required_envs.update(need_envs)
        if check_error_log:
            required_envs["GLOG_v"] = "3"
            required_envs["GLOG_logtostderr"] = "1"
            required_envs["GLOO_LOG_LEVEL"] = "TRACE"
        tr0_out, tr1_out, pid0, pid1 = self._run_cluster(
            model_file, required_envs
        )
        self.assertEqual(
            tr0_out, 'training, inference, fp32, fp16, NCHW, NHWC all passed'
        )
        self.assertEqual(
            tr1_out, 'training, inference, fp32, fp16, NCHW, NHWC all passed'
        )
