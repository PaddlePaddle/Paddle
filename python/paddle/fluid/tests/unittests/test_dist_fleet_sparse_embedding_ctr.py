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

import os
import shutil
import tempfile
import unittest
import paddle

paddle.enable_static()

import paddle.fluid as fluid

from test_dist_fleet_base import TestFleetBase
from dist_fleet_sparse_embedding_ctr import fake_ctr_reader


@unittest.skip(reason="Skip unstable ut, need paddle sync mode fix")
class TestDistMnistSync2x2(TestFleetBase):
    def _setup_config(self):
        self._mode = "sync"
        self._reader = "pyreader"

    def check_with_place(self,
                         model_file,
                         delta=1e-3,
                         check_error_log=False,
                         need_envs={}):
        required_envs = {
            "PATH": os.getenv("PATH", ""),
            "PYTHONPATH": os.getenv("PYTHONPATH", ""),
            "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", ""),
            "FLAGS_rpc_deadline": "5000",  # 5sec to fail fast
            "http_proxy": "",
            "CPU_NUM": "2"
        }

        required_envs.update(need_envs)

        if check_error_log:
            required_envs["GLOG_v"] = "3"
            required_envs["GLOG_logtostderr"] = "1"

        tr0_losses, tr1_losses = self._run_cluster(model_file, required_envs)

    def test_dist_train(self):
        self.check_with_place(
            "dist_fleet_sparse_embedding_ctr.py",
            delta=1e-5,
            check_error_log=True)


class TestDistMnistAsync2x2(TestFleetBase):
    def _setup_config(self):
        self._mode = "async"
        self._reader = "pyreader"

    def check_with_place(self,
                         model_file,
                         delta=1e-3,
                         check_error_log=False,
                         need_envs={}):
        required_envs = {
            "PATH": os.getenv("PATH", ""),
            "PYTHONPATH": os.getenv("PYTHONPATH", ""),
            "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", ""),
            "FLAGS_rpc_deadline": "5000",  # 5sec to fail fast
            "http_proxy": "",
            "CPU_NUM": "2"
        }

        required_envs.update(need_envs)

        if check_error_log:
            required_envs["GLOG_v"] = "3"
            required_envs["GLOG_logtostderr"] = "1"

        tr0_losses, tr1_losses = self._run_cluster(model_file, required_envs)

    def test_dist_train(self):
        self.check_with_place(
            "dist_fleet_sparse_embedding_ctr.py",
            delta=1e-5,
            check_error_log=True)


class TestDistMnistAsync2x2WithDecay(TestFleetBase):
    def _setup_config(self):
        self._mode = "async"
        self._reader = "pyreader"

    def check_with_place(self,
                         model_file,
                         delta=1e-3,
                         check_error_log=False,
                         need_envs={}):
        required_envs = {
            "PATH": os.getenv("PATH", ""),
            "PYTHONPATH": os.getenv("PYTHONPATH", ""),
            "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", ""),
            "FLAGS_rpc_deadline": "5000",  # 5sec to fail fast
            "http_proxy": "",
            "CPU_NUM": "2",
            "DECAY": "0"
        }

        required_envs.update(need_envs)

        if check_error_log:
            required_envs["GLOG_v"] = "3"
            required_envs["GLOG_logtostderr"] = "1"

        tr0_losses, tr1_losses = self._run_cluster(model_file, required_envs)

    def test_dist_train(self):
        self.check_with_place(
            "dist_fleet_sparse_embedding_ctr.py",
            delta=1e-5,
            check_error_log=True)


class TestDistMnistAsync2x2WithUnifrom(TestFleetBase):
    def _setup_config(self):
        self._mode = "async"
        self._reader = "pyreader"

    def check_with_place(self,
                         model_file,
                         delta=1e-3,
                         check_error_log=False,
                         need_envs={}):
        required_envs = {
            "PATH": os.getenv("PATH", ""),
            "PYTHONPATH": os.getenv("PYTHONPATH", ""),
            "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", ""),
            "FLAGS_rpc_deadline": "5000",  # 5sec to fail fast
            "http_proxy": "",
            "CPU_NUM": "2",
            "INITIALIZER": "1"
        }

        required_envs.update(need_envs)

        if check_error_log:
            required_envs["GLOG_v"] = "3"
            required_envs["GLOG_logtostderr"] = "1"

        tr0_losses, tr1_losses = self._run_cluster(model_file, required_envs)

    def test_dist_train(self):
        self.check_with_place(
            "dist_fleet_sparse_embedding_ctr.py",
            delta=1e-5,
            check_error_log=True)


@unittest.skip(reason="Skip unstable ut, need tensor table to enhance")
class TestDistMnistAsync2x2WithGauss(TestFleetBase):
    def _setup_config(self):
        self._mode = "async"
        self._reader = "pyreader"

    def _run_local_infer(self, model_file):
        def net():
            """
            network definition

            Args:
                batch_size(int): the size of mini-batch for training
                lr(float): learning rate of training
            Returns:
                avg_cost: LoDTensor of cost.
            """
            dnn_input_dim, lr_input_dim = 10, 10

            dnn_data = fluid.layers.data(
                name="dnn_data",
                shape=[-1, 1],
                dtype="int64",
                lod_level=1,
                append_batch_size=False)
            lr_data = fluid.layers.data(
                name="lr_data",
                shape=[-1, 1],
                dtype="int64",
                lod_level=1,
                append_batch_size=False)
            label = fluid.layers.data(
                name="click",
                shape=[-1, 1],
                dtype="int64",
                lod_level=0,
                append_batch_size=False)

            datas = [dnn_data, lr_data, label]

            inference = True
            init = fluid.initializer.Uniform()

            dnn_layer_dims = [128, 64, 32]
            dnn_embedding = fluid.contrib.layers.sparse_embedding(
                input=dnn_data,
                size=[dnn_input_dim, dnn_layer_dims[0]],
                is_test=inference,
                param_attr=fluid.ParamAttr(
                    name="deep_embedding", initializer=init))
            dnn_pool = fluid.layers.sequence_pool(
                input=dnn_embedding, pool_type="sum")
            dnn_out = dnn_pool
            for i, dim in enumerate(dnn_layer_dims[1:]):
                fc = fluid.layers.fc(
                    input=dnn_out,
                    size=dim,
                    act="relu",
                    param_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.Constant(value=0.01)),
                    name='dnn-fc-%d' % i)
                dnn_out = fc

            # build lr model
            lr_embbding = fluid.contrib.layers.sparse_embedding(
                input=lr_data,
                size=[lr_input_dim, 1],
                is_test=inference,
                param_attr=fluid.ParamAttr(
                    name="wide_embedding",
                    initializer=fluid.initializer.Constant(value=0.01)))

            lr_pool = fluid.layers.sequence_pool(
                input=lr_embbding, pool_type="sum")
            merge_layer = fluid.layers.concat(input=[dnn_out, lr_pool], axis=1)
            predict = fluid.layers.fc(input=merge_layer, size=2, act='softmax')
            return datas, predict

        reader = paddle.batch(fake_ctr_reader(), batch_size=4)
        datas, predict = net()
        exe = fluid.Executor(fluid.CPUPlace())
        feeder = fluid.DataFeeder(place=fluid.CPUPlace(), feed_list=datas)
        exe.run(fluid.default_startup_program())

        fluid.io.load_persistables(exe, model_file)

        for batch_id, data in enumerate(reader()):
            score = exe.run(fluid.default_main_program(),
                            feed=feeder.feed(data),
                            fetch_list=[predict])

    def check_with_place(self,
                         model_file,
                         delta=1e-3,
                         check_error_log=False,
                         need_envs={}):
        model_dir = tempfile.mkdtemp()
        required_envs = {
            "PATH": os.getenv("PATH", ""),
            "PYTHONPATH": os.getenv("PYTHONPATH", ""),
            "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", ""),
            "FLAGS_rpc_deadline": "5000",  # 5sec to fail fast
            "http_proxy": "",
            "CPU_NUM": "2",
            "INITIALIZER": "2",
            "MODEL_DIR": model_dir
        }

        required_envs.update(need_envs)
        if check_error_log:
            required_envs["GLOG_v"] = "3"
            required_envs["GLOG_logtostderr"] = "1"

        self._run_cluster(model_file, required_envs)
        self._run_local_infer(model_dir)
        shutil.rmtree(model_dir)

    def test_dist_train(self):
        self.check_with_place(
            "dist_fleet_sparse_embedding_ctr.py",
            delta=1e-5,
            check_error_log=True)


if __name__ == "__main__":
    unittest.main()
