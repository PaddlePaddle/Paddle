#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid
import paddle.fluid.layers as layers
import numpy as np
import os
import paddle.fluid.core as core
import unittest
from paddle.fluid.layers.nn import _pull_box_sparse
from paddle.fluid.transpiler import collective


class TestTranspile(unittest.TestCase):
    """  TestCases for BoxPS Preload """

    def get_transpile(self, mode, trainers="127.0.0.1:6174"):
        config = fluid.DistributeTranspilerConfig()
        config.mode = 'collective'
        config.collective_mode = mode
        t = fluid.DistributeTranspiler(config=config)
        return t

    def test_transpile(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        t = self.get_transpile("single_process_multi_thread")
        t.transpile(
            trainer_id=0,
            startup_program=startup_program,
            trainers="127.0.0.1:6174",
            program=main_program)
        t = self.get_transpile("grad_allreduce")
        try:
            t.transpile(
                trainer_id=0,
                startup_program=startup_program,
                trainers="127.0.0.1:6174",
                program=main_program)
        except ValueError as e:
            print(e)

    def test_single_trainers(self):
        transpiler = collective.GradAllReduce(0)
        try:
            transpiler.transpile(
                startup_program=fluid.Program(),
                main_program=fluid.Program(),
                rank=1,
                endpoints="127.0.0.1:6174",
                current_endpoint="127.0.0.1:6174",
                wait_port="6174")
        except ValueError as e:
            print(e)
        transpiler = collective.LocalSGD(0)
        try:
            transpiler.transpile(
                startup_program=fluid.Program(),
                main_program=fluid.Program(),
                rank=1,
                endpoints="127.0.0.1:6174",
                current_endpoint="127.0.0.1:6174",
                wait_port="6174")
        except ValueError as e:
            print(e)


class TestBoxPSPreload(unittest.TestCase):
    """  TestCases for BoxPS Preload """

    def test_boxps_cpu(self):
        self.run_boxps_preload(True)

    def test_boxps_gpu(self):
        self.run_boxps_preload(False)

    def run_boxps_preload(self, is_cpu=True):
        x = fluid.layers.data(name='x', shape=[1], dtype='int64', lod_level=0)
        y = fluid.layers.data(name='y', shape=[1], dtype='int64', lod_level=0)
        emb_x, emb_y = _pull_box_sparse([x, y], size=2)
        emb_xp = _pull_box_sparse(x, size=2)
        concat = layers.concat([emb_x, emb_y], axis=1)
        fc = layers.fc(input=concat,
                       name="fc",
                       size=1,
                       num_flatten_dims=1,
                       bias_attr=False)
        loss = layers.reduce_mean(fc)
        layers.Print(loss)
        place = fluid.CPUPlace() if is_cpu or not core.is_compiled_with_cuda(
        ) else fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        batch_size = 2

        def binary_print(slot, fout):
            fout.write(str(len(slot)) + " ")
            for e in slot:
                fout.write(str(e) + " ")

        batch1 = np.ones(
            (batch_size, 2, 1)).astype("int64").reshape(batch_size, 2, 1)
        filelist = []
        place_str = "cpu" if is_cpu else "gpu"
        for i in range(2):
            filelist.append("test_hdfs_" + place_str + "_" + str(i))
        for f in filelist:
            with open(f, "w") as fout:
                for ins in batch1:
                    for slot in ins:
                        binary_print(slot, fout)
                fout.write("\n")

        def create_dataset():
            dataset = fluid.DatasetFactory().create_dataset("BoxPSDataset")
            dataset.set_date("20190930")
            dataset.set_use_var([x, y])
            dataset.set_batch_size(2)
            dataset.set_thread(1)
            dataset.set_filelist(filelist)
            return dataset

        datasets = []
        datasets.append(create_dataset())
        datasets.append(create_dataset())
        optimizer = fluid.optimizer.SGD(learning_rate=0.5)
        optimizer = fluid.optimizer.PipelineOptimizer(
            optimizer,
            cut_list=[],
            place_list=[place],
            concurrency_list=[1],
            queue_size=1,
            sync_steps=-1)
        optimizer.minimize(loss)
        exe.run(fluid.default_startup_program())
        datasets[0].load_into_memory()
        datasets[0].begin_pass()
        datasets[1].preload_into_memory()
        exe.train_from_dataset(
            program=fluid.default_main_program(),
            dataset=datasets[0],
            print_period=1)
        datasets[0].end_pass()
        datasets[1].wait_preload_done()
        datasets[1].begin_pass()
        exe.train_from_dataset(
            program=fluid.default_main_program(),
            dataset=datasets[1],
            print_period=1,
            debug=True)
        datasets[1].end_pass()
        for f in filelist:
            os.remove(f)


if __name__ == '__main__':
    unittest.main()
