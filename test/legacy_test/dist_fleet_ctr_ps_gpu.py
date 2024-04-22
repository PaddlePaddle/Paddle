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
"""
Distribute CTR model for test fleet api
"""

import os
import shutil
import tempfile
import time

import ctr_dataset_reader
import numpy as np
from dist_fleet_ctr import TestDistCTR2x2, fake_ctr_reader
from test_dist_fleet_base import runtime_main

import paddle
from paddle import base

# Fix seed for test
paddle.seed(1)


class TestDistGpuPsCTR2x2(TestDistCTR2x2):
    """
    For test CTR model, using Fleet api & PS-GPU
    """

    def check_model_right(self, dirname):
        model_filename = os.path.join(dirname, "__model__")

        with open(model_filename, "rb") as f:
            program_desc_str = f.read()

        program = base.Program.parse_from_string(program_desc_str)
        with open(os.path.join(dirname, "__model__.proto"), "w") as wn:
            wn.write(str(program))

    def do_pyreader_training(self, fleet):
        """
        do training using dataset, using fetch handler to catch variable
        Args:
            fleet(Fleet api): the fleet object of Parameter Server, define distribute training role
        """
        device_id = int(os.getenv("FLAGS_selected_gpus", "0"))
        place = base.CUDAPlace(device_id)
        exe = base.Executor(place)

        exe.run(fleet.startup_program)
        fleet.init_worker()

        batch_size = 4
        train_reader = paddle.batch(fake_ctr_reader(), batch_size=batch_size)
        self.reader.decorate_sample_list_generator(train_reader)

        for epoch_id in range(1):
            self.reader.start()
            try:
                pass_start = time.time()
                while True:
                    loss_val = exe.run(
                        program=fleet.main_program,
                        fetch_list=[self.avg_cost.name],
                    )
                    loss_val = np.mean(loss_val)
                    reduce_output = fleet.util.all_reduce(
                        np.array(loss_val), mode="sum"
                    )
                    loss_all_trainer = fleet.util.all_gather(float(loss_val))
                    loss_val = float(reduce_output) / len(loss_all_trainer)
                    message = f"TRAIN ---> pass: {epoch_id} loss: {loss_val}\n"
                    fleet.util.print_on_rank(message, 0)

                pass_time = time.time() - pass_start
            except base.core.EOFException:
                self.reader.reset()

        model_dir = tempfile.mkdtemp()
        fleet.save_inference_model(
            exe, model_dir, [feed.name for feed in self.feeds], self.avg_cost
        )
        if fleet.is_first_worker():
            self.check_model_right(model_dir)
        if fleet.is_first_worker():
            fleet.save_persistables(executor=exe, dirname=model_dir)
        shutil.rmtree(model_dir)

    def do_dataset_training(self, fleet):
        (
            dnn_input_dim,
            lr_input_dim,
            train_file_path,
        ) = ctr_dataset_reader.prepare_data()

        device_id = int(os.getenv("FLAGS_selected_gpus", "0"))
        place = base.CUDAPlace(device_id)
        exe = base.Executor(place)

        exe.run(fleet.startup_program)
        fleet.init_worker()

        thread_num = 2
        batch_size = 128
        filelist = []
        for _ in range(thread_num):
            filelist.append(train_file_path)

        # config dataset
        dataset = paddle.distributed.QueueDataset()
        dataset._set_batch_size(batch_size)
        dataset._set_use_var(self.feeds)
        pipe_command = 'python ctr_dataset_reader.py'
        dataset._set_pipe_command(pipe_command)

        dataset.set_filelist(filelist)
        dataset._set_thread(thread_num)

        for epoch_id in range(1):
            pass_start = time.time()
            dataset.set_filelist(filelist)
            exe.train_from_dataset(
                program=fleet.main_program,
                dataset=dataset,
                fetch_list=[self.avg_cost],
                fetch_info=["cost"],
                print_period=2,
                debug=int(os.getenv("Debug", "0")),
            )
            pass_time = time.time() - pass_start

        if os.getenv("SAVE_MODEL") == "1":
            model_dir = tempfile.mkdtemp()
            fleet.save_inference_model(
                exe,
                model_dir,
                [feed.name for feed in self.feeds],
                self.avg_cost,
            )
            if fleet.is_first_worker():
                self.check_model_right(model_dir)
            if fleet.is_first_worker():
                fleet.save_persistables(executor=exe, dirname=model_dir)
            shutil.rmtree(model_dir)


if __name__ == "__main__":
    runtime_main(TestDistGpuPsCTR2x2)
