#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
TestCases for Monitor
"""

import paddle

paddle.enable_static()

import os
import tempfile
import unittest

from paddle import base
from paddle.base import core


class TestDatasetWithStat(unittest.TestCase):
    """TestCases for Dataset."""

    def setUp(self):
        self.use_data_loader = False
        self.epoch_num = 10
        self.drop_last = False

    def test_dataset_run_with_stat(self):
        temp_dir = tempfile.TemporaryDirectory()
        path_a = os.path.join(temp_dir.name, "test_in_memory_dataset_run_a.txt")
        path_b = os.path.join(temp_dir.name, "test_in_memory_dataset_run_b.txt")
        with open(path_a, "w") as f:
            data = "1 1 2 3 3 4 5 5 5 5 1 1\n"
            data += "1 2 2 3 4 4 6 6 6 6 1 2\n"
            data += "1 3 2 3 5 4 7 7 7 7 1 3\n"
            f.write(data)
        with open(path_b, "w") as f:
            data = "1 4 2 3 3 4 5 5 5 5 1 4\n"
            data += "1 5 2 3 4 4 6 6 6 6 1 5\n"
            data += "1 6 2 3 5 4 7 7 7 7 1 6\n"
            data += "1 7 2 3 6 4 8 8 8 8 1 7\n"
            f.write(data)

        slots = ["slot1", "slot2", "slot3", "slot4"]
        slots_vars = []
        for slot in slots:
            var = paddle.static.data(
                name=slot, shape=[-1, 1], dtype="int64", lod_level=1
            )
            slots_vars.append(var)

        embs = []
        for x in slots_vars:
            emb = paddle.nn.Embedding(
                num_embeddings=100001, embedding_dim=4, sparse=True
            )(x)
            embs.append(emb)

        dataset = paddle.distributed.InMemoryDataset()
        dataset._set_batch_size(32)
        dataset._set_thread(3)
        dataset.set_filelist([path_a, path_b])
        dataset._set_pipe_command("cat")
        dataset._set_use_var(slots_vars)
        dataset.load_into_memory()
        dataset._set_fea_eval(1, True)
        dataset.slots_shuffle(["slot1"])

        exe = base.Executor(base.CPUPlace())
        exe.run(base.default_startup_program())
        if self.use_data_loader:
            data_loader = base.io.DataLoader.from_dataset(
                dataset, base.cpu_places(), self.drop_last
            )
            for i in range(self.epoch_num):
                for data in data_loader():
                    exe.run(base.default_main_program(), feed=data)

        else:
            for i in range(self.epoch_num):
                try:
                    exe.train_from_dataset(
                        base.default_main_program(),
                        dataset,
                        fetch_list=[embs[0], embs[1]],
                        fetch_info=["emb0", "emb1"],
                        print_period=1,
                    )

                except Exception as e:
                    self.assertTrue(False)

        int_stat = core.get_int_stats()
        # total 56 keys
        print(int_stat["STAT_total_feasign_num_in_mem"])

        temp_dir.cleanup()


if __name__ == '__main__':
    unittest.main()
