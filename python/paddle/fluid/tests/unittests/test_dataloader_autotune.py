#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import json
import os
import sys
import tempfile
import unittest
import warnings

import numpy as np

import paddle
import paddle.nn as nn
from paddle.io import DataLoader, Dataset


class RandomDataset(Dataset):
=======
from __future__ import print_function
import unittest
import numpy as np
import tempfile
import warnings
import json
import paddle
import paddle.nn as nn
from paddle.io import Dataset, DataLoader, BatchSampler, SequenceSampler
import sys
import os


class RandomDataset(Dataset):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([10]).astype('float32')
<<<<<<< HEAD
        label = np.random.randint(0, 10 - 1, (1,)).astype('int64')
=======
        label = np.random.randint(0, 10 - 1, (1, )).astype('int64')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        return image, label

    def __len__(self):
        return self.num_samples


class SimpleNet(nn.Layer):
<<<<<<< HEAD
    def __init__(self):
        super().__init__()
=======

    def __init__(self):
        super(SimpleNet, self).__init__()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.fc = nn.Linear(10, 10)

    def forward(self, image):
        return self.fc(image)


class TestAutoTune(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.batch_size = 1
        self.dataset = RandomDataset(10)

    def test_dataloader_use_autotune(self):
        paddle.incubate.autotune.set_config(
<<<<<<< HEAD
            config={
                "dataloader": {
                    "enable": True,
                    "tuning_steps": 1,
                }
            }
        )
        loader = DataLoader(
            self.dataset, batch_size=self.batch_size, num_workers=0
        )
=======
            config={"dataloader": {
                "enable": True,
                "tuning_steps": 1,
            }})
        loader = DataLoader(self.dataset,
                            batch_size=self.batch_size,
                            num_workers=0)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_dataloader_disable_autotune(self):
        config = {"dataloader": {"enable": False, "tuning_steps": 1}}
        tfile = tempfile.NamedTemporaryFile(mode="w+", delete=False)
        json.dump(config, tfile)
        tfile.close()
        paddle.incubate.autotune.set_config(tfile.name)
        os.remove(tfile.name)
<<<<<<< HEAD
        loader = DataLoader(
            self.dataset, batch_size=self.batch_size, num_workers=2
        )
        if sys.platform == 'darwin' or sys.platform == 'win32':
=======
        loader = DataLoader(self.dataset,
                            batch_size=self.batch_size,
                            num_workers=2)
        if (sys.platform == 'darwin' or sys.platform == 'win32'):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            self.assertEqual(loader.num_workers, 0)
        else:
            self.assertEqual(loader.num_workers, 2)

    def test_distributer_batch_sampler_autotune(self):
        paddle.incubate.autotune.set_config(
<<<<<<< HEAD
            config={
                "dataloader": {
                    "enable": True,
                    "tuning_steps": 1,
                }
            }
        )
        batch_sampler = paddle.io.DistributedBatchSampler(
            self.dataset, batch_size=self.batch_size
        )
        loader = DataLoader(
            self.dataset, batch_sampler=batch_sampler, num_workers=2
        )


class TestAutoTuneAPI(unittest.TestCase):
=======
            config={"dataloader": {
                "enable": True,
                "tuning_steps": 1,
            }})
        batch_sampler = paddle.io.DistributedBatchSampler(
            self.dataset, batch_size=self.batch_size)
        loader = DataLoader(self.dataset,
                            batch_sampler=batch_sampler,
                            num_workers=2)


class TestAutoTuneAPI(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_set_config_warnings(self):
        with warnings.catch_warnings(record=True) as w:
            config = {"kernel": {"enable": 1, "tuning_range": True}}
            tfile = tempfile.NamedTemporaryFile(mode="w+", delete=False)
            json.dump(config, tfile)
            tfile.close()
            paddle.incubate.autotune.set_config(tfile.name)
            os.remove(tfile.name)
            self.assertTrue(len(w) == 2)


if __name__ == '__main__':
    unittest.main()
