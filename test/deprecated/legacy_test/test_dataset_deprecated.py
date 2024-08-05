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
TestCases for Dataset,
including create, config, run, etc.
"""

import os
import tempfile
import unittest

import paddle
from paddle import base

paddle.enable_static()


class TestDatasetWithFetchHandler(unittest.TestCase):
    """
    Test Dataset With Fetch Handler. TestCases.
    """

    def net(self):
        """
        Test Dataset With Fetch Handler. TestCases.
        """
        slots = ["slot1", "slot2", "slot3", "slot4"]
        slots_vars = []
        poolings = []
        for slot in slots:
            data = paddle.static.data(
                name=slot, shape=[-1, 1], dtype="int64", lod_level=1
            )
            var = paddle.cast(x=data, dtype='float32')
            pool = paddle.static.nn.sequence_lod.sequence_pool(
                input=var, pool_type='AVERAGE'
            )

            slots_vars.append(data)
            poolings.append(pool)

        concated = paddle.concat(poolings, axis=1)
        fc = paddle.static.nn.fc(x=concated, activation='tanh', size=32)
        return slots_vars, fc

    def get_dataset(self, inputs, files):
        """
        Test Dataset With Fetch Handler. TestCases.

        Args:
            inputs(list): inputs of get_dataset
            files(list): files of  get_dataset
        """
        dataset = paddle.distributed.QueueDataset()
        dataset.init(
            batch_size=32, thread_num=2, pipe_command="cat", use_var=inputs
        )
        dataset.set_filelist(files)
        return dataset

    def setUp(self):
        """
        Test Dataset With Fetch Handler. TestCases.
        """
        self.temp_dir = tempfile.TemporaryDirectory()
        self.filename1 = os.path.join(
            self.temp_dir.name, "test_queue_dataset_run_a.txt"
        )
        self.filename2 = os.path.join(
            self.temp_dir.name, "test_queue_dataset_run_b.txt"
        )

        with open(self.filename1, "w") as f:
            data = "1 1 2 3 3 4 5 5 5 5 1 1\n"
            data += "1 2 2 3 4 4 6 6 6 6 1 2\n"
            data += "1 3 2 3 5 4 7 7 7 7 1 3\n"
            f.write(data)
        with open(self.filename2, "w") as f:
            data = "1 4 2 3 3 4 5 5 5 5 1 4\n"
            data += "1 5 2 3 4 4 6 6 6 6 1 5\n"
            data += "1 6 2 3 5 4 7 7 7 7 1 6\n"
            data += "1 7 2 3 6 4 8 8 8 8 1 7\n"
            f.write(data)

    def tearDown(self):
        """
        Test Dataset With Fetch Handler. TestCases.
        """
        self.temp_dir.cleanup()

    def test_dataset_none(self):
        """
        Test Dataset With Fetch Handler. TestCases.
        """
        slots_vars, out = self.net()
        files = [self.filename1, self.filename2]
        dataset = self.get_dataset(slots_vars, files)

        exe = base.Executor(base.CPUPlace())
        exe.run(base.default_startup_program())

        # test dataset->None
        try:
            exe.train_from_dataset(base.default_main_program(), None)
        except ImportError as e:
            print("warning: we skip trainer_desc_pb2 import problem in windows")
        except RuntimeError as e:
            error_msg = "dataset is need and should be initialized"
            self.assertEqual(error_msg, str(e))
        except Exception as e:
            self.assertTrue(False)

    def test_infer_from_dataset(self):
        """
        Test Dataset With Fetch Handler. TestCases.
        """
        slots_vars, out = self.net()
        files = [self.filename1, self.filename2]
        dataset = self.get_dataset(slots_vars, files)

        exe = base.Executor(base.CPUPlace())
        exe.run(base.default_startup_program())

        try:
            exe.infer_from_dataset(base.default_main_program(), dataset)
        except ImportError as e:
            print("warning: we skip trainer_desc_pb2 import problem in windows")
        except Exception as e:
            self.assertTrue(False)

    def test_fetch_handler(self):
        """
        Test Dataset With Fetch Handler. TestCases.
        """
        slots_vars, out = self.net()
        files = [self.filename1, self.filename2]
        dataset = self.get_dataset(slots_vars, files)

        exe = base.Executor(base.CPUPlace())
        exe.run(base.default_startup_program())

        fh = base.executor.FetchHandler(out.name)
        fh.help()

        try:
            exe.train_from_dataset(
                program=base.default_main_program(),
                dataset=dataset,
                fetch_handler=fh,
            )
        except ImportError as e:
            print("warning: we skip trainer_desc_pb2 import problem in windows")
        except RuntimeError as e:
            error_msg = "dataset is need and should be initialized"
            self.assertEqual(error_msg, str(e))
        except Exception as e:
            self.assertTrue(False)


if __name__ == '__main__':
    unittest.main()
