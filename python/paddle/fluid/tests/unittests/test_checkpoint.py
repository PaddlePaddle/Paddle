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

import paddle.fluid as fluid
import unittest
import os
import tempfile


class TestCheckpoint(unittest.TestCase):
    def setUp(self):
        self.dirname = tempfile.mktemp()
        self.max_num_checkpoints = 3
        self.epoch_interval = 1
        self.step_interval = 1
        self.trainer_id = 0
        self.chief = self.trainer_id == 0
        self.place = fluid.CPUPlace()
        self.epoch_id = 100
        self.step_id = 20

    def test_checkpoint(self):
        self.save_checkpoint()
        serial = fluid.io.get_latest_checkpoint_serial(self.dirname)
        self.assertTrue(serial >= 0)
        trainer_args = ["epoch_id", "step_id"]
        epoch_id, step_id = fluid.io.load_trainer_args(
            self.dirname, serial, self.trainer_id, trainer_args)
        self.assertEqual(self.step_id, int(step_id))
        self.assertEqual(self.epoch_id, int(epoch_id))

        program = fluid.Program()
        with fluid.program_guard(program):
            exe = fluid.Executor(self.place)
            fluid.io.load_checkpoint(exe, self.dirname, serial, program)

        fluid.io.clean_checkpoint(self.dirname, delete_dir=True)
        self.assertFalse(os.path.isdir(self.dirname))

    def save_checkpoint(self):
        config = fluid.CheckpointConfig(self.dirname, self.max_num_checkpoints,
                                        self.epoch_interval, self.step_interval)

        trainer_args = {}
        trainer_args["epoch_id"] = self.epoch_id
        trainer_args["step_id"] = self.step_id

        program = fluid.Program()
        with fluid.program_guard(program):
            program.global_block().create_var(
                name="scale_0",
                psersistable=True,
                dtype="float32",
                shape=[32, 32])

            exe = fluid.Executor(self.place)
            for i in xrange(10):
                fluid.io.save_checkpoint(exe, config.checkpoint_dir,
                                         self.trainer_id, trainer_args, program,
                                         config.max_num_checkpoints)


if __name__ == '__main__':
    unittest.main()
