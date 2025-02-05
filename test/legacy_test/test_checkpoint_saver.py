# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

from paddle.base.incubate.checkpoint.checkpoint_saver import CheckpointSaver
from paddle.distributed.fleet.utils.fs import HDFSClient


class CheckpointSaverTest(unittest.TestCase):
    def test(self):
        fs = HDFSClient("/usr/local/hadoop-2.7.7", None)
        dir_path = "./checkpointsaver_test"
        fs.delete(dir_path)

        s = CheckpointSaver(fs)

        fs.mkdirs(f"{dir_path}/exe.exe")
        fs.mkdirs(f"{dir_path}/exe.1")
        fs.mkdirs(f"{dir_path}/exe")

        a = s.get_checkpoint_no(dir_path)
        self.assertEqual(len(a), 0)

        fs.mkdirs(f"{dir_path}/__paddle_checkpoint__.0")
        fs.mkdirs(f"{dir_path}/__paddle_checkpoint__.exe")

        a = s.get_checkpoint_no(dir_path)
        self.assertEqual(len(a), 1)

        s.clean_redundant_checkpoints(dir_path)
        s.clean_redundant_checkpoints(dir_path)

        fs.delete(dir_path)


if __name__ == '__main__':
    unittest.main()
