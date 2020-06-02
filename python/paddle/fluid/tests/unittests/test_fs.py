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
import paddle.fluid as fluid
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.collective import CollectiveOptimizer, fleet, TrainStatus
import os

from paddle.distributed.fs import LocalFS
from paddle.distributed.hdfs import HDFSClient


class FSTest(unittest.TestCase):
    def _fs_test(self, fs, dir_path):
        # file
        file_path = dir_path + "/" + "fs_test.tmp"
        self.assertTrue(not fs.is_exist(file_path))

        if self.need_upload_download:
            local_fs = LocalFS()
            src_file_path = file_path + ".src"
            fs.upload(src_file_path, file_path)
            self.assertTrue(fs.is_exist(file_path))
        else:
            fs.touch(file_path)
            self.assertTrue(fs.is_exist(file_path))

        self.assertTrue(not fs.is_dir(file_path))

        new_file_path = fs.mv(file_path, new_dir_path)
        self.assertTrue(fs.is_exist(new_file_path))

        # dir
        test_dir_path = dir_path + "/dir_test.tmp"
        self.assertTrue(not fs.is_exist(test_dir_path))
        fs.mkdirs(test_dir_path)
        self.assertTrue(fs.is_exist(test_dir_path))
        self.assertTrue(not fs.is_file(test_dir_path))
        self.assertTrue(fs.is_dir(test_dir_path))

        # mv dir
        new_dir_path = dir_path + "/dir_test.new"
        fs.mv(test_dir_path, new_dir_path)
        self.assertTrue(fs.is_exist(new_dir_path))
        self.assertTrue(fs.is_dir(new_dir_path))

        # delete
        fs.delete(new_file_path)
        self.assertTrue(not fs.is_exist(new_file_path))
        fs.delete(new_dir_path)
        self.assertTrue(not fs.is_exist(new_dir_path))

    def setUp(self):
        fs = LocalFS()
        fs.mkdirs("./fs_test_hdfs")
        fs.mkdirs("./fs_test_local")

    def test_hdfs(self):
        fs = BDFS("/usr/lib/jvm/java-8-openjdk-amd64", None)
        dir_path = "./fs_test_hdfs"
        self._fs_test(fs, dir_path)

    def test_local(self):
        fs = LocalFS()
        dir_path = "./fs_test_local"
        self._fs_test(fs, dir_path)


if __name__ == '__main__':
    unittest.main()
