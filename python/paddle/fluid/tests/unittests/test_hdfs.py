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
import sys

from paddle.fluid.incubate.fleet.utils.fs import LocalFS
from paddle.fluid.incubate.fleet.utils.hdfs import HDFSClient


class FSTest(unittest.TestCase):
    def _test_dirs(self, fs):
        dir_path = os.path.abspath("./test_dir")
        fs.delete(dir_path)
        self.assertTrue(not fs.is_exist(dir_path))

        fs.mkdirs(dir_path)
        self.assertTrue(fs.is_exist(dir_path))
        self.assertTrue(not fs.is_file(dir_path))
        self.assertTrue(fs.is_dir(dir_path))

        new_dir_path = os.path.abspath("./new_test_dir")
        fs.mv(dir_path, new_dir_path)
        self.assertTrue(fs.is_exist(new_dir_path))

        fs.mv(new_dir_path, dir_path)
        self.assertTrue(fs.is_exist(dir_path))

        fs.delete(dir_path)
        self.assertTrue(not fs.is_exist(dir_path))

    def _test_touch_file(self, fs):
        file_path = os.path.abspath("./test_file")

        fs.delete(file_path)
        self.assertTrue(not fs.is_exist(file_path))

        fs.touch(file_path)
        self.assertTrue(fs.is_exist(file_path))
        self.assertTrue(not fs.is_dir(file_path) and fs.is_file(file_path))

        new_file_path = os.path.abspath("./new_test_file")
        fs.mv(file_path, new_file_path)
        self.assertTrue(fs.is_exist(new_file_path))

        fs.mv(new_file_path, file_path)
        self.assertTrue(fs.is_exist(file_path))

        fs.delete(file_path)
        self.assertTrue(not fs.is_exist(file_path))

    def _test_upload_file(self, fs):
        src_file = os.path.abspath("./test_upload.src")
        dst_file = os.path.abspath("./test_uolpad.dst")

        local = LocalFS()
        local.touch(src_file)
        fs.delete(dst_file)

        assert fs.need_upload_download()

        fs.upload(src_file, dst_file)
        self.assertTrue(fs.is_exist(dst_file))
        fs.delete(dst_file)
        fs.delete(src_file)

    def test_exists(self):
        fs = HDFSClient("/usr/local/hadoop-2.7.7/", None, time_out=15 * 1000)
        #self.assertFalse(fs.is_exist(os.path.abspath("./xxxx")))
        #self.assertFalse(fs.is_dir(os.path.abspath("./xxxx")))
        #self.assertTrue(fs.is_dir(os.path.abspath("./xxx/..")))
        dirs, files = fs.ls_dir(
            os.path.abspath(
                "/paddle/build/build_ubuntu_fixtypo_release_gpu_y_grpc_2.7/python/paddle/fluid/tests/unittests/test_hdfs.py"
            ))
        dirs, files = fs.ls_dir(os.path.abspath("./xxx/.."))
        print(dirs, files)

    """
    def test_hdfs(self):
        fs = HDFSClient("/usr/local/hadoop-2.7.7/", None)
        self._test_dirs(fs)
        self._test_upload_file(fs)

    def test_local(self):
        fs = LocalFS()
        self._test_dirs(fs)
        self._test_touch_file(fs)
    """


if __name__ == '__main__':
    unittest.main()
