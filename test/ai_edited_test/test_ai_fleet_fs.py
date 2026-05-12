# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

# [AUTO-GENERATED] Test file for paddle.distributed.fleet.utils.fs
# 覆盖模块: paddle/distributed/fleet/utils/fs.py
# Uncovered lines: 170-432

import os
import tempfile
import unittest

from paddle.distributed.fleet.utils.fs import LocalFS


class TestLocalFS(unittest.TestCase):
    """测试 LocalFS 类
    Test LocalFS class"""

    def setUp(self):
        self.local_fs = LocalFS()
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_local_fs_is_exist(self):
        """测试 LocalFS is_exist 方法
        Test LocalFS is_exist method"""
        test_file = os.path.join(self.test_dir, "test.txt")
        self.assertFalse(self.local_fs.is_exist(test_file))
        self.local_fs.touch(test_file)
        self.assertTrue(self.local_fs.is_exist(test_file))

    def test_local_fs_ls_dir(self):
        """测试 LocalFS ls_dir 方法
        Test LocalFS ls_dir method"""
        for i in range(3):
            path = os.path.join(self.test_dir, f"file_{i}.txt")
            with open(path, "w") as f:
                f.write(f"content_{i}")
        result = self.local_fs.ls_dir(self.test_dir)
        # ls_dir returns (dirs, files) tuple
        self.assertIsNotNone(result)

    def test_local_fs_mkdirs(self):
        """测试 LocalFS mkdirs 方法
        Test LocalFS mkdirs method"""
        new_dir = os.path.join(self.test_dir, "new_dir", "sub_dir")
        self.local_fs.mkdirs(new_dir)
        self.assertTrue(os.path.exists(new_dir))

    def test_local_fs_touch(self):
        """测试 LocalFS touch 方法
        Test LocalFS touch method"""
        test_file = os.path.join(self.test_dir, "touch_test.txt")
        self.local_fs.touch(test_file)
        self.assertTrue(os.path.exists(test_file))

    def test_local_fs_delete(self):
        """测试 LocalFS delete 方法
        Test LocalFS delete method"""
        test_file = os.path.join(self.test_dir, "del_test.txt")
        self.local_fs.touch(test_file)
        self.assertTrue(os.path.exists(test_file))
        self.local_fs.delete(test_file)
        self.assertFalse(os.path.exists(test_file))

    def test_local_fs_is_dir(self):
        """测试 LocalFS is_dir 方法
        Test LocalFS is_dir method"""
        self.assertTrue(self.local_fs.is_dir(self.test_dir))
        test_file = os.path.join(self.test_dir, "file.txt")
        self.local_fs.touch(test_file)
        self.assertFalse(self.local_fs.is_dir(test_file))

    def test_local_fs_is_file(self):
        """测试 LocalFS is_file 方法
        Test LocalFS is_file method"""
        test_file = os.path.join(self.test_dir, "file.txt")
        self.local_fs.touch(test_file)
        self.assertTrue(self.local_fs.is_file(test_file))
        self.assertFalse(self.local_fs.is_file(self.test_dir))

    def test_local_fs_cat_not_implemented(self):
        """测试 LocalFS cat 方法 (not implemented)
        Test LocalFS cat method (not implemented)"""
        test_file = os.path.join(self.test_dir, "cat_test.txt")
        with open(test_file, "w") as f:
            f.write("hello world")
        with self.assertRaises(NotImplementedError):
            self.local_fs.cat(test_file)

    def test_local_fs_mv(self):
        """测试 LocalFS mv 方法
        Test LocalFS mv method"""
        src = os.path.join(self.test_dir, "src.txt")
        dst = os.path.join(self.test_dir, "dst.txt")
        self.local_fs.touch(src)
        self.local_fs.mv(src, dst)
        self.assertFalse(os.path.exists(src))
        self.assertTrue(os.path.exists(dst))


if __name__ == '__main__':
    unittest.main()
