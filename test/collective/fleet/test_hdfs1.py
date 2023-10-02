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

import os
import unittest

from hdfs_test_utils import FSTestBase

from paddle import base
from paddle.distributed.fleet.utils.fs import FSTimeOut, HDFSClient

java_home = os.environ["JAVA_HOME"]


class FSTest1(FSTestBase):
    def test_timeout(self):
        fs = HDFSClient(
            "/usr/local/hadoop-2.7.7/", None, time_out=6 * 1000, sleep_inter=100
        )
        src = "hdfs_test_timeout"
        dst = "new_hdfs_test_timeout"
        fs.delete(dst)
        fs.mkdirs(src)
        fs.mkdirs(dst)
        fs.mkdirs(dst + "/" + src)
        output = ""
        cmd = f"{fs._base_cmd} -mv {src} {dst}"
        try:
            fs.mv(src, dst, test_exists=False)
            self.assertFalse(1, f"can't execute cmd:{cmd} output:{output}")
        except FSTimeOut as e:
            print(f"execute mv {src} to {dst} timeout")

        ret, output = base.core.shell_execute_cmd(cmd, 6 * 1000, 2 * 1000)
        self.assertNotEqual(ret, 0)
        print(f"second mv ret:{ret} output:{output}")

    def test_is_dir(self):
        fs = HDFSClient(
            "/usr/local/hadoop-2.7.7/", None, time_out=6 * 1000, sleep_inter=100
        )
        self.assertFalse(fs.is_dir("./test_hdfs.py"))
        s = """
java.io.IOException: Input/output error
 responseErrorMsg : failed to getFileStatus, errorCode: 3, path: /user/PUBLIC_KM_Data/wangxi16/data/serving_model, lparam: d868f6bb6822c621, errorMessage: inner error
\tat org.apache.hadoop.util.FileSystemUtil.throwException(FileSystemUtil.java:164)
\tat org.apache.hadoop.util.FileSystemUtil.dealWithResponse(FileSystemUtil.java:118)
\tat org.apache.hadoop.lite.client.LiteClientImpl.getFileStatus(LiteClientImpl.java:696)
\tat org.apache.hadoop.fs.LibDFileSystemImpl.getFileStatus(LibDFileSystemImpl.java:297)
\tat org.apache.hadoop.fs.LiteFileSystem.getFileStatus(LiteFileSystem.java:514)
\tat org.apache.hadoop.fs.FsShell.test(FsShell.java:1092)
\tat org.apache.hadoop.fs.FsShell.run(FsShell.java:2285)
\tat org.apache.hadoop.util.ToolRunner.run(ToolRunner.java:65)
\tat org.apache.hadoop.util.ToolRunner.run(ToolRunner.java:79)
\tat org.apache.hadoop.fs.FsShell.main(FsShell.java:2353)
        """

        print("split lines:", s.splitlines())
        self.assertIsNotNone(fs._test_match(s.splitlines()))

    def test_config(self):
        config = {"fs.default.name": "hdfs://xxx", "hadoop.job.ugi": "ugi"}
        fs = HDFSClient(
            "/usr/local/hadoop-2.7.7/",
            config,
            time_out=6 * 1000,
            sleep_inter=100,
        )

    def test_exists(self):
        fs = HDFSClient(
            "/usr/local/hadoop-2.7.7/", None, time_out=6 * 1000, sleep_inter=100
        )
        self.assertFalse(fs.is_exist(os.path.abspath("./xxxx")))
        self.assertFalse(fs.is_dir(os.path.abspath("./xxxx")))
        self.assertTrue(fs.is_dir(os.path.abspath("./xxx/..")))
        dirs, files = fs.ls_dir(os.path.abspath("./test_hdfs1.py"))
        self.assertTrue(dirs == [])
        self.assertTrue(len(files) == 1)
        dirs, files = fs.ls_dir(os.path.abspath("./xxx/.."))


if __name__ == '__main__':
    unittest.main()
