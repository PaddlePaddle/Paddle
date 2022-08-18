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

from paddle.fluid.tests.unittests.hdfs_test_utils import FSTestBase
import unittest
import os

from paddle.distributed.fleet.utils.fs import HDFSClient, LocalFS

java_home = os.environ["JAVA_HOME"]


class FSTest3(FSTestBase):

    def test_hdfs(self):
        fs = HDFSClient("/usr/local/hadoop-2.7.7/",
                        None,
                        time_out=5 * 1000,
                        sleep_inter=100)
        self._test_mkdirs(fs)
        self._test_list_dir(fs)
        self._test_try_upload(fs)
        self._test_try_download(fs)

        self._test_upload(fs)
        self._test_upload_dir(fs)
        self._test_download(fs)
        self._test_download_dir(fs)

    def test_local(self):
        fs = LocalFS()
        self._test_mkdirs(fs)
        self._test_list_dir(fs)
        self._test_try_upload(fs)
        self._test_try_download(fs)


if __name__ == '__main__':
    unittest.main()
