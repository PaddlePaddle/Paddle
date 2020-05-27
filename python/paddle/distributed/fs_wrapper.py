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

import paddle.fluid as fluid
import sys
import abc
import os
from pathlib import PurePosixPath
import shutil

__all__ = ['FS', 'LocalFS', 'HDFS']


class HDFS(FS):
    def __init__(self, hadoop_home, configs):
        self._client = HDFSClient(hadoop_home, configs)

    def list_dirs(self, fs_path):
        if not self.stat(fs_path):
            return []

        dirs, _ = self.ls_dir(fs_path)
        return dirs

    def ls_dir(self, fs_path):
        """
        list directory under fs_path, and only give the pure name, not include the fs_path
        """
        lines = self._client.ls(fs_path)

        dirs = []
        files = []
        for line in lines:
            arr = line.split()
            if len(arr) != 8:
                continue

            if fs_path not in arr[7]:
                continue

            p = PurePosixPath(arr[7])
            if arr[0][0] == 'd':
                dirs.append(p.name)
            else:
                files.append(p.name)

        return dirs, files

    def is_dir(self, fs_path):
        cmd = "{} -test -d {} ; echo $?".format(self._base_cmd, fs_path)

        test = self._run_cmd(cmd)
        if test[0].strip() == "0":
            return True

        return False

    def stat(self, fs_path):
        cmd = "{} -test -e {} ; echo $?".format(self._base_cmd, fs_path)

        test = self._run_cmd(cmd)
        if test[0].strip() == "0":
            return True

        return False

    def upload(self, local_path, fs_path):
        cmd = "{} -put {} {}".format(self._base_cmd, local_path, fs_path)
        fluid.core.run_cmd(cmd, self._time_out, self._sleep_inter)

    def download(self, fs_path, local_path):
        cmd = "{} -get {} {}/".format(self._base_cmd, fs_path, local_path)
        fluid.core.run_cmd(cmd, self._time_out, self._sleep_inter)

    def mkdir(self, fs_path):

        if not self.stat(fs_path):
            cmd = "{} -mkdir {}".format(self._base_cmd, fs_path)
            fluid.core.run_cmd(cmd, self._time_out, self._sleep_inter)

    def mv(self, fs_src_path, fs_dst_path):
        cmd = "{} -mv {} {}".format(self._base_cmd, fs_src_path, fs_dst_path)
        fluid.core.run_cmd(cmd, self._time_out, self._sleep_inter)

    def rmr(self, fs_path):
        if not self.stat(fs_path):
            return

        cmd = "{} -rmr {}".format(self._base_cmd, fs_path)
        return fluid.core.run_cmd(cmd, self._time_out, self._sleep_inter)

    def rm(self, fs_path):
        if not self.stat(fs_path):
            return

        cmd = "{} -rm {}".format(self._base_cmd, fs_path)
        return fluid.core.run_cmd(cmd, self._time_out, self._sleep_inter)

    def delete(self, fs_path):
        if not self.stat(fs_path):
            return

        is_dir = self.is_dir(fs_path)
        if is_dir:
            return self.rmr(fs_path)

        return self.rm(fs_path)

    def need_upload_download(self):
        return True
