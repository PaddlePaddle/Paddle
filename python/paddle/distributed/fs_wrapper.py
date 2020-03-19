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


class FS(object):
    @abc.abstractmethod
    def list_dirs(self, fs_path):
        pass

    @abc.abstractmethod
    def ls(self, fs_path):
        pass

    @abc.abstractmethod
    def stat(self, fs_path):
        pass

    @abc.abstractmethod
    def upload(self, local_path, fs_path):
        pass

    @abc.abstractmethod
    def download(self, fs_path, local_path):
        pass

    @abc.abstractmethod
    def mkdir(self, fs_path):
        pass

    @abc.abstractmethod
    def mv(self, fs_src_path, fs_dst_path):
        pass

    @abc.abstractmethod
    def rmr(self, fs_path):
        pass

    @abc.abstractmethod
    def rm(self, fs_path):
        pass

    @abc.abstractmethod
    def rm_dir_file(self, fs_path):
        pass

    @abc.abstractmethod
    def is_remote(self):
        pass


class LocalFS(FS):
    def list_dirs(self, fs_path):
        return [
            f for f in os.listdir(fs_path) if os.path.isdir(fs_path + "/" + f)
        ]

    def ls(self, fs_path):
        return [f for f in os.listdir(fs_path)]

    def stat(self, fs_path):
        return os.path.exists(fs_path)

    """
    def upload(self, local_path, fs_path):
        #os.symlink(local_path, fs_path)
        self.mv(local_path, fs_path)

    def download(self, fs_path, local_path):
        #os.symlink(fs_path, local_path)
        self.mv(fs_path, local_path)
    """

    def mkdir(self, fs_path):
        assert not os.path.isfile(fs_path), "{} is already a file".format(
            fs_path)
        os.system("mkdir -p {}".format(fs_path))

    def mv(self, fs_src_path, fs_dst_path):
        os.rename(fs_src_path, fs_dst_path)

    def rmr(self, fs_path):
        assert os.path.isdir(fs_path), "{} must be a directory".format(fs_path)
        shutil.rmtree(fs_path)

    def rm(self, fs_path):
        assert os.path.isfile(fs_path), "{} must be a file".format(fs_path)
        os.remove(fs_path)

    def rm_dir_file(self, fs_path):
        if not self.stat(fs_path):
            return

        if os.path.isfile(fs_path):
            self.rm(fs_path)

        if os.path.isdir(fs_path):
            self.rmr(fs_path)

    def is_remote(self):
        return False


class BDFS(FS):
    def __init__(self,
                 hdfs_name,
                 hdfs_ugi,
                 time_out=20 * 60 * 1000,
                 sleep_inter=1000):
        self._base_cmd = "hadoop fs -Dfs.default.name=\"{}\" -Dhadoop.job.ugi=\"{}\"".format(
            hdfs_name, hdfs_ugi)
        self._time_out = time_out
        self._sleep_inter = sleep_inter

    def _run_cmd(self, cmd):
        ret = fluid.core.run_cmd(cmd, self._time_out, self._sleep_inter)
        if len(ret) <= 0:
            return []

        lines = ret.splitlines()
        return lines

    def list_dirs(self, fs_path):
        dirs, _ = self.ls(fs_path)
        return dirs

    def ls(self, fs_path):
        cmd = "{} -ls {}".format(self._base_cmd, fs_path)
        lines = self._run_cmd(cmd)

        dirs = []
        files = []
        for line in lines:
            #print("line:", line)
            arr = line.split()
            #print(arr)
            if len(arr) != 8:
                continue

            if fs_path not in arr[7]:
                continue

            if arr[0][0] == 'd':
                dirs.append(arr[7])
            else:
                files.append(arr[7])

        return dirs, files

    def _is_dir_or_file(self, fs_path):
        cmd = "{} -ls {}".format(self._base_cmd, fs_path)
        dirs, files = self.ls(cmd)
        if fs_path in dirs:
            return True, False
        if fs_path in files:
            return False, True
        return False, False

    def stat(self, fs_path):
        cmd = "{} -stat {}/".format(self._base_cmd, fs_path)
        lines = self._run_cmd(cmd)
        for line in lines:
            if "No such file or directory" in line:
                return False
        return True

    def upload(self, local_path, fs_path):
        assert not self.stat(fs_path), "{} exists now".format(fs_path)
        assert self.stat(local_path), "{} not exists".format(local_path)

        cmd = "{} -put {} {}/".format(self._base_cmd, local_path, fs_path)
        fluid.core.run_cmd(cmd, self._time_out, self._sleep_inter)

    def download(self, fs_path, local_path):
        assert self.stat(fs_path), "{} not exists now".format(fs_path)
        assert not self.stat(local_path), "{} already exists".format(local_path)

        cmd = "{} -get {} {}/".format(self._base_cmd, fs_path, local_path)
        fluid.core.run_cmd(cmd, self._time_out, self._sleep_inter)

    def mkdir(self, fs_path):
        is_dir, is_file = self._is_dir_or_file(fs_path)
        assert not is_file, "{} is already be a file".format(fs_path)

        if not self.stat(fs_path):
            cmd = "{} -mkdir {}".format(self._base_cmd, fs_path)
            fluid.core.run_cmd(cmd, self._time_out, self._sleep_inter)

    def mv(self, fs_src_path, fs_dst_path):
        cmd = "{} -mv {} {}".format(self._base_cmd, fs_src_path, fs_dst_path)
        fluid.core.run_cmd(cmd, self._time_out, self._sleep_inter)

    def rmr(self, fs_path):
        if not self.stat(fs_path):
            return

        is_dir, _ = self.is_dir_or_file(fs_path)
        assert is_dir, "{} must be dir".format(fs_path)

        cmd = "{} -rmr {}".format(self._base_cmd, fs_path)
        return fluid.core.run_cmd(cmd, self._time_out, self._sleep_inter)

    def rm(self, fs_path):
        if not self.stat(fs_path):
            return

        _, is_file = self._is_dir_or_file(fs_path)
        assert is_file, "{} must be file".format(fs_path)

        cmd = "{} -rm {}".format(self._base_cmd, fs_path)
        return fluid.core.run_cmd(cmd, self._time_out, self._sleep_inter)

    def rm_dir_file(self, fs_path):
        if not self.stat(fs_path):
            return

        is_dir, is_file = self._is_dir_or_file(fs_path)
        if is_dir:
            return self.rmr(fs_path)

        if is_file:
            return self.rm(fs_path)

    def is_remote(self):
        return True
