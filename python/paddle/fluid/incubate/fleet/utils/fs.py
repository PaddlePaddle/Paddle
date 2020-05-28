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
import os
import sys
import subprocess
import multiprocessing
from datetime import datetime

import re
import copy
import errno
import time
import logging
import abc
from pathlib import PurePosixPath, Path
import shutil

__all__ = ['FS', 'LocalFS']


class FS(object):
    @abc.abstractmethod
    def ls(self, fs_path):
        pass

    @abc.abstractmethod
    def is_file(self, fs_path):
        pass

    @abc.abstractmethod
    def is_dir(self, fs_path):
        pass

    @abc.abstractmethod
    def is_exist(self, fs_path):
        pass

    @abc.abstractmethod
    def upload(self, local_path, fs_path, overwrite=False):
        pass

    @abc.abstractmethod
    def download(self,
                 fs_path,
                 local_path,
                 multi_processes=5,
                 overwrite=False,
                 retry_times=5):
        pass

    @abc.abstractmethod
    def mkdirs(self, fs_path):
        pass

    @abc.abstractmethod
    def delete(self, fs_path):
        pass

    @abc.abstractmethod
    def need_upload_download(self):
        pass

    @abc.abstractmethod
    def rename(self, fs_src_path, fs_dst_path):
        pass

    @abc.abstractmethod
    def mv(self, fs_src_path, fs_dst_path):
        pass

    @abc.abstractmethod
    def upload_dir(self, local_dir, dest_dir, overwrite=False):
        pass


class LocalFS(FS):
    def ls(self, fs_path):
        return [f for f in os.listdir(fs_path)]

    def mkdirs(self, fs_path):
        assert not os.path.isfile(fs_path), "{} is already a file".format(
            fs_path)
        os.system("mkdir -p {}".format(fs_path))

    def rename(self, fs_src_path, fs_dst_path):
        os.rename(fs_src_path, fs_dst_path)

    def rename(self, fs_src_path, fs_dst_path):
        self.rename(fs_src_path, fs_dst_path)

    def _rmr(self, fs_path):
        shutil.rmtree(fs_path)

    def _rm(self, fs_path):
        os.remove(fs_path)

    def delete(self, fs_path):
        if not self.is_exist(fs_path):
            return

        if os.path.isfile(fs_path):
            return self._rm(fs_path)

        return self._rmr(fs_path)

    def need_upload_download(self):
        return False

    def is_file(self, fs_path):
        return os.path.isfile(fs_path)

    def is_dir(self, fs_path):
        return os.path.isdir(fs_path)

    def is_exist(self, fs_path):
        return os.path.exists(fs_path)

    def touch(self, fs_path):
        return Path(fs_path).touch()
