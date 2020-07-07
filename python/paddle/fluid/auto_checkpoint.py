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

from .wrapped_decorator import signature_safe_contextmanager
import sys
import logging
import hashlib
import json
from contextlib import contextmanager
from . import unique_name
from .checkpointer import SerializableBase
from paddle.fluid import core
from paddle.fluid import framework
import paddle.fluid as fluid
from fluid.incubate.fleet.utils.hdfs import HDFSClient
from . import compiler
from threading import Thread, current_thread


class AutoCheckpointChecker(object):
    def __init__(self):
        self._run_env = None
        self._plat_form = None
        self._job_id = None
        self._hdfs_home = None
        self._hdfs_name = None
        self._hdfs_ugi = None
        self._hdfs_checkpoint_path = None
        self._trainer_id = None
        try:
            self._run_env = os.environ("PADDLE_RUNNING_ENV")
            self._plat_form = os.environ("PADDLE_RUNNING_PLATFORM")
            self._job_id = os.environ("PADDLE_JOB_ID")
            self._hdfs_home = os.environ("PADDLE_EDL_HDFS_HOME")
            self._hdfs_name = os.environ("PADDLE_EDL_HDFS_NAME")
            self._hdfs_ugi = os.environ("PADDLE_EDL_HDFS_UGI")
            self._hdfs_checkpoint_path = os.environ(
                "PADDLE_EDL_HDFS_CHECKPOINT_PATH")
            self._trainer_id = int(os.getenv("PADDLE_EDL_TRAINER_ID"))
        except Exception as e:
            #logging.warning("auto checkpoint must run under PADDLE_RUNNING_ENV,PADDLE_RUNNING_PLATFORM,PADDLE_JOB_ID:{}".format(e))
            return

    def get_job_checkpoint_path(self):
        return "{}/{}".format(self._hdfs_checkpoint_path, self._job_id)

    def valid(self):
        return self._run_env is not None and \
            self._plat_form is not None and \
            self._job_id is not None and \
            self._hdfs_home is not None and \
            self._hdfs_name is not None and \
            self._hdfs_ugi is not None and \
            self._hdfs_checkpoint_path is not None and \
            self._trainer_id is not None

    @property
    def trainer_id(self):
        return self._trainer_id

    @property
    def run_env(self):
        return self.run_env

    @property
    def plat_form(self):
        return self.plat_form

    @property
    def job_id(self):
        return self.job_id

    @property
    def hdfs_home(self):
        return self._hdfs_home

    @property
    def hdfs_ugi(self):
        return self.hdfs_ugi

    @property
    def hdfs_checkpoint_path(self):
        return self._hdfs_checkpoint_path


class ExeTrainStatus(SerializableBase):
    def __init__(self):
        self._epoch_no = -1
        self._hash_key = None
        self._key = None
        self._checkpoint_path = None
        self._checkpoint_no = None
        self._restored = False
        self._exe = None
        self._program = None

        self._file_name = "exe_train_status"

    def next(self):
        return self._epoch_no + 1

    def __eq__(self, t):
        return self._epoch_no == t._epoch_no and \
            self._hash_key == t._hash_key and \
            self._key == t._key

    def __ne__(self, t):
        return not self == t

    def serialize(self, path):
        file_name = "{}/{}".format(path, self._file_name)
        with open(file_name, 'w') as f:
            s = self._serialize()
            f.write(s)

    def _serialize(self, pop_keys=["restored"]):
        d = self._to_dict()
        for k in pop_keys:
            d.pop(k, None)
        return json.dumps(d)

    def deserialize(self, path):
        d = None
        file_name = "{}/{}".format(path, self._file_name)
        with open(file_name, 'r') as f:
            s = f.read()
            d = self._deserialize(s)

    def _deserialize(self, s):
        self._epoch_no = d["epoch_no"]
        self._key = d["key"]
        self._hash_key = d["hash_key"]
        self._checkpoint_path = d["checkpoint_path"]
        self._checkpoint_no = d["checkpoint_no"]
        if "restored" in d:
            self._restored = d["restored"]

    def _to_dict(self):
        return {
            "epoch_no": self._epoch_no,
            "key": self._key,
            "hash_key": self._hash_key,
            "checkpoint_path": self._checkpoint_path,
            "restored": self._restored,
            "exe_name": self._exe._auto_checkpoint_name,
            "program_name": self._program._auto_checkpoint_name,
            "checkpoint_no": self._checkpoint_no
        }

    def __str__(self):
        return self._serialize([])


g_train_epoch_range = None
g_checker = AutoCheckpointChecker()


class TrainEpochRange(SerializableBase):
    def __init__(self, max_epoch_num, name):
        self._max_epoch_num = max_epoch_num
        self._epoch_no = -1  # current
        self._name = name
        self._checkpoint_path = _get_checkpoint_path(name)
        self._restored = False

        self._checker = g_checker
        if not self._checker.valid():
            return

        config = {
            "fs.default.name": self._checker.hdfs_name,
            "hadoop.job.ugi": self._checker.hdfs_ugi
        }

        self._hdfs = HDFSClient(self._hadoop_home, config)

        self._cper = Checkpointer(self._hdfs)
        self._exe_status = {}
        self._file_name = "range_train_status"

        self._cper.load_checkpoint(self._checkpoint_path + "/range", [self])

    def _to_dict(self):
        d = {
            "max_epoch_num": self._max_epoch_num,
            "epoch_no": self._epoch_no,
            "name": self._name,
            "checkpoint_path": self._checkpoint_path,
            "restored": self._restored
        }

    def __str__(self):
        return self._serialize(pop_keys=[])

    def serialize(self, path):
        file_name = "{}/{}".format(path, self._file_name)
        with open(file_name, 'w') as f:
            s = self._serilize()
            f.write(s)

    def _serialize(self, pop_keys=["restored"]):
        # self
        d = self._to_dict()
        for k in pod_keys:
            d.pop(k, None)

        # registerd exes
        d["exe_status"] = {}
        e = ["exe_status"]
        for t in self._exe_status:
            e[t._hash_key] = t._serialize()
        return json.dumps(d)

    def deserialize(self, path):
        d = None
        file_name = "{}/{}".format(path, self._file_name)
        with open(file_name, 'w') as f:
            d = json.load(f)

        # self
        self._max_epoch_num = d["max_epoch_num"]
        self._epoch_no = d["epoch_no"]
        self._name = d["name"]
        self._checkpoint_path = d["checkpoint_path"]
        if "restored" in d:
            self._restored = d["restored"]

        # exes status
        e = d["exe_status"]
        for k, v in e:
            t = ExeTrainStatus()
            t._deserialize(v)
            self._exe_stats[k] = t

    def next(self):
        if self._max_epoch_num < 0:
            self._max_epoch_num = sys.maxint - 1

        assert self._epoch_no >= -1, "self._epoch_no:{} must >=-1".format(
            self._epoch_no)

        start = self._epoch_no + 1
        for i in range(start, self._max_epoch_num + 1):
            yield i
            self._epoch_no = i

            if self._checker.trainer_id == 0:
                self._save_checkpoint()

    def get(self):
        assert self._epoch_no >= 0, "invalid epoch no:{}".format(self._epoch_no)

    def save_checkpoint(self):
        """
        status => /jobid/xxx_range_xx/range/
        model =>                      /exe/
        """
        for t in self._exe_status:
            m = PaddleModel(exe, program)
            path, checkpoint_no = self._cper.save_checkpoint(
                self._checkpoint_path + "/exe")
            # index info
            t._checkpoint_path = path
            t._checkpoint_no = checkpoint_no
            t.epoch_no = self.get()

            logging.info("save exe checkpoint:{}".format(t))

        self._cper.save_checkpoint(self._checkpoint_path + "/range", [self])
        logging.info("save train_epoch_range checkpoint:{}".format(self))

    @static
    def _generate_range_name():
        return unique_name.generate(g_checker._job_id + "_range_")


def _get_train_epoch_range():
    return g_train_epoch_range


def _can_auto_checkpoint():
    return g_checker.valid() and g_train_epoch_range is not None


def _generate_program_name():
    return unique_name.generate(g_checker._job_id + "_program_")


def _generate_executor_name():
    return unique_name.generate(g_checker._job_id + "_executor_")


def _get_checkpoint_path(name):
    return "%s/%s/%s".format(g_checker._hdfs_checkpoint_path, g_checker._job_id,
                             name)


def _get_running_key(exe_name, program_name):
    return "%s_%s".format(exe_name, program_name)


def train_epoch_range(max_epoch_num):
    g_train_epoch_range = TrainEpochRange(
        max_epoch_num, TrainEpochRange._generate_range_name())
    for i in t.next():
        yield i
    g_train_epoch_range = None


def _get_hash(key):
    k = key
    if sys.version_info[0] >= 3:
        k = key.encode('utf-8')

    return hashlib.md5(k).hexdigest()


def _initial_names(exe, program):
    if program._auto_checkpoint_name is None:
        program._auto_checkpoint_name = _generate_program_name()

    if exe._auto_checkpoint_name is None:
        exe._auto_checkpoint_name = _generate_executor_name()


def _auto_checkpoint(exe, program):
    if not _can_auto_checkpoint():
        return

    _initial_names(exe, program)

    exe_status = g_train_epoch_range._exe_status
    key = _get_running_key(exe._auto_checkpoint_name,
                           program._auto_checkpoint_name)

    t = None
    if key in exe_status:
        t = exe_status[key]
        if not t._restored:
            a = Checkpointer(g_train_epoch_range.hdfs)
            m = PaddleModel(exe, program)
            a.load_checkpoint(t._checkpoint_path, [m])
            logging.info("load_checkpoint exe checkpoint from {}".format(t))
            t._restored = True
    else:
        h = _get_hash(key)

        t = TrainStatus()
        t._epoch_no = g_train_epoch_range.get()
        t._hash_key = h
        t._key = key
        t._restored = True
        t._exe = exe
        t._program = program

        # register this <exe,program,io>
        exe_status[key] = t

        logging.info("not found checkpoint, so train from scrach")

    assert current_thread(
    ).name == "MainThread", "auto checkpoint must run under main thread"
