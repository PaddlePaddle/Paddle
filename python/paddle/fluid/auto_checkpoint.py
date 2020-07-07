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
from paddle.fluid import core
from paddle.fluid import framework
import logging
from contextlib import contextmanager
from . import unique_name
import paddle.fluid as fluid
from fluid.incubate.fleet.utils.hdfs import HDFSClient
import hashlib
import json


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


class ExeTrainStatus(object):
    def __init__(self):
        self._epoch_no = -1
        self._hash_key = None
        self._key = None
        self._checkpoint_path = None
        self._restored = False
        self._exe = None
        self._program = None

    def next(self):
        return self._epoch_no + 1

    def __eq__(self, t):
        return self._epoch_no == t._epoch_no and \
            self._hash_key == t._hash_key and \
            self._key == t._key

    def __ne__(self, t):
        return not self == t

    def serialize(self, pop_keys=["restored"]):
        d = self._to_dict()
        for k in pop_keys:
            d.pop(k, None)
        return json.dumps(d)

    def deserialize(self, s):
        d = json.loads(s)
        self._epoch_no = d["epoch_no"]
        self._key = d["key"]
        self._hash_key = d["hash_key"]
        self._checkpoint_path = d["checkpoint_path"]
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
        }

    def __str__(self):
        return self.serialize([])


g_train_epoch_range = None
g_checker = AutoCheckpointChecker()


class TrainEpochRange(object):
    def __init__(self, max_epoch_num, name):
        self._max_epoch_num = max_epoch_num
        self._last_epoch_no = -1  # restored
        self._epoch_no = self._last_epoch_no  # current
        self._name = name
        self._checkpoint_path = _get_checkpoint_path(name)
        self._restored = False

        self._checker = g_checker
        if not self._checker.valid():
            return

        config = {
            "fs.default.name": self._checker._hdfs_name,
            "hadoop.job.ugi": self._checker._hdfs_ugi
        }

        self._hdfs = HDFSClient(self._hadoop_home, config)
        self._cp_path = "{}/{}".format(self._checkpoint_path)

        self._cper = Checkpointer(self._hdfs)
        #self._status = None
        self._exe_status = {}

    def _to_dict(self):
        d = {
            "max_epoch_num": self._max_epoch_num,
            "last_epcho_no": self._last_epoch_no,
            "name": self._name,
            "checkpoint_path": self._checkpoint_path,
            "self._restored": self._restored
        }

    def serialize(self, pop_keys=["restored"]):
        # self
        d = self._to_dict()
        for k in pod_keys:
            d.pop(k, None)

        # registerd exe
        d["exe_status"] = {}
        e = ["exe_status"]
        for t in self._exe_status:
            e[t._hash_key] = t.serialize()
        return json.dumps(d)

    def deserialize(self, s):
        d = json.loads(s)

        # self
        self._last_epoch_no = d["epoch_no"]
        self._name = d["key"]
        self._checkpoint_path = d["checkpoint_path"]
        if "restored" in d:
            self._restored = d["restored"]
        self._epoch_no = self._last_epoch_no

        # exe status
        e = d["exe_status"]
        for k, v in e:
            t = json.loads(v)
            self._exe_stats[k] = t

    def next(self):
        if self._max_epoch_num < 0:
            self._max_epoch_num = sys.maxint - 1

        assert self._last_epoch_no >= -1, "self._epoch_no:{} must >=-1".format(
            self._last_epoch_no)

        for i in range(self._last_epoch_no + 1, self._max_epoch_num + 1):
            yield i
            self._epoch_no = i
            self._save_checkpoint()

    def get(self):
        assert self._current_epoch_no >= 0, "invalid epoch no:{}".format(
            self._current_epoch_no)

    def _save_checkpoint(self):
        for t in self._exe_status:
            path = _save_exe_checkpoint(t._exe, t._program, t)
            t._checkpoint_path = path
            t.epoch_no = self.get()
            logging.info("save exe checkpoint:{}".format(t))
        self._save_range_checkpoint()
        logging.info("save train_epoch_range checkpoint:{}".format(
            self._status))

    def _load_checkpoint(self):
        self._status, self._exe_status = self._load_range_checkpoint()
        if self._status is None:
            t = TrainStatus()
            t._epoch_no = -1
            t._hash_key = name
            t._key = name
            self._status = t

        self._epoch_no = self._status["epoch_no"]
        self._current_epoch_no = None

    @static
    def _generate_range_name():
        return unique_name.generate(g_checker._job_id + "_range_")


def _get_train_epoch_range_obj():
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


def _get_running_key(exe_name, program_name, io_key):
    return "%s_%s_%s".format(exe_name, program_name, io_key)


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


def _initial_names(exe, program, io_key):
    if program._auto_checkpoint_name is None:
        program._auto_checkpoint_name = _generate_program_name()

    if exe._auto_checkpoint_name is None:
        exe._auto_checkpoint_name = _generate_executor_name()


def _load_exe_checkpoint(exe, program, path):
    pass


def _save_exe_checkpoint(exe, program, status):
    pass


def _auto_checkpoint(exe, program, io_key):
    if not _can_auto_checkpoint():
        return

    _initial_names(exe, program)

    exe_status = g_train_epoch_range._exe_status
    key = _get_running_key(exe._auto_checkpoint_name,
                           program._auto_checkpoint_name, io_key)

    t = None
    if key in exe_status:
        t = exe_status[key]
        if not t._restored:
            _try_to_load_exe_checkpoint(exe, program, t._checkpoint_path)
            logging.info("load_checkpoint from path:{} content:{}".format(
                t._checkpoint_path, t))
            t._restored = True
    else:
        h = _get_hash(key)

        t = TrainStatus()
        t._epoch_no = g_train_epoch_range.get()
        t._hash_key = h
        t._key = k
        t._restored = True
        t._exe = exe
        t._program = program

        exe_status[key] = t

        logging.info("not found checkpoint, so train from scrach")
