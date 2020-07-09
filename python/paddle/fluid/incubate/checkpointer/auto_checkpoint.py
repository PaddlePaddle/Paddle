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

import sys
import logging
import hashlib
import json
import os
import six
import time
from threading import Thread, current_thread
from contextlib import contextmanager

from paddle.fluid import unique_name, compiler
from paddle.fluid.incubate.fleet.utils.hdfs import HDFSClient
from .checkpointer import SerializableBase, Checkpointer, PaddleModel

g_train_epoch_range = None
g_checker = None

logger = None


def _get_logger(log_level, name="auto_checkpoint"):
    global logger
    if logger != None:
        return logger

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False

    log_handler = logging.StreamHandler()
    log_format = logging.Formatter(
        '%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s')
    log_handler.setFormatter(log_format)
    logger.addHandler(log_handler)

    return logger


def _thread_checker():
    assert current_thread().name == "MainThread", \
        "auto checkpoint must run under main thread"


class AutoCheckpointChecker(object):
    def __init__(self):
        self._run_env = None
        self._platform = None
        self._job_id = None
        self._hdfs_home = None
        self._hdfs_name = None
        self._hdfs_ugi = None
        self._hdfs_checkpoint_path = None
        self._trainer_id = None
        self._ce_test = None

        self._run_env = os.getenv("PADDLE_RUNNING_ENV")
        if self.run_env != "PADDLE_EDL_AUTO_CHECKPOINT":
            return

        try:
            self._platform = os.environ["PADDLE_RUNNING_PLATFORM"]
            self._job_id = os.environ["PADDLE_JOB_ID"]
            self._hdfs_home = os.environ["PADDLE_EDL_HDFS_HOME"]
            self._hdfs_name = os.environ["PADDLE_EDL_HDFS_NAME"]
            self._hdfs_ugi = os.environ["PADDLE_EDL_HDFS_UGI"]
            self._hdfs_checkpoint_path = os.environ[
                "PADDLE_EDL_HDFS_CHECKPOINT_PATH"]
            self._trainer_id = int(os.environ["PADDLE_EDL_TRAINER_ID"])
            self._ce_test = bool(os.environ["PADDLE_EDL_ONLY_FOR_CE_TEST"])

            if not self._ce_test:
                assert len(self._hdfs_home) > 3 and \
                    len(self._hdfs_name) > 6 and \
                    len(self._hdfs_ugi) > 3 and \
                    len(self._hdfs_checkpoint_path) > 0, "hdfs environ must set"
            else:
                assert len(self._hdfs_home) > 3 and \
                    len(self._hdfs_checkpoint_path) > 0, "hdfs environ must set"

        except Exception as e:
            logger.fatal("exception:", e)

    def get_range_checkpoint_path(self, name):
        return "{}/{}/range/{}".format(self.hdfs_checkpoint_path, self.job_id,
                                       name)

    def get_exe_checkpoint_path(self, name):
        return "{}/{}/exe/{}".format(self.hdfs_checkpoint_path, self.job_id,
                                     name)

    def valid(self):
        return  self._run_env is not None and \
            self._platform is not None and \
            self._job_id is not None and \
            self._hdfs_home is not None and \
            self._hdfs_name is not None and \
            self._hdfs_ugi is not None and \
            self._hdfs_checkpoint_path is not None and \
            self._trainer_id is not None

    def __str__(self):
        return "run_env:{} platform:{} job_id:{} \
            hdfs_home:{} hdfs_name:{} hdfs_ugi:{} \
            hdfs_checkpoint_path:{} trainer_id:{} ce_test".format(
            self._run_env, self._platform, self._hdfs_home, self._hdfs_name,
            self._hdfs_ugi, self._hdfs_checkpoint_path, self._trainer_id,
            self._ce_test)

    @property
    def trainer_id(self):
        return self._trainer_id

    @property
    def run_env(self):
        return self._run_env

    @property
    def platform(self):
        return self._platform

    @property
    def job_id(self):
        return self._job_id

    @property
    def hdfs_home(self):
        return self._hdfs_home

    @property
    def hdfs_name(self):
        return self._hdfs_name

    @property
    def ce_test(self):
        return self._ce_test

    @property
    def hdfs_ugi(self):
        return self._hdfs_ugi

    @property
    def hdfs_checkpoint_path(self):
        return self._hdfs_checkpoint_path

    def generate_range_name(self):
        assert self.valid()
        return unique_name.generate("_range_")

    def generate_program_name(self):
        assert self.valid()
        return unique_name.generate("_program_")

    def generate_executor_name(self):
        assert self.valid()
        return unique_name.generate("_executor_")


class ExeTrainStatus(SerializableBase):
    def __init__(self):
        self._epoch_no = -1  # start epoch_no
        self._hash_key = None
        self._key = None
        self._checkpoint_path = None
        self._checkpoint_no = None
        self._restored = False
        self._exe = None
        self._program = None
        self._exe_name = None
        self._program_name = None

        self._file_name = "exe_train_status"

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
            self._deserialize(s)

    def _deserialize(self, s):
        d = json.loads(s)
        self._epoch_no = d["epoch_no"]
        self._key = d["key"]
        self._hash_key = d["hash_key"]
        self._checkpoint_path = d["checkpoint_path"]
        self._checkpoint_no = d["checkpoint_no"]
        self._exe_name = d["exe_name"]
        self._program_name = d["program_name"]
        self._restored = False

    def _to_dict(self):
        return {
            "epoch_no": self._epoch_no,
            "key": self._key,
            "hash_key": self._hash_key,
            "checkpoint_path": self._checkpoint_path,
            "restored": self._restored,
            "exe_name": self._exe_name,
            "program_name": self._program_name,
            "checkpoint_no": self._checkpoint_no
        }

    def __str__(self):
        return self._serialize([])


class TrainEpochRange(SerializableBase):
    def __init__(self, max_epoch_num, name, save_checkpoint_inter=300):
        self._max_epoch_num = max_epoch_num
        self._epoch_no = -1  # current epoch_no
        self._last_checkpoint_time = None
        self._name = name
        self._restored = False
        self._exe_status = {}
        self._save_checkpoint_inter = save_checkpoint_inter

        self._checker = g_checker
        if not self._checker.valid():
            return

        self._checkpoint_path = self._checker.get_range_checkpoint_path(name)

        config = {
            "fs.default.name": self._checker.hdfs_name,
            "hadoop.job.ugi": self._checker.hdfs_ugi
        }

        if self._checker.ce_test:
            config = None

        self._hdfs = HDFSClient(self._checker.hdfs_home, config)

        self._cper = Checkpointer(self._hdfs)
        self._file_name = "range_train_status"

        _thread_checker()

        if self._cper._get_last_checkpoint_no(self._checkpoint_path) > 0:
            self._cper.load_checkpoint(self._checkpoint_path, [self],
                                       self._checker.trainer_id)
            logger.info("load tain_epoch_range checkpoint:{}".format(self))

    def _to_dict(self):
        d = {
            "max_epoch_num": self._max_epoch_num,
            "epoch_no": self._epoch_no,
            "name": self._name,
            "checkpoint_path": self._checkpoint_path,
            "restored": self._restored
        }
        return d

    def __str__(self):
        return self._serialize(pop_keys=[])

    @property
    def name(self):
        return self._name

    def serialize(self, path):
        file_name = "{}/{}".format(path, self._file_name)
        with open(file_name, 'w') as f:
            s = self._serialize()
            f.write(s)

    def _serialize(self, pop_keys=["restored"]):
        # self
        d = self._to_dict()
        for k in pop_keys:
            d.pop(k, None)

        # registerd exes
        d["exe_status"] = {}
        e = d["exe_status"]
        for k, t in six.iteritems(self._exe_status):
            e[t._key] = t._serialize()
        return json.dumps(d)

    @property
    def is_restored(self):
        return self._restored

    def deserialize(self, path):
        d = None
        file_name = "{}/{}".format(path, self._file_name)
        with open(file_name, 'r') as f:
            d = json.load(f)

        # self
        self._max_epoch_num = d["max_epoch_num"]
        self._epoch_no = d["epoch_no"]
        self._name = d["name"]
        self._checkpoint_path = d["checkpoint_path"]
        self._restored = True

        # exes status
        e = d["exe_status"]
        for k, v in six.iteritems(e):
            t = ExeTrainStatus()
            t._deserialize(v)
            self._exe_status[k] = t

    def next(self):
        _thread_checker()

        if self._max_epoch_num < 0:
            self._max_epoch_num = sys.maxint

        assert self._epoch_no >= -1, "self._epoch_no:{} must >=-1".format(
            self._epoch_no)

        self._last_checkpoint_time = time.time()
        start = self._epoch_no + 1
        logger.info("started epoch_no:{} max_epoch_num:{}".format(
            self._epoch_no, self._max_epoch_num))
        for i in range(start, self._max_epoch_num):
            self._epoch_no = i
            yield i

            # not save last one because exe and program can't be restored.
            if self._checker.trainer_id == 0 and i != self._max_epoch_num - 1:
                if time.time() - self._last_checkpoint_time >= self._save_checkpoint_inter or \
                        i >= self._max_epoch_num:
                    self.save_checkpoint()
                self._last_checkpoint_time = time.time()

    def get(self):
        return self._epoch_no

    def save_checkpoint(self):
        """
        status => /jobid/xxx_range_xx/range/
        model =>                       /exe/
        """
        if not self._checker.valid():
            return

        e = self._exe_status
        l = e.values()
        for t in l:
            m = PaddleModel(t._exe, t._program)
            p = self._checker.get_exe_checkpoint_path(t._hash_key)
            path, checkpoint_no = self._cper.save_checkpoint(
                p, [m], self._checker.trainer_id)
            # index info
            t._checkpoint_path = path
            t._checkpoint_no = checkpoint_no
            t.epoch_no = self.get()

            e[t._key] = t

            logger.info("save executor checkpoint:{}".format(t))

        if len(self._exe_status) > 0:
            self._cper.save_checkpoint(self._checkpoint_path, [self])
            logger.info("save train_epoch_range checkpoint:{}".format(self))


def _get_train_epoch_range():
    return g_train_epoch_range


def _can_auto_checkpoint(program):
    print("program auto checkpoint:", program._auto_checkpoint)
    if isinstance(program, compiler.CompiledProgram):
        if not program._auto_checkpoint or program._program._is_distributed:
            return False
    else:
        if not program._auto_checkpoint or program._is_distributed:
            return False

    _get_checker()
    return g_checker.valid() and g_train_epoch_range is not None


def _get_running_key(exe_name, program_name):
    return "{}_{}".format(exe_name, program_name)


def _get_checker():
    _get_logger(20)
    global g_checker
    if g_checker is None:
        g_checker = AutoCheckpointChecker()

    return g_checker


def train_epoch_range(max_epoch_num, save_checkpoint_inter=300):
    if not _get_checker().valid():
        if max_epoch_num < 0:
            max_epoch_num = sys.maxint
        for i in range(0, max_epoch_num):
            yield i
        logger.warning("auto checkpoint will take effect on PaddleCloud")
        return

    for i in _run_only_for_inter(g_checker.generate_range_name(), max_epoch_num,
                                 save_checkpoint_inter):
        yield i


def _run_only_for_inter(range_name, max_epoch_num, save_checkpoint_inter):
    global g_train_epoch_range
    g_train_epoch_range = TrainEpochRange(
        max_epoch_num, range_name, save_checkpoint_inter=save_checkpoint_inter)

    for i in g_train_epoch_range.next():
        yield i

    g_train_epoch_range = None


def _get_hash(key):
    k = key
    if sys.version_info[0] >= 3:
        k = key.encode('utf-8')

    return hashlib.md5(k).hexdigest()


def _initial_names(exe, program):
    if program._auto_checkpoint_name is None:
        program._auto_checkpoint_name = g_checker.generate_program_name()

    if exe._auto_checkpoint_name is None:
        exe._auto_checkpoint_name = g_checker.generate_executor_name()


def _auto_checkpoint(exe, program):
    if not _can_auto_checkpoint(program):
        return

    _initial_names(exe, program)

    exe_status = g_train_epoch_range._exe_status
    key = _get_running_key(exe._auto_checkpoint_name,
                           program._auto_checkpoint_name)

    if g_train_epoch_range.is_restored:
        assert key in exe_status, "when restored key:{} must be in train_epoch_range:{}".format(
            key, g_train_epoch_range)

    t = None
    if key in exe_status:
        t = exe_status[key]
        if not t._restored:
            logger.info("load executor checkpoint {}".format(t))
            a = Checkpointer(g_train_epoch_range._hdfs)
            m = PaddleModel(exe, program)
            a.load_checkpoint(
                g_checker.get_exe_checkpoint_path(key), [m],
                trainer_id=g_checker.trainer_id,
                checkpoint_no=t._checkpoint_no)
            t._restored = True
        t._exe = exe
        t._program = program
    else:
        #h = _get_hash(key)

        t = ExeTrainStatus()
        t._epoch_no = g_train_epoch_range.get()
        t._hash_key = key
        t._key = key
        t._restored = True
        t._exe = exe
        t._program = program
        t._exe_name = exe._auto_checkpoint_name
        t._program_name = program._auto_checkpoint_name

        # register this <exe,program,io>
        exe_status[key] = t

        logger.info("not found checkpoint, so train from epoch 0")

    _thread_checker()
