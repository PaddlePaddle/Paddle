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

from . import auto_checkpoint as acp

g_ranges = {}

CONST_GENERATOR_BEGIN = "begin"
CONST_GENERATOR_END = "end"


class TrainEpochRangeWrapper(object):
    def __init__(self, name):
        self._running_status = CONST_GENERATOR_BEGIN
        self._epoch_no = -1
        self._checkpoint_epoch_no = None

        self._train_epoch_range = acp.TrainEpochRange(
            -1,
            name,
            checkpoint_inter=acp._get_checker().save_checkpoint_inter,
            load_last=-2)

        if self._train_epoch_range.restored_from == acp.CONST_CHECKPOINT:
            self._checkpoint_epoch_no = self._train_epoch_range._checkpoint_epoch_no

    def save_checkpoint(self):
        logger.info(self)
        if self.beyond_restored():
            self._train_epoch_range.save_checkpoint()

    def __str__(self):
        return "epoch_no:{} checkpoint_epoch_no:{} running_status:{}".format(
            self._epoch_no, self._checkpoint_epoch_no, self._running_status)

    def increment_epoch_no(self):
        self._epoch_no += 1
        self._train_epoch_range._epoch_no = self._epoch_no

    def check(self):
        assert len(
            self._train_epoch_range._exe_status
        ) <= 1, "data loader checkpoint must contain one exe when running"

    def beyond_restored(self):
        if self.is_restored():
            return self._epoch_no > self._checkpoint_epoch_no

        return True

    def is_restored(self):
        return self._checkpoint_epoch_no is not None


logger = acp._get_logger(20)


def _check_env():
    checker = acp._get_checker()
    if not checker.valid():
        return False

    if acp.g_acp_type == acp.CONST_ACP_TYPE:
        return False

    return True


def _current(name):
    init = False
    if name not in g_ranges:
        g_ranges[name] = TrainEpochRangeWrapper(name)
        init = True
    t = g_ranges[name]
    t.check()
    acp.g_train_epoch_range = t._train_epoch_range
    acp.g_acp_type = acp.CONST_DACP_TYPE

    return t, init


def _begin(name):
    if not _check_env():
        return False

    t, init = _current(name)
    if init:
        logger.info("acp_type:{}".format(acp.g_acp_type))
        if t.is_restored:
            logger.info("begin dataloader epoch_no:{} checkpoint_epoch_no:{}".
                        format(t._epoch_no + 1, t._checkpoint_epoch_no))
        else:
            logger.info("begin dataloader epoch_no:{}".format(t._epoch_no + 1))
            return True

    if not t.beyond_restored():
        raise StopIteration

    return True


def _end(name):
    if not _check_env():
        return False

    try:
        # check
        assert name in g_ranges, \
            "internal error: g_ranges must contain the name:{}, now:{}".format(name, g_ranges.keys())
        t = g_ranges[name]
        t._runing_status = CONST_GENERATOR_END
        assert t._train_epoch_range == acp.g_train_epoch_range, "interal error, current running range must equal"

        t.increment_epoch_no()
        t.save_checkpoint()

        if not t.is_restored():
            logger.info("end dataloader epoch_no:{}".format(t._epoch_no))
        else:
            logger.info("end generator epoch_no:{} checkpoint_epoch_no:{}".
                        format(t._epoch_no, t._checkpoint_epoch_no))
    finally:
        # important
        acp.g_train_epoch_range = None

    return True
