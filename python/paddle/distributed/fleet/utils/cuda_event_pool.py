# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import logging

import paddle

logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)


class EventPoolSingleThread:
    def __init__(self, init_size=10):
        self.pool = []
        self.using = {}
        self._inc(init_size)
        self._global_count = -1

    def _inc(self, size=1):
        for i in range(size):
            self.pool.append(paddle.device.cuda.Event(enable_timing=True))

    def _check_tag_valid(self, tag):
        if tag in self.using.keys():
            return False
        return True

    def _preprocess_tag(self, tag):
        self._global_count += 1
        return str(self._global_count) + "_" + tag

    def reset_global_count(self):
        self._global_count = -1

    def get(self, tag, preprocess_tag=False):
        # TODO(@gexiao): for debug usage, should be removed later
        if preprocess_tag:
            tag = self._preprocess_tag(tag)
        if not self._check_tag_valid(tag):
            logger.warning(f"tag {tag} is already used")
            return None
        if len(self.pool) == 0:
            self._inc()
        event = self.pool.pop()
        self.using[tag] = event
        return event

    def release(self, tag):
        self.pool.append(self.using.pop(tag))
