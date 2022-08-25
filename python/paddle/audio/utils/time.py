# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
import math
import time

__all__ = [
    'Timer',
    'seconds_to_hms',
]


class Timer(object):
    '''Calculate runing speed and estimated time of arrival(ETA)'''

    def __init__(self, total_step: int):
        self.total_step = total_step
        self.last_start_step = 0
        self.current_step = 0
        self._is_running = True

    def start(self):
        self.last_time = time.time()
        self.start_time = time.time()

    def stop(self):
        self._is_running = False
        self.end_time = time.time()

    def count(self) -> int:
        if not self.current_step >= self.total_step:
            self.current_step += 1
        return self.current_step

    @property
    def timing(self) -> float:
        run_steps = self.current_step - self.last_start_step
        self.last_start_step = self.current_step
        time_used = time.time() - self.last_time
        self.last_time = time.time()
        return run_steps / time_used

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def eta(self) -> str:
        if not self.is_running:
            return '00:00:00'
        scale = self.total_step / self.current_step
        remaining_time = (time.time() - self.start_time) * scale
        return seconds_to_hms(remaining_time)


def seconds_to_hms(seconds: int) -> str:
    '''Convert the number of seconds to hh:mm:ss'''
    h = math.floor(seconds / 3600)
    m = math.floor((seconds - h * 3600) / 60)
    s = int(seconds - h * 3600 - m * 60)
    hms_str = '{:0>2}:{:0>2}:{:0>2}'.format(h, m, s)
    return hms_str
