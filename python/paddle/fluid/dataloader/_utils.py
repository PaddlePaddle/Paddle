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

from __future__ import print_function

import os

__all__ = ["ParentWatchDog"]


class ParentWatchDog(object):
    def __init__(self):
        self._parent_pid = os.getppid()
        self._parent_alive = True

    def is_alive(self):
        if self._parent_alive:
            self._parent_alive = os.getppid() == self._parent_pid
        return self._parent_alive
