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


class DACP(object):
    def __init__(self):
        pass

    def begin_step(self):
        pass

    def end_step(self):
        pass


g_dacp = None


def _get_dacp():
    if g_dacp is None:
        g_dacp = DACP()

    return g_dacp


def _begin(name):
    dacp = _get_dacp()


def _end(name):
    dacp = g_dacp
    assert dacp is not None, "Internal fatal error: g_dacp must not None"

    dacp.end_step(name)
