#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from . import core

from framework import Program
from executor import global_scope

__all__ = ['Communicator']


class Communicator(object):
    def __init__(
            self,
            program, ):
        # set all recv op to not_run mode
        assert isinstance(program, Program)
        for op in program.block(0).ops:
            if op.type == "recv":
                op._set_attr('do_not_run', True)
        self.communicator_ = core.Communicator(program.desc, global_scope())

    def start(self):
        self.communicator_.start()
