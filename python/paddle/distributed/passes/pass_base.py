# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import six
import sys
from abc import ABC, abstractmethod
from paddle.fluid.framework import program_guard


class PassContext:
    def __init__(self):
        self._applied_passes = []

    def add_pass(self, pass_obj):
        self._applied_passes.append(pass_obj)

    def pop_pass(self):
        del self._applied_passes[-1]


class PassBase(ABC):
    def __init__(self):
        self._attrs = {}

    def set_attr(self, key, value):
        self._attrs[key] = value
        return self

    def get_attr(self, key, default=None):
        return self._attrs.get(key, default)

    def apply(self, main_programs, startup_programs, context):
        if not self.check_before_apply(context):
            return

        assert isinstance(main_programs, list)
        assert isinstance(startup_programs, list)
        assert len(main_programs) == len(startup_programs)

        context.add_pass(self)
        try:
            self.apply_impl(main_programs, startup_programs, context)
            self.check_after_apply(context)
        except:
            context.pop_pass()
            six.reraise(*sys.exc_info())

    def apply_impl(self, main_programs, startup_programs, context):
        for main_program, startup_program in zip(main_programs,
                                                 startup_programs):
            self.apply_single_impl(main_program, startup_program, context)

    @abstractmethod
    def apply_single_impl(self, main_program, startup_program, context):
        pass

    def check_before_apply(self, context):
        pass

    def check_after_apply(self, context):
        pass


# Like AMP, Recompute, etc.
class CalculationOptPass(PassBase):
    pass


# Like FuseAllReduce, FuseGradientMerge, etc. 
class CommunicationOptPass(PassBase):
    pass


class PassManager:
    def __init__(self, passes, context):
        self._passes = _check_pass_conflict(passes, context)
        self._context = context

    def _check_pass_conflict(self, context):
        pass

    def apply(self, main_programs, startup_programs):
        for p in self._passes:
            p.apply(main_programs, startup_programs, self._context)

    @property
    def context(self):
        return self._context
