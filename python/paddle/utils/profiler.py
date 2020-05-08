# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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

import sys
import warnings

from ..fluid import core
from ..fluid.profiler import *

#__all__ = ['ProfilerOptions', 'Profiler']


class ProfilerOptions(object):
    def __init__(self, options=None):
        if options is not None:
            self.options = options  # use deep copy
        else:
            self.options = {
                'state': 'All',
                'sorted_key': 'default',
                'tracer_level': 'Default',
                'batch_range': [0, sys.maxsize],
                'profile_path': 'none',
                'timeline_path': 'none',
                'op_summary_path': 'none'
            }

    def with_state(self, state):
        self.options['state'] = state
        return self

    def __getitem__(self, name):
        if self.options.get(name, None) is None:
            raise ValueError(
                "ProfilerOptions does not have an option named %s." % name)
        else:
            if isinstance(self.options[name],
                          str) and self.options[name] == 'none':
                return None
            else:
                return self.options[name]


class Profiler(object):
    def __init__(self, options=None):
        if options is not None:
            self.profiler_options = options
        else:
            self.profiler_options = ProfilerOptions()
        self.batch_id = 0
        self.enabled = False

    def __enter__(self):
        if self.profiler_options['batch_range'][0] == 0:
            self.start()

    def __exit__(self, exception_type, exception_value, traceback):
        self.stop()

    def start(self):
        if not self.enabled:
            try:
                start_profiler(
                    state=self.profiler_options['state'],
                    tracer_option=self.profiler_options['tracer_level'])
                self.enabled = True
            except Exception as e:
                warnings.warn(
                    "Profiler is not enabled becuase following exception:\n{}".
                    format(e))

    def stop(self):
        if self.enabled:
            try:
                stop_profiler(
                    sorted_key=self.profiler_options['sorted_key'],
                    profile_path=self.profiler_options['profile_path'])
                self.enabled = False
            except Exception as e:
                warnings.warn(
                    "Profiler is not disabled becuase following exception:\n{}".
                    format(e))

    def reset(self):
        if self.enabled:
            reset_profiler()

    def add_step(self, change_profiler_status=True):
        self.batch_id = self.batch_id + 1
        if change_profiler_status:
            if self.batch_id == self.profiler_options['batch_range'][0]:
                if self.enabled:
                    self.reset()
                else:
                    self.start()

            if self.batch_id == self.profiler_options['batch_range'][1]:
                self.stop()
