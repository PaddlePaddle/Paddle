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

<<<<<<< HEAD
=======
from __future__ import print_function

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import sys
import warnings

from ..fluid import core
from ..fluid.profiler import cuda_profiler  # noqa: F401
<<<<<<< HEAD
from ..fluid.profiler import profiler  # noqa: F401
from ..fluid.profiler import reset_profiler, start_profiler, stop_profiler

__all__ = [  # noqa
    'Profiler',
    'get_profiler',
    'ProfilerOptions',
    'cuda_profiler',
    'start_profiler',
    'profiler',
    'stop_profiler',
    'reset_profiler',
]


class ProfilerOptions:
=======
from ..fluid.profiler import start_profiler
from ..fluid.profiler import profiler  # noqa: F401
from ..fluid.profiler import stop_profiler
from ..fluid.profiler import reset_profiler

__all__ = [  #noqa
    'Profiler', 'get_profiler', 'ProfilerOptions', 'cuda_profiler',
    'start_profiler', 'profiler', 'stop_profiler', 'reset_profiler'
]


class ProfilerOptions(object):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self, options=None):
        self.options = {
            'state': 'All',
            'sorted_key': 'default',
            'tracer_level': 'Default',
            'batch_range': [0, sys.maxsize],
            'output_thread_detail': False,
            'profile_path': 'none',
            'timeline_path': 'none',
<<<<<<< HEAD
            'op_summary_path': 'none',
=======
            'op_summary_path': 'none'
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }
        if options is not None:
            for key in self.options.keys():
                if options.get(key, None) is not None:
                    self.options[key] = options[key]

    # function to set one specified option
    def with_state(self, state):
        self.options['state'] = state
        return self

    def __getitem__(self, name):
        if self.options.get(name, None) is None:
            raise ValueError(
<<<<<<< HEAD
                "ProfilerOptions does not have an option named %s." % name
            )
        else:
            if (
                isinstance(self.options[name], str)
                and self.options[name] == 'none'
            ):
=======
                "ProfilerOptions does not have an option named %s." % name)
        else:
            if isinstance(self.options[name],
                          str) and self.options[name] == 'none':
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                return None
            else:
                return self.options[name]


_current_profiler = None


<<<<<<< HEAD
class Profiler:
=======
class Profiler(object):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self, enabled=True, options=None):
        if options is not None:
            self.profiler_options = options
        else:
            self.profiler_options = ProfilerOptions()
        self.batch_id = 0
        self.enabled = enabled

    def __enter__(self):
        # record current profiler
        global _current_profiler
        self.previous_profiler = _current_profiler
        _current_profiler = self

        if self.enabled:
            if self.profiler_options['batch_range'][0] == 0:
                self.start()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        global _current_profiler
        _current_profiler = self.previous_profiler

        if self.enabled:
            self.stop()

    def start(self):
        if self.enabled:
            try:
                start_profiler(
                    state=self.profiler_options['state'],
<<<<<<< HEAD
                    tracer_option=self.profiler_options['tracer_level'],
                )
            except Exception as e:
                warnings.warn(
                    "Profiler is not enabled becuase following exception:\n{}".format(
                        e
                    )
                )
=======
                    tracer_option=self.profiler_options['tracer_level'])
            except Exception as e:
                warnings.warn(
                    "Profiler is not enabled becuase following exception:\n{}".
                    format(e))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def stop(self):
        if self.enabled:
            try:
                stop_profiler(
                    sorted_key=self.profiler_options['sorted_key'],
<<<<<<< HEAD
                    profile_path=self.profiler_options['profile_path'],
                )
            except Exception as e:
                warnings.warn(
                    "Profiler is not disabled becuase following exception:\n{}".format(
                        e
                    )
                )
=======
                    profile_path=self.profiler_options['profile_path'])
            except Exception as e:
                warnings.warn(
                    "Profiler is not disabled becuase following exception:\n{}".
                    format(e))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def reset(self):
        if self.enabled and core.is_profiler_enabled():
            reset_profiler()

    def record_step(self, change_profiler_status=True):
        if not self.enabled:
            return
        self.batch_id = self.batch_id + 1
        if change_profiler_status:
            if self.batch_id == self.profiler_options['batch_range'][0]:
                if core.is_profiler_enabled():
                    self.reset()
                else:
                    self.start()

            if self.batch_id == self.profiler_options['batch_range'][1]:
                self.stop()


def get_profiler():
    global _current_profiler
    if _current_profiler is None:
        _current_profiler = Profiler()
    return _current_profiler
