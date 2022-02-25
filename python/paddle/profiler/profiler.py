# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import socket
import datetime
from enum import Enum
from typing import Any, Callable, Iterable, Optional, Union
from warnings import warn

import paddle
from paddle.fluid.core import (_Profiler, _ProfilerResult, ProfilerOptions,
                               TracerEventType)
from .utils import RecordEvent, wrap_optimizers, wrap_functional
from .profiler_statistic import StatisticData, _build_table, SortedKeys


class ProfilerState(Enum):
    '''
  Profiler state that can be specified to control profiler action.
  '''
    CLOSED = 0
    READY = 1
    RECORD = 2
    RECORD_AND_RETURN = 3  # the last step of RECORD 


class ProfilerTarget(Enum):
    '''
  Target device for profiling.
  '''
    CPU = 0
    GPU = 1


def make_scheduler(*,
                   closed: int,
                   ready: int,
                   record: int,
                   repeat: int=0,
                   skip_first: int=0) -> Callable:
    '''
  Return a scheduler function, which scheduler the state according to the setting.
  The state transform confirms to:

  (CLOSED)  (CLOSED)    (CLOSED)  (READY)    (RECORD,last RETURN)      (CLOSED)
  START -> skip_first -> closed -> ready    ->    record       ->      END
                          |                        |
                          |                        | (if has_repeated < repeat)
                           - - - - - - - - - - - -
  Note that repeat <= 0 means the cycle will continue until the profiler exits.                
  '''

    def getScheduleState(step: int) -> ProfilerState:
        assert step >= 0
        if step < skip_first:  # within skip_first, just skip
            return ProfilerState.CLOSED
        step = step - skip_first
        period_steps = closed + ready + record
        has_repeated = step // period_steps
        if repeat > 0 and has_repeated >= repeat:  # the period has repeated repeat times, return CLOSED state
            return ProfilerState.CLOSED
        mod_step = step % period_steps
        if mod_step < closed:
            return ProfilerState.CLOSED
        elif mod_step >= closed and mod_step < closed + ready:
            return ProfilerState.READY
        else:
            if mod_step < period_steps - 1:
                return ProfilerState.RECORD
            else:
                return ProfilerState.RECORD_AND_RETURN
    assert closed >= 0 and ready >= 0 and record > 0 and \
             repeat >= 0 and skip_first >= 0, "Invalid profiler scheduler arguments"
    if ready == 0:
        warn("Profiler will record data after enabling profiler immediately, \
          some data collected at the beginning of profiling may be 'noisy' because of overhead."
             )
    return getScheduleState


def _default_state_scheduler(step: int):
    '''
  A default state scheduler, keep recording from the begining of the profiler until ending.
  '''
    return ProfilerState.RECORD


def export_chrome_tracing(dir_name: str,
                          worker_name: Optional[str]=None) -> Callable:
    '''
  Return a callable, used for outputing tracing data to chrome tracing format file.
  The output file will be saved in directory 'dir_name', and file name will be set as worker_name.
  if worker_name is not set, the default name is [hostname]_[pid].
  '''
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name, exist_ok=True)
        except Exception:
            raise RuntimeError(
                "Can not create directory '{}' for saving profiling results.".
                format(dir_name))

    def handle_fn(prof: Profiler):
        nonlocal worker_name
        if not worker_name:
            worker_name = "host_{}pid_{}".format(socket.gethostname(),
                                                 str(os.getpid()))
        now = datetime.datetime.now()
        filename = '{}_time_{}.paddle_trace.json'.format(
            worker_name, now.strftime('%Y_%m_%d_%H_%M_%S_%f'))
        prof.export(os.path.join(dir_name, filename), "json")

    return handle_fn


def export_protobuf(dir_name: str, worker_name: Optional[str]=None) -> Callable:
    '''
  Return a callable, used for outputing tracing data to protobuf file.
  The output file will be saved in directory 'dir_name', and file name will be set as worker_name.
  if worker_name is not set, the default name is [hostname]_[pid].
  '''
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name, exist_ok=True)
        except Exception:
            raise RuntimeError(
                "Can not create directory '{}' for saving profiling results.".
                format(dir_name))

    def handle_fn(prof: Profiler):
        nonlocal worker_name
        if not worker_name:
            worker_name = "host_{}pid_{}".format(socket.gethostname(),
                                                 str(os.getpid()))
        now = datetime.datetime.now()
        filename = '{}_time_{}.paddle_trace.pb'.format(
            worker_name, now.strftime('%Y_%m_%d_%H_%M_%S_%f'))
        prof.export(os.path.join(dir_name, filename), "pb")

    return handle_fn


def _get_supported_targets() -> Iterable[ProfilerTarget]:
    '''
  Get the current supported profiler target in the system.
  '''
    if paddle.device.is_compiled_with_cuda():
        return [ProfilerTarget.CPU, ProfilerTarget.GPU]
    return [ProfilerTarget.CPU]


class Profiler:
    '''
  Profiler context manager, user interface to manage profile process.
  Args:
    targets (iterable): list of tracing targets, currently supported values:
      ``paddle.profiler.ProfilerTarget.CPU``,
      ``paddle.profiler.ProfilerTarget.GPU``.
    scheduler (callable or tuple): If it is a callable object, it takes a step number as parameter and return the corresponding ``ProfilerState``. 
        If not provided, the default sheduler will keep tracing until the profiler exits. If it is a tuple, it has two values start_batch and end_batch,
        which means profiling range [start_batch, end_batch).
    on_trace_ready (callable): callable object, takes the Profiler object as parameter, which provides a way for users to do post-processing.
        This callable object will be called when ``sheduler`` returns ``ProfilerState.RECORD_AND_RETURN``.
  Examples:
    1. profiling range [2, 5)
    .. code-block:: python
    import paddle.profiler as profiler
    with profiler.Profiler(targets=[profiler.ProfilerTarget.CPU,
                                    profiler.ProfilerTarget.GPU],
                           scheduler = (2, 5),
                           on_trace_ready = profiler.export_chrome_tracing('./log')
                          ) as p:
        for iter in range(N):
          train()
          p.step()
    2. profiling range [2,4], [7, 9], [11,13]
    .. code-block:: python
    import paddle.profiler as profiler
    with profiler.Profiler(targets=[profiler.ProfilerTarget.CPU,
                                    profiler.ProfilerTarget.GPU],
                           scheduler = profiler.make_scheduler(closed=1, ready=1, record=3, repeat=3),
                           on_trace_ready = profiler.export_chrome_tracing('./log')
                          ) as p:
        for iter in range(N):
          train()
          p.step()
    3. Use profiler without context manager, and use default parameters
    .. code-block:: python
    import paddle.profiler as profiler
    p = profiler.Profiler()
    p.start()
    for iter in range(N):
        train()
        p.step()
    p.stop()
  '''

    def __init__(
            self,
            *,
            targets: Optional[Iterable[ProfilerTarget]]=None,
            scheduler: Union[Callable[[int], ProfilerState], tuple, None]=None,
            on_trace_ready: Optional[Callable[..., Any]]=None):
        supported_targets = _get_supported_targets()
        if targets:
            self.targets = set(targets)
            for target in targets:
                if target not in supported_targets:
                    self.targets.remove(target)
                    warn("Profiling {} is not supported in current context.".
                         format(target))
        else:
            self.targets = supported_targets
        profileoption = ProfilerOptions()
        profileoption.trace_level = 2
        if ProfilerTarget.CPU in self.targets:
            profileoption.trace_switch |= 1
        if ProfilerTarget.GPU in self.targets:
            profileoption.trace_switch |= (1 << 1)
        wrap_optimizers()
        wrap_functional()
        self.profiler = _Profiler.Create(profileoption)
        if callable(scheduler):
            self.scheduler = scheduler
        elif isinstance(scheduler, (tuple, list)):
            assert len(scheduler) == 2 and scheduler[1] > scheduler[0]
            start_batch, end_batch = scheduler
            start_batch = max(start_batch, 0)
            if start_batch >= 1:
                self.scheduler = make_scheduler(
                    closed=max(start_batch - 1, 0),
                    ready=1,
                    record=(end_batch - start_batch),
                    repeat=1)
            else:
                self.scheduler = make_scheduler(
                    closed=0,
                    ready=0,
                    record=(end_batch - start_batch),
                    repeat=1)
        else:
            self.scheduler = _default_state_scheduler

        if on_trace_ready == None:
            self.on_trace_ready = export_chrome_tracing('./profiler_log/')
        else:
            self.on_trace_ready = on_trace_ready
        self.step_num = 0
        self.previous_state = ProfilerState.CLOSED
        self.current_state = self.scheduler(self.step_num)
        self.record_event = None
        self.profiler_result = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        # CLOSED -> self.current_state
        if self.current_state == ProfilerState.READY:
            self.profiler.Prepare()
        elif self.current_state == ProfilerState.RECORD:
            self.profiler.Prepare()
            self.profiler.Start()
        elif self.current_state == ProfilerState.RECORD_AND_RETURN:
            self.profiler.Prepare()
            self.profiler.Start()
        self.record_event = RecordEvent(
            name="ProfileStep#{}".format(self.step_num),
            event_type=TracerEventType.ProfileStep)
        self.record_event.begin()

    def stop(self):
        # self.current_state -> CLOSED
        # In this situation, RECORD state is regarded as RECORD_AND_RETURN
        if self.record_event:
            self.record_event.end()
            self.record_event = None
        if self.current_state == ProfilerState.READY:
            warn(
                "Inproper Profiler state transform: READY->CLOSED, profiler will start and stop without saving data"
            )
            self.profiler.Start()
            self.profiler.Stop()
        if self.current_state == ProfilerState.RECORD or self.current_state == ProfilerState.RECORD_AND_RETURN:
            self.profiler_result = self.profiler.Stop()
            if self.on_trace_ready:
                self.on_trace_ready(self)

    def step(self):
        """
    Signals the profiler that the next profiling step has started.
    Get the new ProfilerState and trigger corresponding action.
    """
        if self.record_event:
            self.record_event.end()
            self.record_event = None
        self.previous_state = self.current_state
        self.step_num += 1
        self.current_state = self.scheduler(self.step_num)
        self._trigger_action()
        self.record_event = RecordEvent(
            name="ProfileStep#{}".format(self.step_num),
            event_type=TracerEventType.ProfileStep)
        self.record_event.begin()

    def _trigger_action(self):
        if self.previous_state == ProfilerState.CLOSED:
            if self.current_state == ProfilerState.READY:  # CLOSED -> READY
                self.profiler.Prepare()
            if self.current_state == ProfilerState.RECORD:  # CLOSED -> RECORD
                self.profiler.Prepare()
                self.profiler.Start()
            if self.current_state == ProfilerState.RECORD_AND_RETURN:  # CLOSED -> RECORD_AND_RETURN
                self.profiler.Prepare()
                self.profiler.Start()

        elif self.previous_state == ProfilerState.READY:
            if self.current_state == ProfilerState.CLOSED:  # READY -> CLOSED
                warn(
                    "Improper schedule: READY->CLOSED, profiler will start and stop without saving data"
                )
                self.profiler.Start()
                self.profiler.Stop()
            if self.current_state == ProfilerState.RECORD:  # READY -> RECORD
                self.profiler.Start()
            if self.current_state == ProfilerState.RECORD_AND_RETURN:  # READY -> RECORD_AND_RETURN
                self.profiler.Start()

        elif self.previous_state == ProfilerState.RECORD:
            if self.current_state == ProfilerState.CLOSED:  # RECORD -> CLOSED
                warn(
                    "Improper schedule: RECORD->CLOSED, profiler will not saving data"
                )
                self.profiler.Stop()

            if self.current_state == ProfilerState.READY:  # RECORD -> READY
                warn(
                    "Improper schedule: RECORD->READY, profiler will stop and re-prepare"
                )
                self.profiler.Stop()
                self.profiler.Prepare()
            if self.current_state == ProfilerState.RECORD_AND_RETURN:  # RECORD -> RECORD_AND_RETURN
                pass

        else:
            assert self.previous_state == ProfilerState.RECORD_AND_RETURN
            if self.current_state == ProfilerState.CLOSED:  # RECORD_AND_RETURN -> CLOSED
                self.profiler_result = self.profiler.Stop()
            if self.current_state == ProfilerState.READY:  # RECORD_AND_RETURN -> READY
                self.profiler_result = self.profiler.Stop()
                self.profiler.Prepare()
            if self.current_state == ProfilerState.RECORD:  # RECORD_AND_RETURN -> RECORD
                self.profiler_result = self.profiler.Stop()
                self.profiler.Prepare()
                self.profiler.Start()
            if self.current_state == ProfilerState.RECORD_AND_RETURN:  # RECORD_AND_RETURN -> RECORD_AND_RETURN
                self.profiler_result = self.profiler.Stop()
                self.profiler.Prepare()
                self.profiler.Start()
            if self.on_trace_ready:
                self.on_trace_ready(self)

    def export(self, path="", format="json"):
        '''
    Exports the tracing data in Chrome tracing data format.
    '''
        if self.profiler_result:
            self.profiler_result.save(path, format)

    def summary(self,
                sorted_by=SortedKeys.OpTotal,
                op_detail=True,
                thread_sep=False,
                time_unit='ms'):
        '''
        sorted_by: how to rank the op table items.
        detail: expand each operator detail information.
        thread_sep: print op table each thread.
        time_unit: can be chosen form ['s', 'ms', 'us', 'ns']
        '''
        if self.profiler_result:
            statistic_data = StatisticData(
                self.profiler_result.get_data(),
                self.profiler_result.get_extra_info())
            print(
                _build_table(
                    statistic_data,
                    sorted_by=sorted_by,
                    op_detail=op_detail,
                    thread_sep=thread_sep,
                    time_unit=time_unit))
