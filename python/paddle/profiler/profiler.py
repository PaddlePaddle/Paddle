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

from .utils import RecordEvent, wrap_optimizers
from .profiler_statistic import StatisticData, _build_table, SortedKeys
from .timer import benchmark


class ProfilerState(Enum):
    r"""
    ProfilerState is used to present the state of :ref:`Profiler <api_paddle_profiler_Profiler>` .

    The meaning of each ProfilerState is as following

    - **ProfilerState.CLOSED** : The profiler is closed, and no profiling data will be recorded.

    - **ProfilerState.READY** : The profiler is open, but the data will not be recorded. This state is used for reducing overhead influence when profiler starts.

    - **ProfilerState.RECORD** : The profiler is open, and the data will be recorded.

    - **ProfilerState.RECORD_AND_RETURN** : The profiler is open, and this state stands for the last batch of "RECORD" state in current profiling period. The collected data will be returned in this state.
    """
    CLOSED = 0
    READY = 1
    RECORD = 2
    RECORD_AND_RETURN = 3  # the last step of RECORD


class ProfilerTarget(Enum):
    r"""
    ProfilerTarget is used to specify target device for :ref:`profiling <api_paddle_profiler_Profiler>` . Only CPU and GPU are supported currently.

    The meaning of each ProfilerState is as following

    - **ProfilerTarget.CPU** : Profile events on CPU.

    - **ProfilerTarget.GPU** : Profile events on GPU.
    """
    CPU = 0
    GPU = 1


def make_scheduler(*,
                   closed: int,
                   ready: int,
                   record: int,
                   repeat: int=0,
                   skip_first: int=0) -> Callable:
    r"""
    Return a scheduler function, which scheduler the :ref:`state <api_paddle_profiler_ProfilerState>` according to the setting.
    The state transform confirms to:

    .. code-block:: text

        (CLOSED)  (CLOSED)    (CLOSED)  (READY)    (RECORD,last RETURN)      (CLOSED)
        START -> skip_first -> closed -> ready    ->    record       ->      END
                                |                        |
                                |                        | (if has_repeated < repeat)
                                - - - - - - - - - - - -
        Note that repeat <= 0 means the cycle will continue until the profiler exits.

    Args:
        closed(int): The number of steps in state ProfilerState.CLOSED.
        ready(int):  The number of steps in state ProfilerState.READY.
        record(int): The number of steps in state ProfilerState.RECORD, and the state in last step will be set as ProfilerState.RECORD_AND_RETURN.
        repeat(int, optional): The number of cycles to repeat above state transform. Default value is 0, which means it will repeat this cycle until profiler exits.
        skip_first(int, optional): The number of first steps to drop, not participate in the state transform, and at ProfilerState.CLOSED state. Default value is 0.

    Returns:
        A scheduler function, conforms to above state transform setting. The function will takes one parameter step_num, and returns corresponding ProfilerState.

    Examples:
        1. profiling range [2, 5]

        Assume batch 0: closed, batch 1: ready, batch [2, 5] record

            .. code-block:: python
                :name: code-example1

                import paddle.profiler as profiler
                profiler.make_scheduler(closed=1, ready=1, record=4, repeat=1)


        2. profiling range [3,6], [9,12], [15,18]...

        Assume batch 0: skiped, batch 1: closed, batch 2: ready, batch [3,6]: record, repeat

            .. code-block:: python
                :name: code-example2

                import paddle.profiler as profiler
                profiler.make_scheduler(closed=1, ready=1, record=4, skip_first=1)
    """

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
    r"""
    A default state scheduler, keep recording from the begining of the profiler until ending.
    """
    return ProfilerState.RECORD


def export_chrome_tracing(dir_name: str,
                          worker_name: Optional[str]=None) -> Callable:
    r"""
    Return a callable, used for outputing tracing data to chrome tracing format file.
    The output file will be saved in directory ``dir_name``, and file name will be set as worker_name.
    if worker_name is not set, the default name is [hostname]_[pid].

    Args:
        dir_name(str): Directory to save profiling data.
        worker_name(str, optional): Prefix of the file name saved, default is [hostname]_[pid].
    
    Returns:
        A callable, which takes a Profiler object as parameter and calls its export method to save data to chrome tracing format file.

    Examples:
        The return value can be used as parameter ``on_trace_ready`` in :ref:`Profiler <api_paddle_profiler_Profiler>` .

        .. code-block:: python
            :name: code-example1

            # required: gpu
            import paddle.profiler as profiler
            with profiler.Profiler(
                    targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
                    scheduler = (3, 10),
                    on_trace_ready=profiler.export_protobuf('./log')) as p:
                for iter in range(10):
                    #train()
                    p.step()
    """
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name, exist_ok=True)
        except Exception:
            raise RuntimeError(
                "Can not create directory '{}' for saving profiling results.".
                format(dir_name))

    def handle_fn(prof):
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
    r"""
    Return a callable, used for outputing tracing data to protobuf file.
    The output file will be saved in directory ``dir_name``, and file name will be set as worker_name.
    if worker_name is not set, the default name is [hostname]_[pid].

    Args:
        dir_name(str): Directory to save profiling data.
        worker_name(str, optional): Prefix of the file name saved, default is [hostname]_[pid].

    Returns:
        A callable, which takes a Profiler object as parameter and calls its export method to save data to protobuf file.

    Examples:
        The return value can be used as parameter ``on_trace_ready`` in :ref:`Profiler <api_paddle_profiler_Profiler>` .

        .. code-block:: python
            :name: code-example1

            # required: gpu
            import paddle.profiler as profiler
            with profiler.Profiler(
                    targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
                    scheduler = (3, 10),
                    on_trace_ready = profiler.export_protobuf('./log')) as p:
                for iter in range(10):
                    #train()
                    p.step()
    """
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name, exist_ok=True)
        except Exception:
            raise RuntimeError(
                "Can not create directory '{}' for saving profiling results.".
                format(dir_name))

    def handle_fn(prof):
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
    r"""
    Get the current supported profiler target in the system.
    """
    if _Profiler.is_cupti_supported():
        return [ProfilerTarget.CPU, ProfilerTarget.GPU]
    return [ProfilerTarget.CPU]


class Profiler:
    r"""
    Profiler context manager, user interface to manage profiling process to start, stop, export profiling data and print summary table.

    Args:
        targets (list, optional): specify target devices to profile, and all existing and supported devices will be chosen by default. Currently supported values, :ref:`ProfilerTarget.CPU <api_paddle_profiler_ProfilerTarget>` and :ref:`ProfilerTarget.GPU <api_paddle_profiler_ProfilerTarget>` .
        scheduler (Callable|tuple, optional): If it is a callable object, it takes a step number as parameter and return the corresponding :ref:`ProfilerState <api_paddle_profiler_ProfilerState>`. This callable object can be generated by :ref:`make_scheduler <api_paddle_profiler_make_scheduler>` function.
            If not provided (None), the default scheduler will keep tracing until the profiler exits. If it is a tuple, it has two values start_batch and end_batch,
            which means profiling range [start_batch, end_batch).
        on_trace_ready (Callable, optional): Callable object, serves as callback function, and takes the Profiler object as parameter, which provides a way for users to do post-processing.
            This callable object will be called when ``scheduler`` returns ``ProfilerState.RECORD_AND_RETURN``. The default value is :ref:`export_chrome_tracing <api_paddle_profiler_export_chrome_tracing>` (./profiler_log/).
        timer_only (bool, optional): If it is True, the cost of Dataloader and every step of the model will be count without profiling. Otherwise, the model will
            be timed and profiled. Default: False.

    Examples:
        1. profiling range [2, 5).

            .. code-block:: python
                :name: code-example1

                # required: gpu
                import paddle.profiler as profiler
                with profiler.Profiler(
                        targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
                        scheduler = (2, 5),
                        on_trace_ready = profiler.export_chrome_tracing('./log')) as p:
                    for iter in range(10):
                        #train()
                        p.step()

        2. profiling range [2,4], [7, 9], [11,13]

            .. code-block:: python
                :name: code-example2

                # required: gpu
                import paddle.profiler as profiler
                with profiler.Profiler(
                        targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
                        scheduler = profiler.make_scheduler(closed=1, ready=1, record=3, repeat=3),
                        on_trace_ready = profiler.export_chrome_tracing('./log')) as p:
                    for iter in range(10):
                        #train()
                        p.step()

        3. Use profiler without context manager, and use default parameters

            .. code-block:: python
                :name: code-example3

                # required: gpu
                import paddle.profiler as profiler
                p = profiler.Profiler()
                p.start()
                for iter in range(10):
                    #train()
                    p.step()
                p.stop()
                p.summary()

        4. Use profiler to get throughput and cost of the model

            .. code-block:: python
                :name: code-example-timer1

                import paddle
                import paddle.profiler as profiler
                
                class RandomDataset(paddle.io.Dataset):
                    def __init__(self, num_samples):
                        self.num_samples = num_samples
                
                    def __getitem__(self, idx):
                        image = paddle.rand(shape=[100], dtype='float32')
                        label = paddle.randint(0, 10, shape=[1], dtype='int64')
                        return image, label
                
                    def __len__(self):
                        return self.num_samples
                
                class SimpleNet(paddle.nn.Layer):
                    def __init__(self):
                        super(SimpleNet, self).__init__()
                        self.fc = paddle.nn.Linear(100, 10)
                
                    def forward(self, image, label=None):
                        return self.fc(image)
                
                dataset = RandomDataset(20 * 4)
                simple_net = SimpleNet()
                opt = paddle.optimizer.SGD(learning_rate=1e-3,
                                           parameters=simple_net.parameters())
                BATCH_SIZE = 4
                loader = paddle.io.DataLoader(
                    dataset,
                    batch_size=BATCH_SIZE)
                p = profiler.Profiler(timer_only=True)
                p.start()
                for i, (image, label) in enumerate(loader()):
                    out = simple_net(image)
                    loss = paddle.nn.functional.cross_entropy(out, label)
                    avg_loss = paddle.mean(loss)
                    avg_loss.backward()
                    opt.minimize(avg_loss)
                    simple_net.clear_gradients()
                    p.step(num_samples=BATCH_SIZE)
                    if i % 10 == 0:
                        step_info = p.step_info(unit='images')
                        print("Iter {}: {}".format(i, step_info))
                        # The average statistics for 10 steps between the last and this call will be
                        # printed when the "step_info" is called at 10 iteration intervals.
                        # The values you get may be different from the following.
                        # Iter 0:  reader_cost: 0.51946 s batch_cost: 0.66077 s ips: 6.054 images/s
                        # Iter 10:  reader_cost: 0.00014 s batch_cost: 0.00441 s ips: 907.009 images/s
                p.stop()
                # The performance summary will be automatically printed when the "stop" is called.
                # Reader Ratio: 2.658%
                # Time Unit: s, IPS Unit: images/s
                # |                 |       avg       |       max       |       min       |
                # |   reader_cost   |     0.00011     |     0.00013     |     0.00007     |
                # |    batch_cost   |     0.00405     |     0.00434     |     0.00326     |
                # |       ips       |    1086.42904   |    1227.30604   |    959.92796    |
    """

    def __init__(
            self,
            *,
            targets: Optional[Iterable[ProfilerTarget]]=None,
            scheduler: Union[Callable[[int], ProfilerState], tuple, None]=None,
            on_trace_ready: Optional[Callable[..., Any]]=None,
            timer_only: Optional[bool]=False):
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
        if ProfilerTarget.CPU in self.targets:
            profileoption.trace_switch |= 1
        if ProfilerTarget.GPU in self.targets:
            profileoption.trace_switch |= (1 << 1)
        wrap_optimizers()
        self.profiler = _Profiler.create(profileoption)
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
        self.timer_only = timer_only

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        r'''
        Start profiler and enter the first profiler step(0).
        State transformed from CLOSED to self.current_state and trigger corresponding action.

        Examples:
            .. code-block:: python
                :name: code-example4

                # required: gpu
                import paddle.profiler as profiler
                prof = profiler.Profiler(
                    targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
                    scheduler = (1, 9),
                    on_trace_ready = profiler.export_chrome_tracing('./log'))
                prof.start()
                for iter in range(10):
                    #train()
                    prof.step()
                prof.stop()

        '''
        # Timing only without profiling
        benchmark().begin()
        if self.timer_only:
            return
        # CLOSED -> self.current_state
        if self.current_state == ProfilerState.READY:
            self.profiler.prepare()
        elif self.current_state == ProfilerState.RECORD:
            self.profiler.prepare()
            self.profiler.start()
        elif self.current_state == ProfilerState.RECORD_AND_RETURN:
            self.profiler.prepare()
            self.profiler.start()
        self.record_event = RecordEvent(
            name="ProfileStep#{}".format(self.step_num),
            event_type=TracerEventType.ProfileStep)
        self.record_event.begin()

    def stop(self):
        r'''
        Stop profiler and State transformed from self.current_state to CLOSED.
        Trigger corresponding action and post-process profiler result using self.on_trace_ready if result exists.

        Examples:
            .. code-block:: python
                :name: code-example5

                # required: gpu
                import paddle.profiler as profiler
                prof = profiler.Profiler(
                    targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
                    scheduler = (1, 7),
                    on_trace_ready = profiler.export_chrome_tracing('./log'))
                prof.start()
                for iter in range(10):
                    #train()
                    prof.step()
                prof.stop()
        '''
        benchmark().end()
        if self.timer_only:
            return
        # self.current_state -> CLOSED
        # In this situation, RECORD state is regarded as RECORD_AND_RETURN
        if self.record_event:
            self.record_event.end()
            self.record_event = None
        if self.current_state == ProfilerState.READY:
            warn(
                "Inproper Profiler state transform: READY->CLOSED, profiler will start and stop without saving data"
            )
            self.profiler.start()
            self.profiler.stop()
        if self.current_state == ProfilerState.RECORD or self.current_state == ProfilerState.RECORD_AND_RETURN:
            self.profiler_result = self.profiler.stop()
            if self.on_trace_ready:
                self.on_trace_ready(self)

    def step(self, num_samples: Optional[int]=None):
        r"""
        Signals the profiler that the next profiling step has started.
        Get the new ProfilerState and trigger corresponding action.

        Args:
            num_samples (int|None, optional): Specifies the batch size of every step of the model
                that is used to compute throughput when timer_only is True. Default: None.

        Examples:
            .. code-block:: python
                :name: code-example6

                # required: gpu
                import paddle.profiler as profiler
                prof = profiler.Profiler(
                    targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
                    scheduler = (3, 7),
                    on_trace_ready = profiler.export_chrome_tracing('./log'))

                prof.start()
                for iter in range(10):
                    #train()
                    prof.step()
                prof.stop()
        """
        benchmark().step(num_samples)
        if self.timer_only:
            return
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

    def step_info(self, unit=None):
        r"""
        Get statistics for current step. If the function is called at certain iteration
        intervals, the result is the average of all steps between the previous call and
        this call. Statistics are as follows：

        1. reader_cost: the cost of loading data measured in seconds.

        2. batch_cost: the cost of step measured in seconds.

        3. ips(Instance Per Second): the throughput of the model measured in `samples/s`
        or others depends on the `unit`. When `num_samples` of `step()` is None, it is
        measured in `steps/s`.

        Args:
            unit (string, optional): The unit of input data is only used When `num_samples`
                of `step()` is specified as a number. For example, when it is `images`, the unit
                of throughput is `images/s`. Default: None, the unit of throughput is `samples/s`.

        Returns:
            string: A string representing the statistic.

        Examples:
            .. code-block:: python
                :name: code-example-timer2

                import paddle.profiler as profiler
                prof = profiler.Profiler(timer_only=True)
                prof.start()
                for iter in range(20):
                    #train()
                    prof.step()
                    if iter % 10 == 0:
                        print("Iter {}: {}".format(iter, prof.step_info()))
                        # The example does not call the DataLoader, so there is no "reader_cost".
                        # Iter 0:  batch_cost: 0.00001 s ips: 86216.623 steps/s
                        # Iter 10:  batch_cost: 0.00001 s ips: 103645.034 steps/s
                prof.stop()
                # Time Unit: s, IPS Unit: steps/s
                # |                 |       avg       |       max       |       min       |
                # |    batch_cost   |     0.00000     |     0.00002     |     0.00000     |
                # |       ips       |   267846.19437  |   712030.38727  |   45134.16662   |
        """
        if unit is None:
            unit = 'samples'
        return benchmark().step_info(unit)

    def _trigger_action(self):
        if self.previous_state == ProfilerState.CLOSED:
            if self.current_state == ProfilerState.READY:  # CLOSED -> READY
                self.profiler.prepare()
            if self.current_state == ProfilerState.RECORD:  # CLOSED -> RECORD
                self.profiler.prepare()
                self.profiler.start()
            if self.current_state == ProfilerState.RECORD_AND_RETURN:  # CLOSED -> RECORD_AND_RETURN
                self.profiler.prepare()
                self.profiler.start()

        elif self.previous_state == ProfilerState.READY:
            if self.current_state == ProfilerState.CLOSED:  # READY -> CLOSED
                warn(
                    "Improper schedule: READY->CLOSED, profiler will start and stop without saving data"
                )
                self.profiler.start()
                self.profiler.stop()
            if self.current_state == ProfilerState.RECORD:  # READY -> RECORD
                self.profiler.start()
            if self.current_state == ProfilerState.RECORD_AND_RETURN:  # READY -> RECORD_AND_RETURN
                self.profiler.start()

        elif self.previous_state == ProfilerState.RECORD:
            if self.current_state == ProfilerState.CLOSED:  # RECORD -> CLOSED
                warn(
                    "Improper schedule: RECORD->CLOSED, profiler will not saving data"
                )
                self.profiler.stop()

            if self.current_state == ProfilerState.READY:  # RECORD -> READY
                warn(
                    "Improper schedule: RECORD->READY, profiler will stop and re-prepare"
                )
                self.profiler.stop()
                self.profiler.prepare()
            if self.current_state == ProfilerState.RECORD_AND_RETURN:  # RECORD -> RECORD_AND_RETURN
                pass

        else:
            assert self.previous_state == ProfilerState.RECORD_AND_RETURN
            if self.current_state == ProfilerState.CLOSED:  # RECORD_AND_RETURN -> CLOSED
                self.profiler_result = self.profiler.stop()
            if self.current_state == ProfilerState.READY:  # RECORD_AND_RETURN -> READY
                self.profiler_result = self.profiler.stop()
                self.profiler.prepare()
            if self.current_state == ProfilerState.RECORD:  # RECORD_AND_RETURN -> RECORD
                self.profiler_result = self.profiler.stop()
                self.profiler.prepare()
                self.profiler.start()
            if self.current_state == ProfilerState.RECORD_AND_RETURN:  # RECORD_AND_RETURN -> RECORD_AND_RETURN
                self.profiler_result = self.profiler.stop()
                self.profiler.prepare()
                self.profiler.start()
            if self.on_trace_ready:
                self.on_trace_ready(self)

    def export(self, path="", format="json"):
        r"""
        Exports the tracing data to file.

        Args:
            path(str): file path of the output.
            format(str, optional): output format, can be chosen from ['json', 'pb], 'json' for chrome tracing and 'pb' for protobuf, default value is "json".


        Examples:
            .. code-block:: python
                :name: code-example7

                # required: gpu
                import paddle.profiler as profiler
                prof = profiler.Profiler(
                    targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
                    scheduler = (3, 7))
                prof.start()
                for iter in range(10):
                    #train()
                    prof.step()
                prof.stop()
                prof.export(path="./profiler_data.json", format="json")
        """
        if self.profiler_result:
            self.profiler_result.save(path, format)

    def summary(self,
                sorted_by=SortedKeys.CPUTotal,
                op_detail=True,
                thread_sep=False,
                time_unit='ms'):
        r"""
        Print the Summary table. Currently support overview, model, distributed, operator, memory manipulation and userdefined summary.

        Args:
            sorted_by( :ref:`SortedKeys <api_paddle_profiler_SortedKeys>` , optional): how to rank the op table items, default value is SortedKeys.CPUTotal.
            op_detail(bool, optional): expand each operator detail information, default value is True.
            thread_sep(bool, optional): print op table each thread, default value is False.
            time_unit(str, optional): time unit for display, can be chosen form ['s', 'ms', 'us', 'ns'], default value is 'ms'.

        Examples:
            .. code-block:: python
                :name: code-example8

                # required: gpu
                import paddle.profiler as profiler
                prof = profiler.Profiler(
                    targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
                    scheduler = (3, 7),
                    on_trace_ready = profiler.export_chrome_tracing('./log'))
                prof.start()
                for iter in range(10):
                    #train()
                    prof.step()
                prof.stop()
                prof.summary(sorted_by=profiler.SortedKeys.CPUTotal, op_detail=True, thread_sep=False, time_unit='ms')
        """
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
