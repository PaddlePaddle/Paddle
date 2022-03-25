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
import collections
from enum import Enum

from paddle.fluid.core import TracerEventType

from .statistic_helper import *

_AllTracerEventType = [
    TracerEventType.Operator, TracerEventType.Dataloader,
    TracerEventType.ProfileStep, TracerEventType.CudaRuntime,
    TracerEventType.Kernel, TracerEventType.Memcpy, TracerEventType.Memset,
    TracerEventType.UserDefined, TracerEventType.OperatorInner,
    TracerEventType.Forward, TracerEventType.Backward,
    TracerEventType.Optimization, TracerEventType.Communication,
    TracerEventType.PythonOp, TracerEventType.PythonUserDefined
]

_CommunicationOpName = ['reduce', 'broadcast', 'rpc']


class SortedKeys(Enum):
    r"""
    Sorted keys for printing summary table.

    CPUTotal: Sorted by CPU total time.

    CPUAvg: Sorted by CPU average time.

    CPUMax: Sorted by CPU max time.

    CPUMin: Sorted by CPU min time.

    GPUTotal: Sorted by GPU total time.

    GPUAvg: Sorted by GPU average time.

    GPUMax: Sorted by GPU max time.

    GPUMin: Sorted by GPU min time.
    """
    CPUTotal = 0
    CPUAvg = 1
    CPUMax = 2
    CPUMin = 3
    GPUTotal = 4
    GPUAvg = 5
    GPUMax = 6
    GPUMin = 7


class HostStatisticNode:
    r'''
    Wrap original node for calculating statistic metrics.
    '''

    def __init__(self, hostnode):
        self.hostnode = hostnode
        self.children_node = []
        self.runtime_node = []
        self.cpu_time = 0
        self.self_cpu_time = 0
        self.gpu_time = 0
        self.self_gpu_time = 0

    def cal_statistic(self):
        for child in self.children_node:
            child.cal_statistic()
        for rt in self.runtime_node:
            rt.cal_statistic()

        self.cpu_time = self.hostnode.end_ns - self.hostnode.start_ns
        for child in self.children_node:
            self.gpu_time += child.gpu_time
            self.self_cpu_time -= (child.end_ns - child.start_ns)
        for rt in self.runtime_node:
            self.self_cpu_time -= (rt.end_ns - rt.start_ns)
            self.gpu_time += rt.gpu_time
            self.self_gpu_time += rt.gpu_time
        for device in self.hostnode.device_node:
            self.gpu_time += (device.end_ns - device.start_ns)
            self.self_gpu_time += (device.end_ns - device.start_ns)

    @property
    def end_ns(self):
        return self.hostnode.end_ns

    @property
    def start_ns(self):
        return self.hostnode.start_ns

    def __getattr__(self, name):
        return getattr(self.hostnode, name)


def traverse_tree(nodetrees):
    results = collections.defaultdict(list)
    for thread_id, rootnode in nodetrees.items():
        stack = []
        stack.append(rootnode)
        threadlist = results[thread_id]
        while stack:
            current_node = stack.pop()
            threadlist.append(current_node)
            for childnode in current_node.children_node:
                stack.append(childnode)
    return results


def wrap_tree(nodetrees):
    '''
    Using HostStatisticNode to wrap original profiler result tree, and calculate node statistic metrics.
    '''
    node_statistic_tree = {}
    results = collections.defaultdict(list)
    newresults = collections.defaultdict(list)
    for thread_id, rootnode in nodetrees.items():
        stack = []
        stack.append(rootnode)
        root_statistic_node = HostStatisticNode(rootnode)
        newstack = []
        newstack.append(root_statistic_node)
        node_statistic_tree[thread_id] = root_statistic_node
        threadlist = results[thread_id]
        newthreadlist = newresults[thread_id]
        while stack:
            current_node = stack.pop()
            threadlist.append(current_node)
            current_statistic_node = newstack.pop()
            newthreadlist.append(current_statistic_node)
            for childnode in current_node.children_node:
                stack.append(childnode)
                child_statistic_node = HostStatisticNode(childnode)
                current_statistic_node.children_node.append(
                    child_statistic_node)
                newstack.append(child_statistic_node)
            for runtimenode in current_node.runtime_node:
                runtime_statistic_node = HostStatisticNode(runtimenode)
                current_statistic_node.runtime_node.append(
                    runtime_statistic_node)
    # recursive calculate node statistic values
    for thread_id, root_statistic_node in node_statistic_tree.items():
        root_statistic_node.cal_statistic()

    return node_statistic_tree, newresults


class TimeRangeSummary:
    r"""
    Analyse time ranges for each TracerEventType, and summarize the time.
    """

    def __init__(self):
        self.CPUTimeRange = collections.defaultdict(list)
        self.GPUTimeRange = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )  # GPU events should be divided into different devices
        self.CPUTimeRangeSum = collections.defaultdict(int)
        self.GPUTimeRangeSum = collections.defaultdict(
            lambda: collections.defaultdict(int))
        self.call_times = collections.defaultdict(int)

    def parse(self, nodetrees):
        r"""
        Analysis node trees in profiler result, and get time range for different tracer event type.
        """
        thread2hostnodes = traverse_tree(nodetrees)
        for threadid, hostnodes in thread2hostnodes.items():
            CPUTimeRange = collections.defaultdict(list)
            GPUTimeRange = collections.defaultdict(
                lambda: collections.defaultdict(lambda: collections.defaultdict(list))
            )  # device_id/type/stream_id
            for hostnode in hostnodes[1:]:  #skip root node
                CPUTimeRange[hostnode.type].append(
                    (hostnode.start_ns, hostnode.end_ns))
                self.call_times[hostnode.type] += 1
                if hostnode.type == TracerEventType.Operator and any([
                        name in hostnode.name for name in _CommunicationOpName
                ]):  # special case, communication op
                    CPUTimeRange[TracerEventType.Communication].append(
                        (hostnode.start_ns, hostnode.end_ns))
                    self.call_times[TracerEventType.Communication] += 1
                is_communication_node = (
                    hostnode.type == TracerEventType.Communication
                ) or (hostnode.type == TracerEventType.Operator and any(
                    [name in hostnode.name for name in _CommunicationOpName]))
                for runtimenode in hostnode.runtime_node:
                    CPUTimeRange[runtimenode.type].append(
                        (runtimenode.start_ns, runtimenode.end_ns))
                    self.call_times[runtimenode.type] += 1
                    for devicenode in runtimenode.device_node:
                        GPUTimeRange[devicenode.device_id][devicenode.type][
                            devicenode.stream_id].append(
                                (devicenode.start_ns, devicenode.end_ns))
                        self.call_times[devicenode.type] += 1
                        if is_communication_node:  # gpu activity for communication node
                            GPUTimeRange[devicenode.device_id][
                                TracerEventType.Communication][
                                    devicenode.stream_id].append((
                                        devicenode.start_ns, devicenode.end_ns))
                            self.call_times[TracerEventType.Communication] += 1

            for event_type, time_ranges in CPUTimeRange.items():
                time_ranges = merge_self_ranges(time_ranges, is_sorted=False)
                self.CPUTimeRange[event_type] = merge_ranges(
                    self.CPUTimeRange[event_type], time_ranges, is_sorted=True)
            for device_id, device_time_ranges in GPUTimeRange.items():
                for event_type, event_time_ranges in device_time_ranges.items():
                    for stream_id, time_ranges in event_time_ranges.items():
                        time_ranges = merge_self_ranges(
                            time_ranges, is_sorted=False)
                        self.GPUTimeRange[device_id][event_type] = merge_ranges(
                            self.GPUTimeRange[device_id][event_type],
                            time_ranges,
                            is_sorted=True)

        for event_type, time_ranges in self.CPUTimeRange.items():
            self.CPUTimeRangeSum[event_type] = sum_ranges(time_ranges)
        for device_id, device_time_ranges in self.GPUTimeRange.items():
            for event_type, time_ranges in device_time_ranges.items():
                self.GPUTimeRangeSum[device_id][event_type] = sum_ranges(
                    time_ranges)

    def get_gpu_devices(self):
        return self.GPUTimeRange.keys()

    def get_gpu_range_sum(self, device_id, event_type):
        return self.GPUTimeRangeSum[device_id][event_type]

    def get_cpu_range_sum(self, event_type):
        return self.CPUTimeRangeSum[event_type]


class EventSummary:
    r"""
    Analyse operator event in profiling data, correlate with its device event.
    """

    class DeviceItem:
        def __init__(self, name):
            self.name = name
            self.call = 0
            self.gpu_time = 0
            self.max_gpu_time = 0
            self.min_gpu_time = float('inf')

        @property
        def avg_gpu_time(self):
            return self.gpu_time / self.call

        def add_gpu_time(self, time):
            if time > self.max_gpu_time:
                self.max_gpu_time = time
            if time < self.min_gpu_time:
                self.min_gpu_time = time
            self.gpu_time += time

        def add_item(self, node):
            self.call += 1
            self.add_gpu_time(node.end_ns - node.start_ns)

    class OperatorItem:
        def __init__(self, name):
            self.name = name
            self.call = 0
            self.cpu_time = 0
            self.gpu_time = 0
            self.max_cpu_time = 0
            self.min_cpu_time = float('inf')
            self.max_gpu_time = 0
            self.min_gpu_time = float('inf')
            self.devices = {}
            self.operator_inners = {}

        @property
        def avg_cpu_time(self):
            return self.cpu_time / self.call

        @property
        def avg_gpu_time(self):
            return self.gpu_time / self.call

        def add_cpu_time(self, time):
            if time > self.max_cpu_time:
                self.max_cpu_time = time
            if time < self.min_cpu_time:
                self.min_cpu_time = time
            self.cpu_time += time

        def add_gpu_time(self, time):
            if time > self.max_gpu_time:
                self.max_gpu_time = time
            if time < self.min_gpu_time:
                self.min_gpu_time = time
            self.gpu_time += time

        def add_call(self):
            self.call += 1

        def add_item(self, node):
            self.add_call()
            self.add_cpu_time(node.cpu_time)
            self.add_gpu_time(node.gpu_time)
            for child in node.children_node:
                if child.name not in self.operator_inners:
                    self.operator_inners[
                        child.name] = EventSummary.OperatorItem(child.name)
                self.operator_inners[child.name].add_item(child)

            for runtimenode in node.runtime_node:
                for devicenode in runtimenode.device_node:
                    if devicenode.name not in self.devices:
                        self.devices[devicenode.name] = EventSummary.DeviceItem(
                            devicenode.name)
                    self.devices[devicenode.name].add_item(devicenode)

    class GeneralItem:
        def __init__(self, name):
            self.name = name
            self.call = 0
            self.cpu_time = 0
            self.max_cpu_time = 0
            self.min_cpu_time = float('inf')
            self.gpu_time = 0
            self.max_gpu_time = 0
            self.min_gpu_time = float('inf')

        @property
        def avg_cpu_time(self):
            return self.cpu_time / self.call

        @property
        def avg_gpu_time(self):
            return self.gpu_time / self.call

        def add_cpu_time(self, time):
            if time > self.max_cpu_time:
                self.max_cpu_time = time
            if time < self.min_cpu_time:
                self.min_cpu_time = time
            self.cpu_time += time

        def add_gpu_time(self, time):
            if time > self.max_gpu_time:
                self.max_gpu_time = time
            if time < self.min_gpu_time:
                self.min_gpu_time = time
            self.gpu_time += time

        def add_call(self):
            self.call += 1

        def add_item(self, node):
            self.add_call()
            self.add_cpu_time(node.cpu_time)
            self.add_gpu_time(node.gpu_time)

    def __init__(self):
        self.items = {}  # for operator summary
        self.thread_items = collections.defaultdict(
            dict)  # for operator summary
        self.userdefined_items = {}  # for userdefined summary
        self.userdefined_thread_items = collections.defaultdict(
            dict)  # for userdefined summary
        self.model_perspective_items = {}  # for model summary
        self.memory_manipulation_items = {}  # for memory manipulation summary

    def parse(self, nodetrees):
        r"""
        Analysis operator event in the nodetress.
        """
        node_statistic_trees, thread2host_statistic_nodes = wrap_tree(nodetrees)
        for threadid, host_statistic_nodes in thread2host_statistic_nodes.items(
        ):
            for host_statistic_node in host_statistic_nodes[
                    1:]:  #skip root node
                if host_statistic_node.type == TracerEventType.Operator:
                    self.add_operator_item(host_statistic_node)
                if host_statistic_node.type == TracerEventType.UserDefined\
                    or host_statistic_node.type == TracerEventType.PythonUserDefined:
                    if 'memcpy' in host_statistic_node.name.lower() or 'memorycopy' in host_statistic_node.name.lower()\
                        or 'memset' in host_statistic_node.name.lower():
                        self.add_memory_manipulation_item(host_statistic_node)
                    else:
                        self.add_userdefined_item(host_statistic_node)

        for threadid, root_statistic_node in node_statistic_trees.items():
            deque = collections.deque()
            deque.append(root_statistic_node)
            while deque:
                current_node = deque.popleft()
                for child in current_node.children_node:
                    if child.type == TracerEventType.Forward or child.type == TracerEventType.Dataloader\
                        or child.type == TracerEventType.Backward or child.type == TracerEventType.Optimization:
                        self.add_model_perspective_item(
                            child)  #find first model perspective node
                    else:
                        deque.append(child)

    def add_operator_item(self, operator_node):
        if operator_node.name not in self.items:
            self.items[operator_node.name] = EventSummary.OperatorItem(
                operator_node.name)

        self.items[operator_node.name].add_item(operator_node)

        if operator_node.name not in self.thread_items[operator_node.thread_id]:
            self.thread_items[operator_node.thread_id][
                operator_node.name] = EventSummary.OperatorItem(
                    operator_node.name)
        self.thread_items[operator_node.thread_id][operator_node.name].add_item(
            operator_node)

    def add_userdefined_item(self, userdefined_node):
        if userdefined_node.name not in self.userdefined_items:
            self.userdefined_items[
                userdefined_node.name] = EventSummary.GeneralItem(
                    userdefined_node.name)

        self.userdefined_items[userdefined_node.name].add_item(userdefined_node)

        if userdefined_node.name not in self.userdefined_thread_items[
                userdefined_node.thread_id]:
            self.userdefined_thread_items[userdefined_node.thread_id][
                userdefined_node.name] = EventSummary.GeneralItem(
                    userdefined_node.name)
        self.userdefined_thread_items[userdefined_node.thread_id][
            userdefined_node.name].add_item(userdefined_node)

    def add_memory_manipulation_item(self, memory_manipulation_node):
        if memory_manipulation_node.name not in self.memory_manipulation_items:
            self.memory_manipulation_items[
                memory_manipulation_node.name] = EventSummary.GeneralItem(
                    memory_manipulation_node.name)
        self.memory_manipulation_items[memory_manipulation_node.name].add_item(
            memory_manipulation_node)

    def add_model_perspective_item(self, model_perspective_node):
        if model_perspective_node.type == TracerEventType.Forward:
            name = 'Forward'
        elif model_perspective_node.type == TracerEventType.Backward:
            name = 'Backward'
        elif model_perspective_node.type == TracerEventType.Optimization:
            name = 'Optimization'
        elif model_perspective_node.type == TracerEventType.Dataloader:
            name = 'Dataloader'
        else:
            return
        if name not in self.model_perspective_items:
            self.model_perspective_items[name] = EventSummary.GeneralItem(name)
        self.model_perspective_items[name].add_item(model_perspective_node)


class StatisticData:
    r"""
    Hold all analysed results.
    """

    def __init__(self, node_trees, extra_info):
        self.node_trees = node_trees
        self.extra_info = extra_info
        self.time_range_summary = TimeRangeSummary()
        self.event_summary = EventSummary()
        self.time_range_summary.parse(node_trees)
        self.event_summary.parse(node_trees)


def _build_table(statistic_data,
                 sorted_by=SortedKeys.CPUTotal,
                 op_detail=True,
                 thread_sep=False,
                 time_unit='ms',
                 row_limit=100,
                 max_src_column_width=75):
    """Prints a summary of events."""
    # format table row
    SPACING_SIZE = 2
    row_format_list = [""]
    header_sep_list = [""]
    line_length_list = [-SPACING_SIZE]

    def add_column(padding, text_dir='<'):
        row_format_list[0] += '{: ' + text_dir + str(padding) + '}' + (
            ' ' * SPACING_SIZE)
        header_sep_list[0] += '-' * padding + (' ' * SPACING_SIZE)
        line_length_list[0] += padding + SPACING_SIZE

    def add_title(padding, text):
        left_length = padding - len(text)
        half = left_length // 2
        return '-' * half + text + '-' * (left_length - half)

    result = []

    def append(s):
        result.append(s)
        result.append('\n')

    def format_time(time, unit='ms', indent=0):
        r"""
        Transform time in ns to time in unit.
        """
        if time == float('inf'):
            return '-'
        else:
            result = float(time)
            if unit == 's':
                result /= 1e9
            elif unit == 'ms':
                result /= 1e6
            elif unit == 'us':
                result /= 1e3
            return '{}{:.2f}'.format(' ' * indent, result)

    def format_ratio(ratio, indent=0):
        r"""
        Transform ratio within [0, 1] to percentage presentation.
        """
        return '{}{:.2f}'.format(' ' * indent, ratio * 100)

    total_time = statistic_data.time_range_summary.get_cpu_range_sum(
        TracerEventType.ProfileStep)
    ###### Print Device Summary ######
    headers = ['Device', 'Utilization (%)']
    name_column_width = 30
    DEFAULT_COLUMN_WIDTH = 20
    add_column(name_column_width)
    for _ in headers[1:]:
        add_column(DEFAULT_COLUMN_WIDTH)

    row_format = row_format_list[0]
    header_sep = header_sep_list[0]
    line_length = line_length_list[0]

    # construct table string

    append(add_title(line_length, "Device Summary"))
    append('Time unit: {}'.format(time_unit))
    append(header_sep)
    append(row_format.format(*headers))
    append(header_sep)
    row_values = [
        'CPU(Process)', format_ratio(
            float(statistic_data.extra_info['Process Cpu Utilization']))
    ]
    append(row_format.format(*row_values))
    row_values = [
        'CPU(System)', format_ratio(
            float(statistic_data.extra_info['System Cpu Utilization']))
    ]
    append(row_format.format(*row_values))
    for gpu_name in statistic_data.time_range_summary.get_gpu_devices():
        gpu_time = float(
            statistic_data.time_range_summary.get_gpu_range_sum(
                gpu_name, TracerEventType.Kernel))
        utilization = gpu_time / total_time
        row_values = ['GPU{}'.format(gpu_name), format_ratio(utilization)]
        append(row_format.format(*row_values))

    append(header_sep)
    append(
        "Note:\nCPU(Process) Utilization = Current process CPU time over all cpu cores / elapsed time, so max utilization can be reached 100% * number of cpu cores.\n"
        "CPU(System) Utilization = All processes CPU time over all cpu cores(busy time) / (busy time + idle time).\n"
        "GPU Utilization = Current process GPU time / elapsed time")
    append('-' * line_length)
    append('')
    append('')

    if total_time == 0:
        return ''.join(result)

    ###### Print Overview Summary ######
    headers = ['Event Type', 'CPU Time', 'Ratio (%)']
    row_format_list = [""]
    header_sep_list = [""]
    line_length_list = [-SPACING_SIZE]

    DEFAULT_COLUMN_WIDTH = 25
    for _ in headers:
        add_column(DEFAULT_COLUMN_WIDTH)

    row_format = row_format_list[0]
    header_sep = header_sep_list[0]
    line_length = line_length_list[0]

    # construct table string
    append(add_title(line_length, "Overview Summary"))
    append('Time unit: {}'.format(time_unit))
    append(header_sep)
    append(row_format.format(*headers))
    append(header_sep)
    row_values = [
        'Total Time', format_time(
            total_time, unit=time_unit), format_ratio(1)
    ]
    append(row_format.format(*row_values))
    cpu_type_time = collections.defaultdict(int)
    gpu_type_time = collections.defaultdict(int)
    for event_type, value in statistic_data.time_range_summary.CPUTimeRangeSum.items(
    ):
        cpu_type_time[event_type] = value

    gpu_time_range = collections.defaultdict(list)
    for device_id, device_time_ranges in statistic_data.time_range_summary.GPUTimeRange.items(
    ):
        for event_type, time_range in device_time_ranges.items():
            gpu_time_range[event_type] = merge_ranges(
                gpu_time_range[event_type], time_range, is_sorted=True)
    for event_type, time_range in gpu_time_range.items():
        gpu_type_time[event_type] = sum_ranges(time_range)

    sorted_items = sorted(
        cpu_type_time.items(), key=lambda x: x[1], reverse=True)
    for event_type, time in sorted_items:
        row_values = [
            '  {}'.format(str(event_type).split('.')[1]), format_time(
                time, unit=time_unit), format_ratio(float(time) / total_time)
        ]
        append(row_format.format(*row_values))
    append(header_sep)
    headers = ['', 'GPU Time', 'Ratio (%)']
    append(row_format.format(*headers))
    append(header_sep)
    for event_type, time in gpu_type_time.items():
        row_values = [
            '  {}'.format(str(event_type).split('.')[1]), format_time(
                time, unit=time_unit), format_ratio(float(time) / total_time)
        ]
        append(row_format.format(*row_values))

    append(header_sep)
    append(
        "Note:\nIn this table, We sum up all collected events in terms of event type.\n"
        "The time of events collected on host are presented as CPU Time, and as GPU Time if on device.\n"
        "ratio = CPU(GPU) Time / Total Time."
        "Events with different types may overlap or inclusion, e.g. Operator includes OperatorInner, so the sum of ratios is not 100%.\n"
        "The time of events in the same type with overlap will not calculate twice, and all time is summed after merged.\n"
        "Example:\n"
        "Thread 1:\n"
        "  Operator: |___________|     |__________|\n"
        "Thread 2:\n"
        "  Operator:   |____________|     |___|\n"
        "After merged:\n"
        "  Result:   |______________|  |__________|\n")
    append('-' * line_length)
    append('')
    append('')

    ###### Print Model Summary Report ######
    model_perspective_items = statistic_data.event_summary.model_perspective_items
    if model_perspective_items:
        headers = [
            'Name', 'Calls', 'CPU Total / Avg / Max / Min / Ratio(%)',
            'GPU Total / Avg / Max / Min / Ratio(%)'
        ]
        row_format_list = [""]
        header_sep_list = [""]
        line_length_list = [-SPACING_SIZE]
        name_column_width = 15
        add_column(name_column_width)
        add_column(6)
        add_column(40)
        add_column(40)

        row_format = row_format_list[0]
        header_sep = header_sep_list[0]
        line_length = line_length_list[0]

        # construct table string
        append(add_title(line_length, "Model Summary"))
        append('Time unit: {}'.format(time_unit))
        append(header_sep)
        append(row_format.format(*headers))
        append(header_sep)
        accmulation_time = 0
        row_values = [
            'Total Time', '-', '{} / - / - / - / {}'.format(
                format_time(
                    total_time, unit=time_unit), format_ratio(1)),
            '- / - / - / -/ -'
        ]
        append(row_format.format(*row_values))
        for name in ['Dataloader', 'Forward', 'Backward', 'Optimization']:
            if name in model_perspective_items:
                item = model_perspective_items[name]
                row_values = [
                    '  {}'.format(name), item.call,
                    '{} / {} / {} / {} / {}'.format(
                        format_time(
                            item.cpu_time, unit=time_unit),
                        format_time(
                            item.avg_cpu_time, unit=time_unit),
                        format_time(
                            item.max_cpu_time, unit=time_unit),
                        format_time(
                            item.min_cpu_time, unit=time_unit),
                        format_ratio(float(item.cpu_time) / total_time)),
                    '{} / {} / {} / {} / {}'.format(
                        format_time(
                            item.gpu_time, unit=time_unit),
                        format_time(
                            item.avg_gpu_time, unit=time_unit),
                        format_time(
                            item.max_gpu_time, unit=time_unit),
                        format_time(
                            item.min_gpu_time, unit=time_unit),
                        format_ratio(float(item.gpu_time) / total_time))
                ]
                append(row_format.format(*row_values))
                accmulation_time += item.cpu_time

        other_time = total_time - accmulation_time
        row_values = [
            '  Others', '-', '{} / - / - / - / {}'.format(
                format_time(
                    other_time, unit=time_unit),
                format_ratio(float(other_time) / total_time)),
            '- / - / - / - / -'
        ]
        append(row_format.format(*row_values))
        append(header_sep)
        append('')
        append('')

    ###### Print Distribution Summary Report ######
    if TracerEventType.Communication in statistic_data.time_range_summary.CPUTimeRange:
        headers = [
            'Name',
            'Total Time',
            'Ratio (%)',
        ]
        row_format_list = [""]
        header_sep_list = [""]
        line_length_list = [-SPACING_SIZE]

        DEFAULT_COLUMN_WIDTH = 20
        for _ in headers:
            add_column(DEFAULT_COLUMN_WIDTH)

        row_format = row_format_list[0]
        header_sep = header_sep_list[0]
        line_length = line_length_list[0]

        # construct table string
        append(add_title(line_length, "Distribution Summary"))
        append('Time unit: {}'.format(time_unit))
        append(header_sep)
        append(row_format.format(*headers))
        append(header_sep)
        cpu_communication_time_range = []
        gpu_communication_time_range = []
        cpu_communication_time_range = merge_ranges(
            statistic_data.time_range_summary.CPUTimeRange[
                TracerEventType.Communication], cpu_communication_time_range)
        kernel_time_range = []
        for device_id, device_time_ranges in statistic_data.time_range_summary.GPUTimeRange.items(
        ):
            kernel_time_range = merge_ranges(
                device_time_ranges[TracerEventType.Kernel],
                kernel_time_range,
                is_sorted=True)
            gpu_communication_time_range = merge_ranges(
                device_time_ranges[TracerEventType.Communication],
                gpu_communication_time_range,
                is_sorted=True)
        communication_time_range = merge_ranges(
            cpu_communication_time_range,
            gpu_communication_time_range,
            is_sorted=True)
        computation_time_range = subtract_ranges(kernel_time_range,
                                                 gpu_communication_time_range)
        overlap_time_range = intersection_ranges(communication_time_range,
                                                 computation_time_range)
        communication_time = sum_ranges(communication_time_range)
        computation_time = sum_ranges(computation_time_range)
        overlap_time = sum_ranges(overlap_time_range)
        row_values = [
            'Communication', format_time(
                communication_time, unit=time_unit),
            format_ratio(float(communication_time) / total_time)
        ]
        append(row_format.format(*row_values))

        row_values = [
            'Computation', format_time(
                computation_time, unit=time_unit),
            format_ratio(float(computation_time) / total_time)
        ]
        append(row_format.format(*row_values))

        row_values = [
            'Overlap', format_time(
                overlap_time, unit=time_unit),
            format_ratio(float(overlap_time) / total_time)
        ]
        append(row_format.format(*row_values))
        append(header_sep)
        append(
            "Note:\nCommunication time: Communication Op time and its kernel time on gpu.\n"
            "Computation time: Kernel time, substract kernels belong to communication op.\n"
            "Overlap time: Communication time intersect with computation time.\n"
            "Example:\n"
            "Communication:\n"
            "  CPU:              |_________________|\n"
            "  GPU:                                  |______________|\n"
            "  Total:            |_________________| |______________|\n"
            "Computation time(Kernel):\n"
            "  GPU:         |________________|\n"
            "Overlap time:       |___________|\n")
        append('-' * line_length)
        append('')
        append('')

    ###### Print Operator Summary Report ######
    if statistic_data.event_summary.items:
        headers = [
            'Name', 'Calls', 'CPU Total / Avg / Max / Min / Ratio(%)',
            'GPU Total / Avg / Max / Min / Ratio(%)'
        ]
        row_format_list = [""]
        header_sep_list = [""]
        line_length_list = [-SPACING_SIZE]
        name_column_width = 50
        add_column(name_column_width)
        add_column(6)
        add_column(40)
        add_column(40)

        row_format = row_format_list[0]
        header_sep = header_sep_list[0]
        line_length = line_length_list[0]

        # construct table string
        append(add_title(line_length, "Operator Summary"))
        append('Time unit: {}'.format(time_unit))
        append(header_sep)
        append(row_format.format(*headers))
        append(header_sep)
        if thread_sep == True:
            thread_items = statistic_data.event_summary.thread_items
        else:
            thread_items = {
                'All threads merged': statistic_data.event_summary.items
            }
        for thread_id, items in thread_items.items():
            append(add_title(line_length, "Thread: {}".format(thread_id)))
            if sorted_by == SortedKeys.CPUTotal:
                sorted_items = sorted(
                    items.items(), key=lambda x: x[1].cpu_time, reverse=True)
            elif sorted_by == SortedKeys.CPUAvg:
                sorted_items = sorted(
                    items.items(),
                    key=lambda x: x[1].avg_cpu_time,
                    reverse=True)
            elif sorted_by == SortedKeys.CPUMax:
                sorted_items = sorted(
                    items.items(),
                    key=lambda x: x[1].max_cpu_time,
                    reverse=True)
            elif sorted_by == SortedKeys.CPUMin:
                sorted_items = sorted(
                    items.items(), key=lambda x: x[1].min_cpu_time)
            elif sorted_by == SortedKeys.GPUTotal:
                sorted_items = sorted(
                    items.items(), key=lambda x: x[1].gpu_time, reverse=True)
            elif sorted_by == SortedKeys.GPUAvg:
                sorted_items = sorted(
                    items.items(),
                    key=lambda x: x[1].avg_gpu_time,
                    reverse=True)
            elif sorted_by == SortedKeys.GPUMax:
                sorted_items = sorted(
                    items.items(),
                    key=lambda x: x[1].max_gpu_time,
                    reverse=True)
            elif sorted_by == SortedKeys.GPUMin:
                sorted_items = sorted(
                    items.items(), key=lambda x: x[1].min_gpu_time)

            for name, item in sorted_items:
                row_values = [
                    name, item.call, '{} / {} / {} / {} / {}'.format(
                        format_time(
                            item.cpu_time, unit=time_unit),
                        format_time(
                            item.avg_cpu_time, unit=time_unit),
                        format_time(
                            item.max_cpu_time, unit=time_unit),
                        format_time(
                            item.min_cpu_time, unit=time_unit),
                        format_ratio(float(item.cpu_time) / total_time)),
                    '{} / {} / {} / {} / {}'.format(
                        format_time(
                            item.gpu_time, unit=time_unit),
                        format_time(
                            item.avg_gpu_time, unit=time_unit),
                        format_time(
                            item.max_gpu_time, unit=time_unit),
                        format_time(
                            item.min_gpu_time, unit=time_unit),
                        format_ratio(float(item.gpu_time) / total_time))
                ]
                append(row_format.format(*row_values))
                if op_detail:
                    for innerop_name, innerop_node in item.operator_inners.items(
                    ):
                        row_values = [
                            '  {}'.format(innerop_name), innerop_node.call,
                            '{} / {} / {} / {} / {}'.format(
                                format_time(
                                    innerop_node.cpu_time, unit=time_unit),
                                format_time(
                                    innerop_node.avg_cpu_time, unit=time_unit),
                                format_time(
                                    innerop_node.max_cpu_time, unit=time_unit),
                                format_time(
                                    innerop_node.min_cpu_time, unit=time_unit),
                                format_ratio(
                                    float(innerop_node.cpu_time) / total_time)),
                            '{} / {} / {} / {} / {}'.format(
                                format_time(
                                    innerop_node.gpu_time, unit=time_unit),
                                format_time(
                                    innerop_node.avg_gpu_time, unit=time_unit),
                                format_time(
                                    innerop_node.max_gpu_time, unit=time_unit),
                                format_time(
                                    innerop_node.min_gpu_time, unit=time_unit),
                                format_ratio(
                                    float(innerop_node.gpu_time) / total_time))
                        ]
                        append(row_format.format(*row_values))
                        for device_node_name, devicenode in innerop_node.devices.items(
                        ):
                            if len(device_node_name) + 4 > name_column_width:
                                device_node_name = device_node_name[:
                                                                    name_column_width
                                                                    - 7]
                                device_node_name += "..."
                            row_values = [
                                '    {}'.format(device_node_name),
                                devicenode.call, '- / - / - / - / -',
                                '{} / {} / {} / {} / {}'.format(
                                    format_time(
                                        devicenode.gpu_time, unit=time_unit),
                                    format_time(
                                        devicenode.avg_gpu_time,
                                        unit=time_unit),
                                    format_time(
                                        devicenode.max_gpu_time,
                                        unit=time_unit),
                                    format_time(
                                        devicenode.min_gpu_time,
                                        unit=time_unit),
                                    format_ratio(
                                        float(devicenode.gpu_time) /
                                        total_time))
                            ]
                            append(row_format.format(*row_values))
                    for device_node_name, device_node in item.devices.items():
                        if len(device_node_name) + 2 > name_column_width:
                            device_node_name = device_node_name[:
                                                                name_column_width
                                                                - 5]
                            device_node_name += "..."
                        row_values = [
                            '    {}'.format(device_node_name), devicenode.call,
                            '- / - / - / - / -',
                            '{} / {} / {} / {} / {}'.format(
                                format_time(
                                    devicenode.gpu_time, unit=time_unit),
                                format_time(
                                    devicenode.avg_gpu_time, unit=time_unit),
                                format_time(
                                    devicenode.max_gpu_time, unit=time_unit),
                                format_time(
                                    devicenode.min_gpu_time, unit=time_unit),
                                format_ratio(
                                    float(devicenode.gpu_time) / total_time))
                        ]
                        append(row_format.format(*row_values))
        append(header_sep)
        append('')
        append('')

    ###### Print Memory Manipulation Summary Report ######
    if statistic_data.event_summary.memory_manipulation_items:
        headers = [
            'Name', 'Calls', 'CPU Total / Avg / Max / Min / Ratio(%)',
            'GPU Total / Avg / Max / Min / Ratio(%)'
        ]
        row_format_list = [""]
        header_sep_list = [""]
        line_length_list = [-SPACING_SIZE]
        name_column_width = 30
        add_column(name_column_width)
        add_column(6)
        add_column(40)
        add_column(40)

        row_format = row_format_list[0]
        header_sep = header_sep_list[0]
        line_length = line_length_list[0]

        # construct table string
        append(add_title(line_length, "Memory Manipulation Summary"))
        append('Time unit: {}'.format(time_unit))
        append(header_sep)
        append(row_format.format(*headers))
        append(header_sep)
        memory_manipulation_items = statistic_data.event_summary.memory_manipulation_items
        for name, item in memory_manipulation_items.items():
            row_values = [
                name,
                item.call,
                '{} / {} / {} / {} / {}'.format(
                    format_time(
                        item.cpu_time, unit=time_unit),
                    format_time(
                        item.avg_cpu_time, unit=time_unit),
                    format_time(
                        item.max_cpu_time, unit=time_unit),
                    format_time(
                        item.min_cpu_time, unit=time_unit),
                    format_ratio(float(item.cpu_time) / total_time)),
                '{} / {} / {} / {} / {}'.format(
                    format_time(
                        item.gpu_time, unit=time_unit),
                    format_time(
                        item.avg_gpu_time, unit=time_unit),
                    format_time(
                        item.max_gpu_time, unit=time_unit),
                    format_time(
                        item.min_gpu_time, unit=time_unit),
                    format_ratio(float(item.gpu_time) / total_time)),
            ]
            append(row_format.format(*row_values))
        append(header_sep)
        append('')
        append('')
    ###### Print UserDefined Summary Report ######
    if statistic_data.event_summary.userdefined_items:
        headers = [
            'Name', 'Calls', 'CPU Total / Avg / Max / Min / Ratio(%)',
            'GPU Total / Avg / Max / Min / Ratio(%)'
        ]
        row_format_list = [""]
        header_sep_list = [""]
        line_length_list = [-SPACING_SIZE]
        name_column_width = 30
        add_column(name_column_width)
        add_column(6)
        add_column(40)
        add_column(40)

        row_format = row_format_list[0]
        header_sep = header_sep_list[0]
        line_length = line_length_list[0]

        # construct table string
        append(add_title(line_length, "UserDefined Summary"))
        append('Time unit: {}'.format(time_unit))
        append(header_sep)
        append(row_format.format(*headers))
        append(header_sep)
        if thread_sep == True:
            userdefined_thread_items = statistic_data.event_summary.userdefined_thread_items
        else:
            userdefined_thread_items = {
                'All threads merged':
                statistic_data.event_summary.userdefined_items
            }
        for thread_id, items in userdefined_thread_items.items():
            append(add_title(line_length, "Thread: {}".format(thread_id)))
            if sorted_by == SortedKeys.CPUTotal:
                sorted_items = sorted(
                    items.items(), key=lambda x: x[1].cpu_time, reverse=True)
            elif sorted_by == SortedKeys.CPUAvg:
                sorted_items = sorted(
                    items.items(),
                    key=lambda x: x[1].avg_cpu_time,
                    reverse=True)
            elif sorted_by == SortedKeys.CPUMax:
                sorted_items = sorted(
                    items.items(),
                    key=lambda x: x[1].max_cpu_time,
                    reverse=True)
            elif sorted_by == SortedKeys.CPUMin:
                sorted_items = sorted(
                    items.items(), key=lambda x: x[1].min_cpu_time)
            elif sorted_by == SortedKeys.GPUTotal:
                sorted_items = sorted(
                    items.items(), key=lambda x: x[1].gpu_time, reverse=True)
            elif sorted_by == SortedKeys.GPUAvg:
                sorted_items = sorted(
                    items.items(),
                    key=lambda x: x[1].avg_gpu_time,
                    reverse=True)
            elif sorted_by == SortedKeys.GPUMax:
                sorted_items = sorted(
                    items.items(),
                    key=lambda x: x[1].max_gpu_time,
                    reverse=True)
            elif sorted_by == SortedKeys.GPUMin:
                sorted_items = sorted(
                    items.items(), key=lambda x: x[1].min_gpu_time)

            for name, item in sorted_items:
                row_values = [
                    name,
                    item.call,
                    '{} / {} / {} / {} / {}'.format(
                        format_time(
                            item.cpu_time, unit=time_unit),
                        format_time(
                            item.avg_cpu_time, unit=time_unit),
                        format_time(
                            item.max_cpu_time, unit=time_unit),
                        format_time(
                            item.min_cpu_time, unit=time_unit),
                        format_ratio(float(item.cpu_time) / total_time)),
                    '{} / {} / {} / {} / {}'.format(
                        format_time(
                            item.gpu_time, unit=time_unit),
                        format_time(
                            item.avg_gpu_time, unit=time_unit),
                        format_time(
                            item.max_gpu_time, unit=time_unit),
                        format_time(
                            item.min_gpu_time, unit=time_unit),
                        format_ratio(float(item.gpu_time) / total_time)),
                ]
                append(row_format.format(*row_values))
            append(header_sep)
    return ''.join(result)
