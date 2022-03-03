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
    Sorted keys for printing op summary table.
    """
    OpTotal = 0
    OpAvg = 1
    OpMax = 2
    OpMin = 3
    KernelTotal = 4
    KernelAvg = 5
    KernelMax = 6
    KernelMin = 7


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
                if hostnode.type != TracerEventType.OperatorInner:
                    CPUTimeRange[hostnode.type].append(
                        (hostnode.start_ns, hostnode.end_ns))
                else:
                    if 'infer_shape' in hostnode.name:
                        CPUTimeRange['infer_shape'].append(
                            (hostnode.start_ns, hostnode.end_ns))
                    elif 'prepare_data' in hostnode.name:
                        CPUTimeRange['prepare_data'].append(
                            (hostnode.start_ns, hostnode.end_ns))
                    elif 'compute' in hostnode.name:
                        CPUTimeRange['compute'].append(
                            (hostnode.start_ns, hostnode.end_ns))

                if hostnode.type == TracerEventType.Operator and any(
                    [name in hostnode.name for name in
                     _CommunicationOpName]):  # special case, communication op
                    CPUTimeRange[TracerEventType.Communication].append(
                        (hostnode.start_ns, hostnode.end_ns))
                is_communication_node = (
                    hostnode.type == TracerEventType.Communication
                ) or (hostnode.type == TracerEventType.Operator and any(
                    [name in hostnode.name for name in _CommunicationOpName]))
                for runtimenode in hostnode.runtime_node:
                    CPUTimeRange[runtimenode.type].append(
                        (runtimenode.start_ns, runtimenode.end_ns))
                    for devicenode in runtimenode.device_node:
                        GPUTimeRange[devicenode.device_id][devicenode.type][
                            devicenode.stream_id].append(
                                (devicenode.start_ns, devicenode.end_ns))
                        if is_communication_node:  # gpu activity for communication node
                            GPUTimeRange[devicenode.device_id][
                                TracerEventType.Communication][
                                    devicenode.stream_id].append((
                                        devicenode.start_ns, devicenode.end_ns))

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
            self.infer_shape_time = 0
            self.prepare_data_time = 0
            self.computation_time = 0
            self.max_cpu_time = 0
            self.min_cpu_time = float('inf')
            self.max_infer_shape_time = 0
            self.min_infer_shape_time = float('inf')
            self.max_prepare_data_time = 0
            self.min_prepare_data_time = float('inf')
            self.max_computation_time = 0
            self.min_computation_time = float('inf')
            self.max_gpu_time = 0
            self.min_gpu_time = float('inf')
            self.devices = {}

        @property
        def avg_cpu_time(self):
            return self.cpu_time / self.call

        @property
        def avg_gpu_time(self):
            return self.gpu_time / self.call

        @property
        def avg_infer_shape_time(self):
            return self.infer_shape_time / self.call

        @property
        def avg_prepare_data_time(self):
            return self.prepare_data_time / self.call

        @property
        def avg_computation_time(self):
            return self.computation_time / self.call

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

        def add_infer_shape_time(self, time):
            if time > self.max_infer_shape_time:
                self.max_infer_shape_time = time
            if time < self.min_infer_shape_time:
                self.min_infer_shape_time = time
            self.infer_shape_time += time

        def add_prepare_data_time(self, time):
            if time > self.max_prepare_data_time:
                self.max_prepare_data_time = time
            if time < self.min_prepare_data_time:
                self.min_prepare_data_time = time
            self.prepare_data_time += time

        def add_computation_time(self, time):
            if time > self.max_computation_time:
                self.max_computation_time = time
            if time < self.min_computation_time:
                self.min_computation_time = time
            self.computation_time += time

        def add_call(self):
            self.call += 1

        def add_item(self, node):
            self.call += 1
            self.add_cpu_time(node.end_ns - node.start_ns)
            for child in node.children_node:
                if child.type == TracerEventType.OperatorInner:
                    if 'infer_shape' in child.name:
                        self.add_infer_shape_time(child.end_ns - child.start_ns)
                    if 'prepare_data' in child.name:
                        self.add_prepare_data_time(child.end_ns -
                                                   child.start_ns)
                    if 'compute' in child.name:
                        self.add_computation_time(child.end_ns - child.start_ns)
                for runtimenode in child.runtime_node:
                    for devicenode in runtimenode.device_node:
                        if devicenode.type == TracerEventType.Kernel:
                            self.add_gpu_time(devicenode.end_ns -
                                              devicenode.start_ns)
                        if devicenode.name in self.devices:
                            self.devices[devicenode.name].add_item(devicenode)
                        else:
                            self.devices[
                                devicenode.name] = EventSummary.DeviceItem(
                                    devicenode.name)
                            self.devices[devicenode.name].add_item(devicenode)
            for runtimenode in node.runtime_node:
                for devicenode in runtimenode.device_node:
                    if devicenode.type == TracerEventType.Kernel:
                        self.add_gpu_time(devicenode.end_ns -
                                          devicenode.start_ns)
                    if devicenode.name in self.devices:
                        self.devices[devicenode.name].add_item(devicenode)
                    else:
                        self.devices[devicenode.name] = EventSummary.DeviceItem(
                            devicenode.name)
                        self.devices[devicenode.name].add_item(devicenode)

    class UserDefinedItem:
        def __init__(self, name):
            self.name = name
            self.call = 0
            self.cpu_time = 0
            self.max_cpu_time = 0
            self.min_cpu_time = float('inf')

        @property
        def avg_cpu_time(self):
            return self.cpu_time / self.call

        def add_cpu_time(self, time):
            if time > self.max_cpu_time:
                self.max_cpu_time = time
            if time < self.min_cpu_time:
                self.min_cpu_time = time
            self.cpu_time += time

        def add_item(self, node):
            self.call += 1
            self.add_cpu_time(node.end_ns - node.start_ns)

    def __init__(self):
        self.items = {}  # for operator
        self.thread_items = collections.defaultdict(dict)  # for operator
        self.userdefined_items = {}  # for userdefined
        self.userdefined_thread_items = collections.defaultdict(
            dict)  # for userdefined

    def parse(self, nodetrees):
        r"""
        Analysis operator event in the nodetress.
        """
        thread2hostnodes = traverse_tree(nodetrees)
        for threadid, hostnodes in thread2hostnodes.items():
            for hostnode in hostnodes[1:]:  #skip root node
                if hostnode.type == TracerEventType.Operator:
                    self.add_operator_item(hostnode)
                if hostnode.type == TracerEventType.UserDefined or hostnode.type == TracerEventType.PythonUserDefined:
                    self.add_userdefined_item(hostnode)

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
                userdefined_node.name] = EventSummary.UserDefinedItem(
                    userdefined_node.name)

        self.userdefined_items[userdefined_node.name].add_item(userdefined_node)

        if userdefined_node.name not in self.userdefined_thread_items[
                userdefined_node.thread_id]:
            self.userdefined_thread_items[userdefined_node.thread_id][
                userdefined_node.name] = EventSummary.UserDefinedItem(
                    userdefined_node.name)
        self.userdefined_thread_items[userdefined_node.thread_id][
            userdefined_node.name].add_item(userdefined_node)


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
                 sorted_by=SortedKeys.OpTotal,
                 op_detail=True,
                 thread_sep=False,
                 time_unit='ms',
                 row_limit=100,
                 max_src_column_width=75):
    pass
