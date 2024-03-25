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
import re
from enum import Enum

from paddle.base.core import TracerEventType, TracerMemEventType
from paddle.utils.flops import flops

from .statistic_helper import (
    intersection_ranges,
    merge_ranges,
    merge_self_ranges,
    sum_ranges,
)

_AllTracerEventType = [
    TracerEventType.Operator,
    TracerEventType.Dataloader,
    TracerEventType.ProfileStep,
    TracerEventType.CudaRuntime,
    TracerEventType.Kernel,
    TracerEventType.Memcpy,
    TracerEventType.Memset,
    TracerEventType.UserDefined,
    TracerEventType.OperatorInner,
    TracerEventType.Forward,
    TracerEventType.Backward,
    TracerEventType.Optimization,
    TracerEventType.Communication,
    TracerEventType.PythonOp,
    TracerEventType.PythonUserDefined,
]

_CommunicationOpName = ['allreduce', 'broadcast', 'rpc']


class SortedKeys(Enum):
    r"""
    SortedKeys is used to specify how to sort items when printing ``paddle.profiler.Profiler.summary`` table.

    The meaning of each SortedKeys is as following

    - **SortedKeys.CPUTotal** :  Sorted by CPU total time.

    - **SortedKeys.CPUAvg**  : Sorted by CPU average time.

    - **SortedKeys.CPUMax**  : Sorted by CPU max time.

    - **SortedKeys.CPUMin**  : Sorted by CPU min time.

    - **SortedKeys.GPUTotal**  : Sorted by GPU total time.

    - **SortedKeys.GPUAvg**  : Sorted by GPU average time.

    - **SortedKeys.GPUMax**  : Sorted by GPU max time.

    - **SortedKeys.GPUMin**  : Sorted by GPU min time.
    """
    CPUTotal = 0
    CPUAvg = 1
    CPUMax = 2
    CPUMin = 3
    GPUTotal = 4
    GPUAvg = 5
    GPUMax = 6
    GPUMin = 7


def _nodename2opname(name):
    r'''
    convert static host node name to operator name
    '''
    op_name = name.replace(' compute', '')
    op_name = op_name.replace(' dygraph', '')
    op_name = op_name.replace(' pybind_imperative_func', '')
    return op_name


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
        self.gpu_time = 0  # kernel time
        self.self_gpu_time = 0
        self.general_gpu_time = 0  # besides kernel, include time of gpu events like memcpy and memset
        self.self_general_gpu_time = 0
        self.flops = 0

    def cal_flops(self):
        if self.hostnode.type == TracerEventType.Operator:
            if hasattr(self.hostnode, 'input_shapes'):
                op_name = _nodename2opname(self.hostnode.name)
                self.flops = flops(
                    op_name,
                    self.hostnode.input_shapes,
                    self.hostnode.attributes,
                )

    def cal_statistic(self):
        self.cpu_time = self.hostnode.end_ns - self.hostnode.start_ns
        self.self_cpu_time = self.cpu_time
        self.cal_flops()
        for child in self.children_node:
            child.cal_flops()
            child.cal_statistic()
            self.gpu_time += child.gpu_time
            self.general_gpu_time += child.general_gpu_time
            self.self_cpu_time -= child.end_ns - child.start_ns
            self.flops += child.flops

        for rt in self.runtime_node:
            rt.cal_statistic()
            self.self_cpu_time -= rt.end_ns - rt.start_ns
            self.gpu_time += rt.gpu_time
            self.self_gpu_time += rt.gpu_time
            self.general_gpu_time += rt.general_gpu_time
            self.self_general_gpu_time += rt.general_gpu_time

        for device in self.hostnode.device_node:
            if device.type == TracerEventType.Kernel:
                self.gpu_time += device.end_ns - device.start_ns
                self.self_gpu_time += device.end_ns - device.start_ns
            self.general_gpu_time += device.end_ns - device.start_ns
            self.self_general_gpu_time += device.end_ns - device.start_ns

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


def get_device_nodes(hostnode):
    '''
    Get all device nodes called in the time range of hostnode.
    '''
    stack = []
    device_nodes = []
    stack.append(hostnode)
    while stack:
        current_node = stack.pop()
        for childnode in current_node.children_node:
            stack.append(childnode)
        for runtimenode in current_node.runtime_node:
            for devicenode in runtimenode.device_node:
                device_nodes.append(devicenode)
    return device_nodes


def _build_layer_from_tree(nodetrees):
    def build_layer(node, depth=0):
        if "GradNode" in node.name:
            return [], 0

        if node.type in [
            TracerEventType.Backward,
            TracerEventType.Optimization,
        ]:
            return [], 0

        if node.type == TracerEventType.Operator:
            stat_node = HostStatisticNode(node)
            stat_node.cal_statistic()
            return stat_node, stat_node.flops

        layer = []
        nflops = 0
        for c in node.children_node:
            l, f = build_layer(c, depth + 1)
            if l:
                nflops += f
                layer.append(l)

        if node.type == TracerEventType.Forward:
            stat_node = HostStatisticNode(node)
            stat_node.cal_statistic()
            stat_node.flops = nflops
            return [stat_node, layer], nflops

        return layer, nflops

    ret = []
    for _, rootnode in nodetrees.items():
        layer, _ = build_layer(rootnode)
        ret.append(layer)

    return ret


def _format_large_number(n, precision=2):
    if n // 1e12 > 0:
        return f"{round(n / 1e12, precision)} T"
    if n // 1e9 > 0:
        return f"{round(n / 1e9, precision)} G"
    if n // 1e6 > 0:
        return f"{round(n / 1e6, precision)} M"
    if n // 1e3 > 0:
        return f"{round(n / 1e3, precision)} K"
    return f"{round(n, precision)}"


def _format_time(n, precision=2):
    if n // 1e9 > 0:
        return f"{round(n / 1e9, precision)} s"
    if n // 1e6 > 0:
        return f"{round(n / 1e6, precision)} ms"
    if n // 1e3 > 0:
        return f"{round(n / 1e3, precision)} us"
    return f"{round(n, precision)} ns"


def _gen_layer_flops(node, repeat=1):
    ret = []
    offset = []
    loop = []

    def print_layer_tree(node, depth=0):
        if isinstance(node, list):
            for n in node:
                print_layer_tree(n, depth + 1)

        elif node.type in [TracerEventType.Forward, TracerEventType.Operator]:
            if len(offset) == 0:
                offset.append(depth)

            name = _nodename2opname(node.name)

            if (
                depth == offset[-1] and len(ret) > 0 and ret[0].startswith(name)
            ):  # repeat begin
                loop.append(1)

            if len(loop) >= repeat:
                return "".join(ret)

            align = " " * (depth - offset[-1])
            tm = _format_time(node.cpu_time)
            flops_n = _format_large_number(node.flops)
            flops_s = _format_large_number(node.flops * 1e9 / node.cpu_time)
            ret.append(
                f"{align}{name} latency: {tm}, FLOPs: {flops_n}, FLOPS: {flops_s}\n"
            )

    for n in node[1:]:
        print_layer_tree(n)

    return "".join(ret)


def gen_layer_flops(nodetrees, repeat=1):
    r'''
    gen_layer_flops generate flops/runtime information depend on layer/operator.
    '''
    layer_tree = _build_layer_from_tree(nodetrees)
    return _gen_layer_flops(layer_tree, repeat)


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
                    child_statistic_node
                )
                newstack.append(child_statistic_node)
            for runtimenode in current_node.runtime_node:
                runtime_statistic_node = HostStatisticNode(runtimenode)
                current_statistic_node.runtime_node.append(
                    runtime_statistic_node
                )
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
            lambda: collections.defaultdict(int)
        )
        self.call_times = collections.defaultdict(int)

    def parse(self, nodetrees):
        r"""
        Analysis node trees in profiler result, and get time range for different tracer event type.
        """
        thread2hostnodes = traverse_tree(nodetrees)
        for threadid, hostnodes in thread2hostnodes.items():
            CPUTimeRange = collections.defaultdict(list)
            GPUTimeRange = collections.defaultdict(
                lambda: collections.defaultdict(
                    lambda: collections.defaultdict(list)
                )
            )  # device_id/type/stream_id
            for hostnode in hostnodes[1:]:  # skip root node
                CPUTimeRange[hostnode.type].append(
                    (hostnode.start_ns, hostnode.end_ns)
                )
                self.call_times[hostnode.type] += 1
                for runtimenode in hostnode.runtime_node:
                    CPUTimeRange[runtimenode.type].append(
                        (runtimenode.start_ns, runtimenode.end_ns)
                    )
                    self.call_times[runtimenode.type] += 1
                    for devicenode in runtimenode.device_node:
                        GPUTimeRange[devicenode.device_id][devicenode.type][
                            devicenode.stream_id
                        ].append((devicenode.start_ns, devicenode.end_ns))
                        self.call_times[devicenode.type] += 1

            for event_type, time_ranges in CPUTimeRange.items():
                time_ranges = merge_self_ranges(time_ranges, is_sorted=False)
                self.CPUTimeRange[event_type] = merge_ranges(
                    self.CPUTimeRange[event_type], time_ranges, is_sorted=True
                )
            for device_id, device_time_ranges in GPUTimeRange.items():
                for event_type, event_time_ranges in device_time_ranges.items():
                    for stream_id, time_ranges in event_time_ranges.items():
                        time_ranges = merge_self_ranges(
                            time_ranges, is_sorted=False
                        )
                        self.GPUTimeRange[device_id][event_type] = merge_ranges(
                            self.GPUTimeRange[device_id][event_type],
                            time_ranges,
                            is_sorted=True,
                        )

        for event_type, time_ranges in self.CPUTimeRange.items():
            self.CPUTimeRangeSum[event_type] = sum_ranges(time_ranges)
        for device_id, device_time_ranges in self.GPUTimeRange.items():
            for event_type, time_ranges in device_time_ranges.items():
                self.GPUTimeRangeSum[device_id][event_type] = sum_ranges(
                    time_ranges
                )

    def get_gpu_devices(self):
        return self.GPUTimeRange.keys()

    def get_gpu_range_sum(self, device_id, event_type):
        return self.GPUTimeRangeSum[device_id][event_type]

    def get_cpu_range_sum(self, event_type):
        return self.CPUTimeRangeSum[event_type]


class DistributedSummary:
    r"""
    Analysis communication and computation time range, and their overlap.
    The computation time is all kernel except kernels for communication like nccl.
    """

    def __init__(self):
        self.cpu_communication_range = []
        self.gpu_communication_range = []
        self.communication_range = []
        self.computation_range = []
        self.overlap_range = []
        self.cpu_calls = 0
        self.gpu_calls = 0

    def parse(self, nodetrees):
        '''
        Collect all communication and computation time ranges.
        '''
        thread2hostnodes = traverse_tree(nodetrees)
        for threadid, hostnodes in thread2hostnodes.items():
            for hostnode in hostnodes[1:]:  # skip root node
                # case 1: TracerEventType is Communication
                if hostnode.type == TracerEventType.Communication:
                    self.cpu_communication_range.append(
                        (hostnode.start_ns, hostnode.end_ns)
                    )
                    device_nodes = get_device_nodes(hostnode)
                    for device_node in device_nodes:
                        if device_node.type == TracerEventType.Kernel:
                            self.gpu_communication_range.append(
                                (device_node.start_ns, device_node.end_ns)
                            )

                # case 2: TracerEventType is Operator but is communication op
                elif hostnode.type == TracerEventType.Operator and any(
                    name in hostnode.name.lower()
                    for name in _CommunicationOpName
                ):
                    self.cpu_communication_range.append(
                        (hostnode.start_ns, hostnode.end_ns)
                    )
                    device_nodes = get_device_nodes(hostnode)
                    for device_node in device_nodes:
                        if device_node.type == TracerEventType.Kernel:
                            self.gpu_communication_range.append(
                                (device_node.start_ns, device_node.end_ns)
                            )

                # case 3: Others, filter kernels named with nccl
                else:
                    for runtimenode in hostnode.runtime_node:
                        for devicenode in runtimenode.device_node:
                            if devicenode.type == TracerEventType.Kernel:
                                kernel_name = devicenode.name.lower()
                                if (
                                    'nccl' in kernel_name
                                    or 'xccl' in kernel_name
                                ):
                                    self.gpu_communication_range.append(
                                        (devicenode.start_ns, devicenode.end_ns)
                                    )
                                else:
                                    self.computation_range.append(
                                        (devicenode.start_ns, devicenode.end_ns)
                                    )
        self.cpu_calls = len(set(self.cpu_communication_range))
        self.gpu_calls = len(set(self.gpu_communication_range))
        self.cpu_communication_range = merge_self_ranges(
            self.cpu_communication_range, is_sorted=False
        )
        self.gpu_communication_range = merge_self_ranges(
            self.gpu_communication_range, is_sorted=False
        )
        self.communication_range = merge_ranges(
            self.cpu_communication_range,
            self.gpu_communication_range,
            is_sorted=True,
        )
        self.computation_range = merge_self_ranges(
            self.computation_range, is_sorted=False
        )
        self.overlap_range = intersection_ranges(
            self.communication_range, self.computation_range, is_sorted=True
        )


class EventSummary:
    r"""
    Analyse operator event in profiling data, correlate with its device event.
    """

    class ItemBase:
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
            self.general_gpu_time = 0
            self.min_general_gpu_time = float('inf')
            self.max_general_gpu_time = 0
            self._flops = 0

        @property
        def flops(self):
            return self._flops

        @property
        def avg_cpu_time(self):
            return self.cpu_time / self.call

        @property
        def avg_gpu_time(self):
            return self.gpu_time / self.call

        @property
        def avg_general_gpu_time(self):
            return self.general_gpu_time / self.call

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

        def add_general_gpu_time(self, time):
            if time > self.max_general_gpu_time:
                self.max_general_gpu_time = time
            if time < self.min_general_gpu_time:
                self.min_general_gpu_time = time
            self.general_gpu_time += time

        def add_call(self):
            self.call += 1

        def add_flops(self, flops):
            self._flops += flops

        def add_item(self, node):
            raise NotImplementedError

    class DeviceItem(ItemBase):
        def add_item(self, node):
            self.call += 1
            self.add_gpu_time(node.end_ns - node.start_ns)

    class OperatorItem(ItemBase):
        def add_item(self, node):
            self.add_call()
            self.add_cpu_time(node.cpu_time)
            self.add_gpu_time(node.gpu_time)
            self.add_general_gpu_time(node.general_gpu_time)
            self.add_flops(node.flops)
            for child in node.children_node:
                if child.type != TracerEventType.Operator:
                    if child.name not in self.operator_inners:
                        self.operator_inners[
                            child.name
                        ] = EventSummary.OperatorItem(child.name)
                    self.operator_inners[child.name].add_item(child)

            for runtimenode in node.runtime_node:
                for devicenode in runtimenode.device_node:
                    name = devicenode.name
                    if name not in self.devices:
                        self.devices[name] = EventSummary.DeviceItem(name)
                    self.devices[name].add_item(devicenode)

    class ForwardItem(ItemBase):
        def add_item(self, node):
            self.add_call()
            self.add_cpu_time(node.cpu_time)
            self.add_gpu_time(node.gpu_time)
            self.add_general_gpu_time(node.general_gpu_time)
            self.add_flops(node.flops)
            for child in node.children_node:
                if child.type != TracerEventType.Operator:
                    if child.name not in self.operator_inners:
                        self.operator_inners[
                            child.name
                        ] = EventSummary.OperatorItem(child.name)
                    self.operator_inners[child.name].add_item(child)

    class GeneralItem(ItemBase):
        def add_item(self, node):
            self.add_call()
            self.add_cpu_time(node.cpu_time)
            self.add_gpu_time(node.gpu_time)
            self.add_general_gpu_time(node.general_gpu_time)

    def __init__(self):
        self.items = {}  # for operator summary
        self.thread_items = collections.defaultdict(
            dict
        )  # for operator summary
        self.userdefined_items = {}  # for userdefined summary
        self.userdefined_thread_items = collections.defaultdict(
            dict
        )  # for userdefined summary
        self.model_perspective_items = {}  # for model summary
        self.memory_manipulation_items = {}  # for memory manipulation summary
        self.kernel_items = {}  # for kernel summary

    def parse(self, nodetrees):
        r"""
        Analysis operator event in the nodetress.
        """
        node_statistic_trees, thread2host_statistic_nodes = wrap_tree(nodetrees)
        for (
            threadid,
            host_statistic_nodes,
        ) in thread2host_statistic_nodes.items():
            for host_statistic_node in host_statistic_nodes[
                1:
            ]:  # skip root node
                if host_statistic_node.type == TracerEventType.Operator:
                    self.add_operator_item(host_statistic_node)
                if (
                    host_statistic_node.type == TracerEventType.UserDefined
                    or host_statistic_node.type
                    == TracerEventType.PythonUserDefined
                ):
                    if (
                        'memcpy' in host_statistic_node.name.lower()
                        or 'memorycopy' in host_statistic_node.name.lower()
                        or 'memset' in host_statistic_node.name.lower()
                    ):
                        self.add_memory_manipulation_item(host_statistic_node)
                    else:
                        if (
                            host_statistic_node.type
                            == TracerEventType.PythonUserDefined
                        ):
                            self.add_userdefined_item(host_statistic_node)
            self.add_kernel_item(host_statistic_nodes[0])

        for threadid, root_statistic_node in node_statistic_trees.items():
            deque = collections.deque()
            deque.append(root_statistic_node)
            while deque:
                current_node = deque.popleft()
                for child in current_node.children_node:
                    if (
                        child.type == TracerEventType.Forward
                        or child.type == TracerEventType.Dataloader
                        or child.type == TracerEventType.Backward
                        or child.type == TracerEventType.Optimization
                    ):
                        self.add_model_perspective_item(
                            child
                        )  # find first model perspective node
                    else:
                        if child.type == TracerEventType.ProfileStep:
                            self.add_model_perspective_item(child)
                        deque.append(child)

    def add_forward_item(self, operator_node):
        pass

    def add_operator_item(self, operator_node):
        if operator_node.name not in self.items:
            self.items[operator_node.name] = EventSummary.OperatorItem(
                operator_node.name
            )

        self.items[operator_node.name].add_item(operator_node)

        if operator_node.name not in self.thread_items[operator_node.thread_id]:
            self.thread_items[operator_node.thread_id][
                operator_node.name
            ] = EventSummary.OperatorItem(operator_node.name)
        self.thread_items[operator_node.thread_id][operator_node.name].add_item(
            operator_node
        )

    def add_userdefined_item(self, userdefined_node):
        if userdefined_node.name not in self.userdefined_items:
            self.userdefined_items[
                userdefined_node.name
            ] = EventSummary.GeneralItem(userdefined_node.name)

        self.userdefined_items[userdefined_node.name].add_item(userdefined_node)

        if (
            userdefined_node.name
            not in self.userdefined_thread_items[userdefined_node.thread_id]
        ):
            self.userdefined_thread_items[userdefined_node.thread_id][
                userdefined_node.name
            ] = EventSummary.GeneralItem(userdefined_node.name)
        self.userdefined_thread_items[userdefined_node.thread_id][
            userdefined_node.name
        ].add_item(userdefined_node)

    def add_memory_manipulation_item(self, memory_manipulation_node):
        if memory_manipulation_node.name not in self.memory_manipulation_items:
            self.memory_manipulation_items[
                memory_manipulation_node.name
            ] = EventSummary.GeneralItem(memory_manipulation_node.name)
        self.memory_manipulation_items[memory_manipulation_node.name].add_item(
            memory_manipulation_node
        )

    def add_model_perspective_item(self, model_perspective_node):
        if model_perspective_node.type == TracerEventType.Forward:
            name = 'Forward'
        elif model_perspective_node.type == TracerEventType.Backward:
            name = 'Backward'
        elif model_perspective_node.type == TracerEventType.Optimization:
            name = 'Optimization'
        elif model_perspective_node.type == TracerEventType.Dataloader:
            name = 'Dataloader'
        elif model_perspective_node.type == TracerEventType.ProfileStep:
            name = 'ProfileStep'
        else:
            return
        if name not in self.model_perspective_items:
            self.model_perspective_items[name] = EventSummary.GeneralItem(name)
        self.model_perspective_items[name].add_item(model_perspective_node)

    def add_kernel_item(self, root_node):
        device_nodes = get_device_nodes(root_node)
        for device_node in device_nodes:
            if device_node.type == TracerEventType.Kernel:
                name = device_node.name
                if name not in self.kernel_items:
                    self.kernel_items[name] = EventSummary.DeviceItem(name)
                self.kernel_items[name].add_item(device_node)


class MemorySummary:
    r"""
    Analyse memory events in profiling data.
    """

    class MemoryItem:
        def __init__(self, event_name, place, memory_type='Allocated'):
            self.event_name = event_name
            self.place = place
            self.allocation_count = 0
            self.free_count = 0
            self.allocation_size = 0
            self.free_size = 0
            self.increase_size = 0
            self.memory_type = memory_type

        def add_memory_record(self, size, allocation_type):
            if (
                allocation_type == TracerMemEventType.Allocate
                or allocation_type == TracerMemEventType.ReservedAllocate
            ):
                self.allocation_count += 1
                self.allocation_size += size

            elif (
                allocation_type == TracerMemEventType.Free
                or allocation_type == TracerMemEventType.ReservedFree
            ):
                self.free_count += 1
                self.free_size -= size  # size is sign(-) when free.

            else:
                print("No corresponding type.")
            self.increase_size = self.allocation_size - self.free_size

    def __init__(self):
        self.allocated_items = collections.defaultdict(
            dict
        )  # for memory summary, device type: event
        self.reserved_items = collections.defaultdict(
            dict
        )  # for memory summary, device type: event
        self.peak_allocation_values = collections.defaultdict(int)
        self.peak_reserved_values = collections.defaultdict(int)

    def _analyse_node_memory(self, event_name, node):
        for memnode in node.mem_node:  # self mem node
            if (
                memnode.type == TracerMemEventType.Allocate
                or memnode.type == TracerMemEventType.Free
            ):
                if event_name not in self.allocated_items[memnode.place]:
                    self.allocated_items[memnode.place][
                        event_name
                    ] = MemorySummary.MemoryItem(
                        event_name, memnode.place, 'Allocated'
                    )
                self.allocated_items[memnode.place][
                    event_name
                ].add_memory_record(memnode.increase_bytes, memnode.type)
            elif (
                memnode.type == TracerMemEventType.ReservedAllocate
                or memnode.type == TracerMemEventType.ReservedFree
            ):
                if event_name not in self.reserved_items[memnode.place]:
                    self.reserved_items[memnode.place][
                        event_name
                    ] = MemorySummary.MemoryItem(
                        event_name, memnode.place, 'Reserved'
                    )
                self.reserved_items[memnode.place][
                    event_name
                ].add_memory_record(memnode.increase_bytes, memnode.type)
            self.peak_allocation_values[memnode.place] = max(
                self.peak_allocation_values[memnode.place],
                memnode.peak_allocated,
            )
            self.peak_reserved_values[memnode.place] = max(
                self.peak_reserved_values[memnode.place], memnode.peak_reserved
            )

    def parse(self, nodetrees):
        r"""
        Analyse memory event in the nodetress.
        """
        thread2hostnodes = traverse_tree(nodetrees)
        for threadid, host_nodes in thread2hostnodes.items():
            for host_node in host_nodes[1:]:  # skip root node
                if host_node.type == TracerEventType.OperatorInner:
                    continue
                if host_node.type == TracerEventType.Operator:
                    for child in host_node.children_node:
                        self._analyse_node_memory(host_node.name, child)
                self._analyse_node_memory(host_node.name, host_node)


class StatisticData:
    r"""
    Hold all analysed results.
    """

    def __init__(self, node_trees, extra_info):
        self.node_trees = node_trees
        self.extra_info = extra_info
        self.time_range_summary = TimeRangeSummary()
        self.event_summary = EventSummary()
        self.distributed_summary = DistributedSummary()
        self.memory_summary = MemorySummary()
        self.time_range_summary.parse(node_trees)
        self.event_summary.parse(node_trees)
        self.distributed_summary.parse(node_trees)
        self.memory_summary.parse(node_trees)


def _build_table(
    statistic_data,
    sorted_by=SortedKeys.CPUTotal,
    op_detail=True,
    thread_sep=False,
    time_unit='ms',
    row_limit=100,
    max_src_column_width=75,
    views=None,
):
    from .profiler import SummaryView

    """Prints a summary of events."""
    # format table row
    SPACING_SIZE = 2
    row_format_list = [""]
    header_sep_list = [""]
    line_length_list = [-SPACING_SIZE]

    def add_column(padding, text_dir='<'):
        row_format_list[0] += (
            '{: ' + text_dir + str(padding) + '}' + (' ' * SPACING_SIZE)
        )
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
        TracerEventType.ProfileStep
    )

    if views is None or SummaryView.DeviceView in views:
        # ----- Print Device Summary ----- #
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
        append(header_sep)
        append(row_format.format(*headers))
        append(header_sep)
        row_values = [
            'CPU(Process)',
            format_ratio(
                float(statistic_data.extra_info['Process Cpu Utilization'])
            ),
        ]
        append(row_format.format(*row_values))
        row_values = [
            'CPU(System)',
            format_ratio(
                float(statistic_data.extra_info['System Cpu Utilization'])
            ),
        ]
        append(row_format.format(*row_values))
        for gpu_name in statistic_data.time_range_summary.get_gpu_devices():
            gpu_time = float(
                statistic_data.time_range_summary.get_gpu_range_sum(
                    gpu_name, TracerEventType.Kernel
                )
            )
            utilization = gpu_time / total_time
            row_values = [f'GPU{gpu_name}', format_ratio(utilization)]
            append(row_format.format(*row_values))

        append(header_sep)
        append(
            "Note:\nCPU(Process) Utilization = Current process CPU time over all cpu cores / elapsed time, so max utilization can be reached 100% * number of cpu cores.\n"
            "CPU(System) Utilization = All processes CPU time over all cpu cores(busy time) / (busy time + idle time).\n"
            "GPU Utilization = Current process GPU time / elapsed time."
        )
        append('-' * line_length)
        append('')
        append('')

        if total_time == 0:
            return ''.join(result)

    if views is None or SummaryView.OverView in views:
        # ----- Print Overview Summary ----- #
        headers = ['Event Type', 'Calls', 'CPU Time', 'Ratio (%)']
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
        append(f'Time unit: {time_unit}')
        append(header_sep)
        append(row_format.format(*headers))
        append(header_sep)
        cpu_type_time = collections.defaultdict(int)
        gpu_type_time = collections.defaultdict(int)
        cpu_call_times = collections.defaultdict(int)
        gpu_call_times = collections.defaultdict(int)
        cpu_call_times.update(statistic_data.time_range_summary.call_times)
        gpu_call_times.update(statistic_data.time_range_summary.call_times)

        for (
            event_type,
            value,
        ) in statistic_data.time_range_summary.CPUTimeRangeSum.items():
            if event_type != TracerEventType.Communication:
                cpu_type_time[event_type] = value
        if statistic_data.distributed_summary.cpu_communication_range:
            cpu_type_time[TracerEventType.Communication] = sum_ranges(
                statistic_data.distributed_summary.cpu_communication_range
            )
            cpu_call_times[
                TracerEventType.Communication
            ] = statistic_data.distributed_summary.cpu_calls

        for event_type in [
            TracerEventType.Dataloader,
            TracerEventType.Forward,
            TracerEventType.Backward,
            TracerEventType.Optimization,
        ]:
            event_type_name = str(event_type).split('.')[1]
            if (
                event_type in cpu_call_times
                and event_type_name
                in statistic_data.event_summary.model_perspective_items
            ):
                cpu_call_times[
                    event_type
                ] = statistic_data.event_summary.model_perspective_items[
                    event_type_name
                ].call
                cpu_type_time[
                    event_type
                ] = statistic_data.event_summary.model_perspective_items[
                    event_type_name
                ].cpu_time

        gpu_time_range = collections.defaultdict(list)
        for (
            device_id,
            device_time_ranges,
        ) in statistic_data.time_range_summary.GPUTimeRange.items():
            for event_type, time_range in device_time_ranges.items():
                gpu_time_range[event_type] = merge_ranges(
                    gpu_time_range[event_type], time_range, is_sorted=True
                )
        for event_type, time_range in gpu_time_range.items():
            gpu_type_time[event_type] = sum_ranges(time_range)
        if statistic_data.distributed_summary.gpu_communication_range:
            gpu_type_time[TracerEventType.Communication] = sum_ranges(
                statistic_data.distributed_summary.gpu_communication_range
            )
            gpu_call_times[
                TracerEventType.Communication
            ] = statistic_data.distributed_summary.gpu_calls

        sorted_items = sorted(
            cpu_type_time.items(), key=lambda x: x[1], reverse=True
        )
        event_type, time = sorted_items[0]
        row_values = [
            '{}'.format(str(event_type).split('.')[1]),
            cpu_call_times[event_type],
            format_time(time, unit=time_unit),
            format_ratio(float(time) / total_time),
        ]
        append(row_format.format(*row_values))
        for event_type, time in sorted_items[1:]:
            row_values = [
                '  {}'.format(str(event_type).split('.')[1]),
                cpu_call_times[event_type],
                format_time(time, unit=time_unit),
                format_ratio(float(time) / total_time),
            ]
            append(row_format.format(*row_values))
        append(header_sep)
        headers = ['', 'Calls', 'GPU Time', 'Ratio (%)']
        append(row_format.format(*headers))
        append(header_sep)
        for event_type, time in gpu_type_time.items():
            row_values = [
                '  {}'.format(str(event_type).split('.')[1]),
                gpu_call_times[event_type],
                format_time(time, unit=time_unit),
                format_ratio(float(time) / total_time),
            ]
            append(row_format.format(*row_values))

        append(header_sep)
        append(
            "Note:\nIn this table, We sum up all collected events in terms of event type.\n"
            "The time of events collected on host are presented as CPU Time, and as GPU Time if on device.\n"
            "Events with different types may overlap or inclusion, e.g. Operator includes OperatorInner, so the sum of ratios is not 100%.\n"
            "The time of events in the same type with overlap will not calculate twice, and all time is summed after merged.\n"
            "Example:\n"
            "Thread 1:\n"
            "  Operator: |___________|     |__________|\n"
            "Thread 2:\n"
            "  Operator:   |____________|     |___|\n"
            "After merged:\n"
            "  Result:   |______________|  |__________|\n"
        )
        append('-' * line_length)
        append('')
        append('')

    if views is None or SummaryView.ModelView in views:
        # ----- Print Model Summary Report ----- #
        model_perspective_items = (
            statistic_data.event_summary.model_perspective_items
        )
        if len(model_perspective_items) > 1:
            all_row_values = []
            accumulation_time = 0
            gpu_accumulation_time = 0
            gpu_total_time = (
                statistic_data.event_summary.model_perspective_items[
                    'ProfileStep'
                ].gpu_time
            )
            for name in [
                'ProfileStep',
                'Dataloader',
                'Forward',
                'Backward',
                'Optimization',
            ]:
                if name in model_perspective_items:
                    item = model_perspective_items[name]
                    if gpu_total_time == 0:
                        gpu_ratio = 0
                    else:
                        gpu_ratio = float(item.gpu_time) / gpu_total_time
                    name = f'{name}' if 'ProfileStep' in name else f'  {name}'
                    row_values = [
                        f'{name}',
                        item.call,
                        '{} / {} / {} / {} / {}'.format(
                            format_time(item.cpu_time, unit=time_unit),
                            format_time(item.avg_cpu_time, unit=time_unit),
                            format_time(item.max_cpu_time, unit=time_unit),
                            format_time(item.min_cpu_time, unit=time_unit),
                            format_ratio(float(item.cpu_time) / total_time),
                        ),
                        '{} / {} / {} / {} / {}'.format(
                            format_time(item.gpu_time, unit=time_unit),
                            format_time(item.avg_gpu_time, unit=time_unit),
                            format_time(item.max_gpu_time, unit=time_unit),
                            format_time(item.min_gpu_time, unit=time_unit),
                            format_ratio(gpu_ratio),
                        ),
                    ]
                    all_row_values.append(row_values)
                    if 'ProfileStep' not in name:
                        accumulation_time += item.cpu_time
                        gpu_accumulation_time += item.gpu_time

            other_time = total_time - accumulation_time
            other_gpu_time = gpu_total_time - gpu_accumulation_time
            if gpu_total_time == 0:
                gpu_ratio = 0
            else:
                gpu_ratio = float(other_gpu_time) / gpu_total_time
            row_values = [
                '  Others',
                '-',
                '{} / - / - / - / {}'.format(
                    format_time(other_time, unit=time_unit),
                    format_ratio(float(other_time) / total_time),
                ),
                '{} / - / - / - / {}'.format(
                    format_time(other_gpu_time, unit=time_unit),
                    format_ratio(gpu_ratio),
                ),
            ]
            all_row_values.append(row_values)
            # Calculate the column width
            calltime_width = 6
            cpu_data_description_width = 40
            gpu_data_description_width = 40
            for row_values in all_row_values:
                if (
                    isinstance(row_values[1], int)
                    and len(str(row_values[1])) > calltime_width
                ):
                    calltime_width = len(str(row_values[1]))
                if len(row_values[2]) > cpu_data_description_width:
                    cpu_data_description_width = len(row_values[2])
                if len(row_values[3]) > gpu_data_description_width:
                    gpu_data_description_width = len(row_values[3])
            headers = [
                'Name',
                'Calls',
                'CPU Total / Avg / Max / Min / Ratio(%)',
                'GPU Total / Avg / Max / Min / Ratio(%)',
            ]
            row_format_list = [""]
            header_sep_list = [""]
            line_length_list = [-SPACING_SIZE]
            name_column_width = 15
            add_column(name_column_width)
            add_column(calltime_width)
            add_column(cpu_data_description_width)
            add_column(gpu_data_description_width)

            row_format = row_format_list[0]
            header_sep = header_sep_list[0]
            line_length = line_length_list[0]

            # construct table string
            append(add_title(line_length, "Model Summary"))
            append(f'Time unit: {time_unit}')
            append(header_sep)
            append(row_format.format(*headers))
            append(header_sep)
            for row_values in all_row_values:
                append(row_format.format(*row_values))
            append(header_sep)
            append(
                "Note:\nIn this table, GPU time is the sum of all device(GPU) events called in the phase.\n"
                "Unlike overview summary, if two device(GPU) events execute on different streams with overlap time, we sum them directly here.\n"
            )
            append('-' * line_length)
            append('')
            append('')

    if views is None or SummaryView.DistributedView in views:
        # ----- Print Distribution Summary Report ----- #
        if statistic_data.distributed_summary.communication_range:
            headers = [
                'Name',
                'Total Time',
                'Ratio (%)',
            ]
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
            append(add_title(line_length, "Distribution Summary"))
            append(f'Time unit: {time_unit}')
            append(header_sep)
            append(row_format.format(*headers))
            append(header_sep)
            communication_time = sum_ranges(
                statistic_data.distributed_summary.communication_range
            )
            computation_time = sum_ranges(
                statistic_data.distributed_summary.computation_range
            )
            overlap_time = sum_ranges(
                statistic_data.distributed_summary.overlap_range
            )
            row_values = [
                'ProfileStep',
                format_time(total_time, unit=time_unit),
                format_ratio(float(total_time) / total_time),
            ]
            append(row_format.format(*row_values))
            row_values = [
                '  Communication',
                format_time(communication_time, unit=time_unit),
                format_ratio(float(communication_time) / total_time),
            ]
            append(row_format.format(*row_values))

            row_values = [
                '  Computation',
                format_time(computation_time, unit=time_unit),
                format_ratio(float(computation_time) / total_time),
            ]
            append(row_format.format(*row_values))

            row_values = [
                '  Overlap',
                format_time(overlap_time, unit=time_unit),
                format_ratio(float(overlap_time) / total_time),
            ]
            append(row_format.format(*row_values))
            append(header_sep)
            append(
                "Note:\nCommunication time: Communication Event time, Communication Op time and its kernel time on gpu.\n"
                "Computation time: Kernel time, except kernels belong to communication(nccl kernels).\n"
                "Overlap time: Communication time intersects with computation time.\n"
                "Example:\n"
                "Communication:\n"
                "  CPU:              |_________________|\n"
                "  GPU:                                  |______________|\n"
                "  Total:            |_________________| |______________|\n"
                "Computation time(Kernel):\n"
                "  GPU:         |________________|\n"
                "Overlap time:       |___________|\n"
            )
            append('-' * line_length)
            append('')
            append('')

    if views is None or SummaryView.OperatorView in views:
        # ----- Print Operator Summary Report ----- #
        if statistic_data.event_summary.items:
            all_row_values = []
            name_column_width = 52
            if thread_sep:
                thread_items = statistic_data.event_summary.thread_items
            else:
                thread_items = {
                    'All threads merged': statistic_data.event_summary.items
                }
            for thread_id, items in thread_items.items():
                all_row_values.append(f"Thread: {thread_id}")
                if sorted_by == SortedKeys.CPUTotal:
                    sorted_items = sorted(
                        items.items(), key=lambda x: x[1].cpu_time, reverse=True
                    )
                elif sorted_by == SortedKeys.CPUAvg:
                    sorted_items = sorted(
                        items.items(),
                        key=lambda x: x[1].avg_cpu_time,
                        reverse=True,
                    )
                elif sorted_by == SortedKeys.CPUMax:
                    sorted_items = sorted(
                        items.items(),
                        key=lambda x: x[1].max_cpu_time,
                        reverse=True,
                    )
                elif sorted_by == SortedKeys.CPUMin:
                    sorted_items = sorted(
                        items.items(), key=lambda x: x[1].min_cpu_time
                    )
                elif sorted_by == SortedKeys.GPUTotal:
                    sorted_items = sorted(
                        items.items(),
                        key=lambda x: x[1].general_gpu_time,
                        reverse=True,
                    )
                elif sorted_by == SortedKeys.GPUAvg:
                    sorted_items = sorted(
                        items.items(),
                        key=lambda x: x[1].avg_general_gpu_time,
                        reverse=True,
                    )
                elif sorted_by == SortedKeys.GPUMax:
                    sorted_items = sorted(
                        items.items(),
                        key=lambda x: x[1].max_general_gpu_time,
                        reverse=True,
                    )
                elif sorted_by == SortedKeys.GPUMin:
                    sorted_items = sorted(
                        items.items(), key=lambda x: x[1].min_general_gpu_time
                    )
                total_op_cpu_time = 0
                total_op_gpu_time = 0

                for name, item in sorted_items:
                    total_op_cpu_time += item.cpu_time
                    total_op_gpu_time += item.general_gpu_time

                for name, item in sorted_items:
                    if total_op_cpu_time == 0:
                        cpu_ratio = 0
                    else:
                        cpu_ratio = float(item.cpu_time) / total_op_cpu_time
                    if total_op_gpu_time == 0:
                        gpu_ratio = 0
                    else:
                        gpu_ratio = (
                            float(item.general_gpu_time) / total_op_gpu_time
                        )
                    row_values = [
                        name,
                        item.call,
                        '{} / {} / {} / {} / {}'.format(
                            format_time(item.cpu_time, unit=time_unit),
                            format_time(item.avg_cpu_time, unit=time_unit),
                            format_time(item.max_cpu_time, unit=time_unit),
                            format_time(item.min_cpu_time, unit=time_unit),
                            format_ratio(cpu_ratio),
                        ),
                        '{} / {} / {} / {} / {}'.format(
                            format_time(item.general_gpu_time, unit=time_unit),
                            format_time(
                                item.avg_general_gpu_time, unit=time_unit
                            ),
                            format_time(
                                item.max_general_gpu_time, unit=time_unit
                            ),
                            format_time(
                                item.min_general_gpu_time, unit=time_unit
                            ),
                            format_ratio(gpu_ratio),
                        ),
                        item.flops,
                    ]
                    all_row_values.append(row_values)
                    if op_detail:
                        for (
                            innerop_name,
                            innerop_node,
                        ) in item.operator_inners.items():
                            if item.cpu_time == 0:
                                cpu_ratio = 0
                            else:
                                cpu_ratio = (
                                    float(innerop_node.cpu_time) / item.cpu_time
                                )
                            if item.general_gpu_time == 0:
                                gpu_ratio = 0
                            else:
                                gpu_ratio = (
                                    float(innerop_node.general_gpu_time)
                                    / item.general_gpu_time
                                )
                            if len(innerop_name) + 2 > name_column_width:
                                innerop_name = innerop_name[
                                    : name_column_width - 5
                                ]
                                innerop_name += "..."
                            row_values = [
                                f'  {innerop_name}',
                                innerop_node.call,
                                '{} / {} / {} / {} / {}'.format(
                                    format_time(
                                        innerop_node.cpu_time, unit=time_unit
                                    ),
                                    format_time(
                                        innerop_node.avg_cpu_time,
                                        unit=time_unit,
                                    ),
                                    format_time(
                                        innerop_node.max_cpu_time,
                                        unit=time_unit,
                                    ),
                                    format_time(
                                        innerop_node.min_cpu_time,
                                        unit=time_unit,
                                    ),
                                    format_ratio(cpu_ratio),
                                ),
                                '{} / {} / {} / {} / {}'.format(
                                    format_time(
                                        innerop_node.general_gpu_time,
                                        unit=time_unit,
                                    ),
                                    format_time(
                                        innerop_node.avg_general_gpu_time,
                                        unit=time_unit,
                                    ),
                                    format_time(
                                        innerop_node.max_general_gpu_time,
                                        unit=time_unit,
                                    ),
                                    format_time(
                                        innerop_node.min_general_gpu_time,
                                        unit=time_unit,
                                    ),
                                    format_ratio(gpu_ratio),
                                ),
                                '-',
                            ]
                            all_row_values.append(row_values)
                            for (
                                device_node_name,
                                device_node,
                            ) in innerop_node.devices.items():
                                if innerop_node.general_gpu_time == 0:
                                    gpu_ratio = 0
                                else:
                                    gpu_ratio = (
                                        float(device_node.gpu_time)
                                        / innerop_node.general_gpu_time
                                    )
                                if (
                                    len(device_node_name) + 4
                                    > name_column_width
                                ):
                                    device_node_name = device_node_name[
                                        : name_column_width - 7
                                    ]
                                    device_node_name += "..."
                                row_values = [
                                    f'    {device_node_name}',
                                    device_node.call,
                                    '- / - / - / - / -',
                                    '{} / {} / {} / {} / {}'.format(
                                        format_time(
                                            device_node.gpu_time, unit=time_unit
                                        ),
                                        format_time(
                                            device_node.avg_gpu_time,
                                            unit=time_unit,
                                        ),
                                        format_time(
                                            device_node.max_gpu_time,
                                            unit=time_unit,
                                        ),
                                        format_time(
                                            device_node.min_gpu_time,
                                            unit=time_unit,
                                        ),
                                        format_ratio(gpu_ratio),
                                    ),
                                    '-',
                                ]
                                all_row_values.append(row_values)
                        for (
                            device_node_name,
                            device_node,
                        ) in item.devices.items():
                            if item.general_gpu_time == 0:
                                gpu_ratio = 0
                            else:
                                gpu_ratio = (
                                    float(device_node.gpu_time)
                                    / item.general_gpu_time
                                )
                            if len(device_node_name) + 2 > name_column_width:
                                device_node_name = device_node_name[
                                    : name_column_width - 5
                                ]
                                device_node_name += "..."
                            row_values = [
                                f'  {device_node_name}',
                                device_node.call,
                                '- / - / - / - / -',
                                '{} / {} / {} / {} / {}'.format(
                                    format_time(
                                        device_node.gpu_time, unit=time_unit
                                    ),
                                    format_time(
                                        device_node.avg_gpu_time, unit=time_unit
                                    ),
                                    format_time(
                                        device_node.max_gpu_time, unit=time_unit
                                    ),
                                    format_time(
                                        device_node.min_gpu_time, unit=time_unit
                                    ),
                                    format_ratio(gpu_ratio),
                                ),
                                '-',
                            ]
                            all_row_values.append(row_values)
            # Calculate the column width
            calltime_width = 6
            cpu_data_description_width = 40
            gpu_data_description_width = 40
            flops_width = 10
            for row_values in all_row_values:
                if isinstance(row_values, str):
                    continue
                if (
                    isinstance(row_values[1], int)
                    and len(str(row_values[1])) > calltime_width
                ):
                    calltime_width = len(str(row_values[1]))
                if len(row_values[2]) > cpu_data_description_width:
                    cpu_data_description_width = len(row_values[2])
                if len(row_values[3]) > gpu_data_description_width:
                    gpu_data_description_width = len(row_values[3])
            headers = [
                'Name',
                'Calls',
                'CPU Total / Avg / Max / Min / Ratio(%)',
                'GPU Total / Avg / Max / Min / Ratio(%)',
                'FLOPs',
            ]
            row_format_list = [""]
            header_sep_list = [""]
            line_length_list = [-SPACING_SIZE]
            add_column(name_column_width)
            add_column(calltime_width)
            add_column(cpu_data_description_width)
            add_column(gpu_data_description_width)
            add_column(flops_width)

            row_format = row_format_list[0]
            header_sep = header_sep_list[0]
            line_length = line_length_list[0]

            # construct table string
            append(add_title(line_length, "Operator Summary"))
            append(f'Time unit: {time_unit}')
            append(header_sep)
            append(row_format.format(*headers))
            append(header_sep)
            for row_values in all_row_values:
                if isinstance(row_values, str):
                    append(add_title(line_length, row_values))
                else:
                    append(row_format.format(*row_values))
            append(header_sep)
            append('')
            append('')

    if views is None or SummaryView.KernelView in views:
        # ----- Print Kernel Summary Report ----- #
        if statistic_data.event_summary.kernel_items:
            all_row_values = []
            kernel_items = statistic_data.event_summary.kernel_items
            if sorted_by == SortedKeys.GPUAvg:
                sorted_items = sorted(
                    kernel_items.items(),
                    key=lambda x: x[1].avg_gpu_time,
                    reverse=True,
                )
            elif sorted_by == SortedKeys.GPUMax:
                sorted_items = sorted(
                    kernel_items.items(),
                    key=lambda x: x[1].max_gpu_time,
                    reverse=True,
                )
            elif sorted_by == SortedKeys.GPUMin:
                sorted_items = sorted(
                    kernel_items.items(), key=lambda x: x[1].min_gpu_time
                )
            else:
                sorted_items = sorted(
                    kernel_items.items(),
                    key=lambda x: x[1].gpu_time,
                    reverse=True,
                )

            total_kernel_gpu_time = 0
            for name, item in sorted_items:
                total_kernel_gpu_time += item.gpu_time
            for name, item in sorted_items:
                if total_kernel_gpu_time == 0:
                    gpu_ratio = 0
                else:
                    gpu_ratio = float(item.gpu_time) / total_kernel_gpu_time
                row_values = [
                    name,
                    item.call,
                    '{} / {} / {} / {} / {}'.format(
                        format_time(item.gpu_time, unit=time_unit),
                        format_time(item.avg_gpu_time, unit=time_unit),
                        format_time(item.max_gpu_time, unit=time_unit),
                        format_time(item.min_gpu_time, unit=time_unit),
                        format_ratio(gpu_ratio),
                    ),
                ]
                all_row_values.append(row_values)

            headers = [
                'Name',
                'Calls',
                'GPU Total / Avg / Max / Min / Ratio(%)',
            ]
            # Calculate the column width
            name_column_width = 90
            calltime_width = 6
            gpu_data_description_width = 40
            for row_values in all_row_values:
                if (
                    isinstance(row_values[1], int)
                    and len(str(row_values[1])) > calltime_width
                ):
                    calltime_width = len(str(row_values[1]))
                if len(row_values[2]) > gpu_data_description_width:
                    gpu_data_description_width = len(row_values[2])

            row_format_list = [""]
            header_sep_list = [""]
            line_length_list = [-SPACING_SIZE]
            add_column(name_column_width)
            add_column(calltime_width)
            add_column(gpu_data_description_width)

            row_format = row_format_list[0]
            header_sep = header_sep_list[0]
            line_length = line_length_list[0]

            # construct table string
            append(add_title(line_length, "Kernel Summary"))
            append(f'Time unit: {time_unit}')
            append(header_sep)
            append(row_format.format(*headers))
            append(header_sep)
            kernel_name_pattern = re.compile(r'(.+?)(<.*>)(\(.*\))')
            for row_values in all_row_values:
                match = kernel_name_pattern.match(row_values[0])
                if match:
                    name = match.group(1) + match.group(2)
                else:
                    name = row_values[0]
                if len(name) > name_column_width:
                    row_values[0] = name[: name_column_width - 3] + '...'
                else:
                    row_values[0] = name
                append(row_format.format(*row_values))
            append(header_sep)
            append('')
            append('')

    if views is None or SummaryView.MemoryManipulationView in views:
        # ----- Print Memory Manipulation Summary Report ----- #
        if statistic_data.event_summary.memory_manipulation_items:
            all_row_values = []
            memory_manipulation_items = (
                statistic_data.event_summary.memory_manipulation_items
            )
            gpu_total_time = (
                statistic_data.event_summary.model_perspective_items[
                    'ProfileStep'
                ].general_gpu_time
            )
            for name, item in memory_manipulation_items.items():
                if gpu_total_time == 0:
                    gpu_ratio = 0
                else:
                    gpu_ratio = float(item.general_gpu_time) / gpu_total_time
                row_values = [
                    name,
                    item.call,
                    '{} / {} / {} / {} / {}'.format(
                        format_time(item.cpu_time, unit=time_unit),
                        format_time(item.avg_cpu_time, unit=time_unit),
                        format_time(item.max_cpu_time, unit=time_unit),
                        format_time(item.min_cpu_time, unit=time_unit),
                        format_ratio(float(item.cpu_time) / total_time),
                    ),
                    '{} / {} / {} / {} / {}'.format(
                        format_time(item.general_gpu_time, unit=time_unit),
                        format_time(item.avg_general_gpu_time, unit=time_unit),
                        format_time(item.max_general_gpu_time, unit=time_unit),
                        format_time(item.min_general_gpu_time, unit=time_unit),
                        format_ratio(gpu_ratio),
                    ),
                ]
                all_row_values.append(row_values)

            headers = [
                'Name',
                'Calls',
                'CPU Total / Avg / Max / Min / Ratio(%)',
                'GPU Total / Avg / Max / Min / Ratio(%)',
            ]
            # Calculate the column width
            name_column_width = 0
            calltime_width = 6
            cpu_data_description_width = 40
            gpu_data_description_width = 40
            for row_values in all_row_values:
                if len(row_values[0]) > name_column_width:
                    name_column_width = len(row_values[0])
                if (
                    isinstance(row_values[1], int)
                    and len(str(row_values[1])) > calltime_width
                ):
                    calltime_width = len(str(row_values[1]))
                if len(row_values[2]) > cpu_data_description_width:
                    cpu_data_description_width = len(row_values[2])
                if len(row_values[3]) > gpu_data_description_width:
                    gpu_data_description_width = len(row_values[3])

            row_format_list = [""]
            header_sep_list = [""]
            line_length_list = [-SPACING_SIZE]
            add_column(name_column_width)
            add_column(calltime_width)
            add_column(cpu_data_description_width)
            add_column(gpu_data_description_width)

            row_format = row_format_list[0]
            header_sep = header_sep_list[0]
            line_length = line_length_list[0]

            # construct table string
            append(add_title(line_length, "Memory Manipulation Summary"))
            append(f'Time unit: {time_unit}')
            append(header_sep)
            append(row_format.format(*headers))
            append(header_sep)
            for row_values in all_row_values:
                append(row_format.format(*row_values))
            append(header_sep)
            append('')
            append('')

    if views is None or SummaryView.UDFView in views:
        # ----- Print UserDefined Summary Report ----- #
        if statistic_data.event_summary.userdefined_items:
            all_row_values = []
            gpu_total_time = (
                statistic_data.event_summary.model_perspective_items[
                    'ProfileStep'
                ].general_gpu_time
            )
            if thread_sep:
                userdefined_thread_items = (
                    statistic_data.event_summary.userdefined_thread_items
                )
            else:
                userdefined_thread_items = {
                    'All threads merged': statistic_data.event_summary.userdefined_items
                }
            for thread_id, items in userdefined_thread_items.items():
                all_row_values.append(f"Thread: {thread_id}")
                if sorted_by == SortedKeys.CPUTotal:
                    sorted_items = sorted(
                        items.items(), key=lambda x: x[1].cpu_time, reverse=True
                    )
                elif sorted_by == SortedKeys.CPUAvg:
                    sorted_items = sorted(
                        items.items(),
                        key=lambda x: x[1].avg_cpu_time,
                        reverse=True,
                    )
                elif sorted_by == SortedKeys.CPUMax:
                    sorted_items = sorted(
                        items.items(),
                        key=lambda x: x[1].max_cpu_time,
                        reverse=True,
                    )
                elif sorted_by == SortedKeys.CPUMin:
                    sorted_items = sorted(
                        items.items(), key=lambda x: x[1].min_cpu_time
                    )
                elif sorted_by == SortedKeys.GPUTotal:
                    sorted_items = sorted(
                        items.items(),
                        key=lambda x: x[1].general_gpu_time,
                        reverse=True,
                    )
                elif sorted_by == SortedKeys.GPUAvg:
                    sorted_items = sorted(
                        items.items(),
                        key=lambda x: x[1].avg_general_gpu_time,
                        reverse=True,
                    )
                elif sorted_by == SortedKeys.GPUMax:
                    sorted_items = sorted(
                        items.items(),
                        key=lambda x: x[1].max_general_gpu_time,
                        reverse=True,
                    )
                elif sorted_by == SortedKeys.GPUMin:
                    sorted_items = sorted(
                        items.items(), key=lambda x: x[1].min_general_gpu_time
                    )

                for name, item in sorted_items:
                    if gpu_total_time == 0:
                        gpu_ratio = 0
                    else:
                        gpu_ratio = (
                            float(item.general_gpu_time) / gpu_total_time
                        )
                    row_values = [
                        name,
                        item.call,
                        '{} / {} / {} / {} / {}'.format(
                            format_time(item.cpu_time, unit=time_unit),
                            format_time(item.avg_cpu_time, unit=time_unit),
                            format_time(item.max_cpu_time, unit=time_unit),
                            format_time(item.min_cpu_time, unit=time_unit),
                            format_ratio(float(item.cpu_time) / total_time),
                        ),
                        '{} / {} / {} / {} / {}'.format(
                            format_time(item.general_gpu_time, unit=time_unit),
                            format_time(
                                item.avg_general_gpu_time, unit=time_unit
                            ),
                            format_time(
                                item.max_general_gpu_time, unit=time_unit
                            ),
                            format_time(
                                item.min_general_gpu_time, unit=time_unit
                            ),
                            format_ratio(gpu_ratio),
                        ),
                    ]
                    all_row_values.append(row_values)

            # Calculate the column width
            name_column_width = 0
            calltime_width = 6
            cpu_data_description_width = 40
            gpu_data_description_width = 40
            for row_values in all_row_values:
                if isinstance(row_values, str):
                    continue
                if len(row_values[0]) > name_column_width:
                    name_column_width = len(row_values[0])
                if (
                    isinstance(row_values[1], int)
                    and len(str(row_values[1])) > calltime_width
                ):
                    calltime_width = len(str(row_values[1]))
                if len(row_values[2]) > cpu_data_description_width:
                    cpu_data_description_width = len(row_values[2])
                if len(row_values[3]) > gpu_data_description_width:
                    gpu_data_description_width = len(row_values[3])

            headers = [
                'Name',
                'Calls',
                'CPU Total / Avg / Max / Min / Ratio(%)',
                'GPU Total / Avg / Max / Min / Ratio(%)',
            ]
            row_format_list = [""]
            header_sep_list = [""]
            line_length_list = [-SPACING_SIZE]

            add_column(name_column_width)
            add_column(calltime_width)
            add_column(cpu_data_description_width)
            add_column(gpu_data_description_width)

            row_format = row_format_list[0]
            header_sep = header_sep_list[0]
            line_length = line_length_list[0]

            # construct table string
            append(add_title(line_length, "UserDefined Summary"))
            append(f'Time unit: {time_unit}')
            append(header_sep)
            append(row_format.format(*headers))
            append(header_sep)
            for row_values in all_row_values:
                if isinstance(row_values, str):
                    append(add_title(line_length, row_values))
                else:
                    append(row_format.format(*row_values))
            append('')
            append('')

    if views is None or SummaryView.MemoryView in views:
        # ----- Print Memory Summary Report ----- #
        if (
            statistic_data.memory_summary.allocated_items
            or statistic_data.memory_summary.reserved_items
        ):
            for (
                device_type,
                memory_events,
            ) in statistic_data.memory_summary.allocated_items.items():
                all_row_values = []
                sorted_items = sorted(
                    memory_events.items(),
                    key=lambda x: x[1].increase_size,
                    reverse=True,
                )

                for event_name, item in sorted_items:
                    row_values = [
                        event_name,
                        item.memory_type,
                        item.allocation_count,
                        item.free_count,
                        item.allocation_size,
                        item.free_size,
                        item.increase_size,
                    ]
                    all_row_values.append(row_values)

                sorted_reserved_items = sorted(
                    statistic_data.memory_summary.reserved_items[
                        device_type
                    ].items(),
                    key=lambda x: x[1].increase_size,
                    reverse=True,
                )
                for event_name, item in sorted_reserved_items:
                    row_values = [
                        event_name,
                        item.memory_type,
                        item.allocation_count,
                        item.free_count,
                        item.allocation_size,
                        item.free_size,
                        item.increase_size,
                    ]
                    all_row_values.append(row_values)

                # Calculate the column width
                headers = [
                    'Name',
                    'Type',
                    'Allocation Count',
                    'Free Count',
                    'Allocation Size',
                    'Free Size',
                    'Increased Size',
                ]
                row_format_list = [""]
                header_sep_list = [""]
                line_length_list = [-SPACING_SIZE]
                name_column_width = 50
                number_column_width = 15
                add_column(name_column_width)
                add_column(12)
                add_column(number_column_width)
                add_column(number_column_width)
                add_column(number_column_width)
                add_column(number_column_width)
                add_column(number_column_width)

                row_format = row_format_list[0]
                header_sep = header_sep_list[0]
                line_length = line_length_list[0]

                # construct table string
                append(
                    add_title(line_length, f"Memory Summary - {device_type}")
                )
                append(
                    'Peak Allocated Memory: {}'.format(
                        statistic_data.memory_summary.peak_allocation_values[
                            device_type
                        ]
                    )
                )
                append(
                    'Peak Reserved Memory: {}'.format(
                        statistic_data.memory_summary.peak_reserved_values[
                            device_type
                        ]
                    )
                )
                append(header_sep)
                append(row_format.format(*headers))
                append(header_sep)
                for row_values in all_row_values:
                    if isinstance(row_values, str):
                        append(add_title(line_length, row_values))
                    else:
                        append(row_format.format(*row_values))
                append('')
                append('')

    return ''.join(result)
