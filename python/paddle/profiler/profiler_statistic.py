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

from .statistic_helper import *

_AllTracerEventType = [
    TracerEventType.Operator, TracerEventType.Dataloader,
    TracerEventType.ProfileStep, TracerEventType.CudaRuntime,
    TracerEventType.Kernel, TracerEventType.Memcpy, TracerEventType.Memset,
    TracerEventType.UserDefined, TracerEventType.OperatorInner,
    TracerEventType.Forward, TracerEventType.Backward,
    TracerEventType.Optimization, TracerEventType.Communication,
    TracerEventType.PythonOp
]

_CommunicationOpName = ['reduce', 'broadcast', 'rpc']


class SortedKeys(Enum):
    '''
  Sorted keys for printing op summary table.
  '''
    OpTotal = 0
    OpAvg = 1
    OpMax = 2
    OpMin = 3
    KernelTotal = 4
    KernelAvg = 5
    KernelMax = 6
    KernelMin = 7


class TimeRangeSummary:
    '''
    Analyse time ranges for each TracerEventType, and summarize the time.
    '''

    def __init__(self):
        self.CPUTimeRange = collections.defaultdict(list)
        self.GPUTimeRange = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )  # GPU events should be divided into different devices
        self.CPUTimeRangeSum = collections.defaultdict(int)
        self.GPUTimeRangeSum = collections.defaultdict(
            lambda: collections.defaultdict(int))

    def parse(self, nodetrees):
        '''
    Analysis node trees in profiler result, and get time range for different tracer event type
    '''
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


class OperatorSummary:
    '''
    Analyse operator event in profiling data, correlate with its device event.
    '''

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
                                devicenode.name] = OperatorSummary.DeviceItem(
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
                        self.devices[
                            devicenode.name] = OperatorSummary.DeviceItem(
                                devicenode.name)
                        self.devices[devicenode.name].add_item(devicenode)

    def __init__(self):
        self.items = {}
        self.thread_items = collections.defaultdict(dict)

    def parse(self, nodetrees):
        '''
    Analysis operator event in the nodetress.
    '''
        thread2hostnodes = traverse_tree(nodetrees)
        for threadid, hostnodes in thread2hostnodes.items():
            for hostnode in hostnodes:
                if hostnode.type == TracerEventType.Operator:
                    self.add_operator_item(hostnode)

    def add_operator_item(self, operator_node):
        if operator_node.name not in self.items:
            self.items[operator_node.name] = OperatorSummary.OperatorItem(
                operator_node.name)

        self.items[operator_node.name].add_item(operator_node)

        if operator_node.name not in self.thread_items[operator_node.thread_id]:
            self.thread_items[operator_node.thread_id][
                operator_node.name] = OperatorSummary.OperatorItem(
                    operator_node.name)
        self.thread_items[operator_node.thread_id][operator_node.name].add_item(
            operator_node)


class StatisticData:
    '''
    Hold all analysed results.
    '''

    def __init__(self, node_trees, extra_info):
        self.node_trees = node_trees
        self.extra_info = extra_info
        self.time_range_summary = TimeRangeSummary()
        self.operator_summary = OperatorSummary()
        self.time_range_summary.parse(node_trees)
        self.operator_summary.parse(node_trees)


def _build_table(statistic_data,
                 sorted_by=SortedKeys.OpTotal,
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

    total_time = statistic_data.time_range_summary.get_cpu_range_sum(
        TracerEventType.ProfileStep)
    ###### Print Device Summary ######
    headers = [
        'Device',
        'Utilization (%)',
    ]
    device_names = ['CPU']
    device_names.extend([
        'GPU{}'.format(key)
        for key in statistic_data.time_range_summary.GPUTimeRange.keys()
    ])
    MAX_NAME_COLUMN_WIDTH = 50
    name_column_width = max(
        [len(device_name) for device_name in device_names]) + 4
    name_column_width = max(len(headers[0]), name_column_width)
    name_column_width = min(name_column_width, MAX_NAME_COLUMN_WIDTH)

    DEFAULT_COLUMN_WIDTH = 20

    add_column(name_column_width)
    for _ in headers[1:]:
        add_column(DEFAULT_COLUMN_WIDTH)

    row_format = row_format_list[0]
    header_sep = header_sep_list[0]
    line_length = line_length_list[0]

    # construct table string
    result = []

    def append(s):
        result.append(s)
        result.append('\n')

    def format_time(time, unit='ms', indent=0):
        '''
      Transform time in ns to time in unit.
      '''
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
        '''
      Transform ratio within [0, 1] to percentage presentation.
      '''
        return '{}{:.2f}'.format(' ' * indent, ratio * 100)

    append(add_title(line_length, "Device Summary"))
    append('Time unit: {}'.format(time_unit))
    append(header_sep)
    append(row_format.format(*headers))
    append(header_sep)
    row_values = []
    row_values.extend([
        'CPU', format_ratio(
            float(statistic_data.extra_info['Process Cpu Utilization']))
    ])
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
        "Note:\nCPU Utilization = process CPU time over all cpu cores / elapsed time, so max utilization can be reached 100% * number of cpu cores.\n"
        "GPU Utilization = process GPU time / elapsed time")
    append('-' * line_length)
    append('')
    append('')
    ###### Print Overview Summary ######
    headers = [
        'Name',
        'Time',
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
    append(add_title(line_length, "Overview Summary"))
    append('Time unit: {}'.format(time_unit))
    append(header_sep)
    append(row_format.format(*headers))
    append(header_sep)
    row_values = [
        'Total time', format_time(
            total_time, unit=time_unit), format_ratio(1)
    ]
    append(row_format.format(*row_values))
    dataloader_time = statistic_data.time_range_summary.get_cpu_range_sum(
        TracerEventType.Dataloader)
    row_values = [
        '  DataLoader', format_time(
            dataloader_time, unit=time_unit, indent=2), format_ratio(
                float(dataloader_time) / total_time, indent=2)
    ]
    append(row_format.format(*row_values))
    operator_time = statistic_data.time_range_summary.get_cpu_range_sum(
        TracerEventType.Operator)
    row_values = [
        '  Operator', format_time(
            operator_time, unit=time_unit, indent=2), format_ratio(
                float(operator_time) / total_time, indent=2)
    ]
    append(row_format.format(*row_values))
    if op_detail:
        prepare_data_time = statistic_data.time_range_summary.get_cpu_range_sum(
            'prepare_data')
        row_values = [
            '    Prepare data', format_time(
                prepare_data_time, unit=time_unit, indent=4), format_ratio(
                    float(prepare_data_time) / operator_time, indent=4)
        ]
        append(row_format.format(*row_values))
        infer_shape_time = statistic_data.time_range_summary.get_cpu_range_sum(
            'infer_shape')
        row_values = [
            '    Infer Shape', format_time(
                infer_shape_time, unit=time_unit, indent=4), format_ratio(
                    float(infer_shape_time) / operator_time, indent=4)
        ]
        append(row_format.format(*row_values))
        compute_time = statistic_data.time_range_summary.get_cpu_range_sum(
            'compute')
        row_values = [
            '    Compute', format_time(
                compute_time, unit=time_unit, indent=4), format_ratio(
                    float(compute_time) / operator_time, indent=4)
        ]
        append(row_format.format(*row_values))

    cudaapi_time = statistic_data.time_range_summary.get_cpu_range_sum(
        TracerEventType.CudaRuntime)
    row_values = [
        '  CudaRuntime', format_time(
            cudaapi_time, unit=time_unit, indent=2), format_ratio(
                float(cudaapi_time) / total_time, indent=2)
    ]
    append(row_format.format(*row_values))
    append(header_sep)
    kernel_range = []
    for device_id, device_time_ranges in statistic_data.time_range_summary.GPUTimeRange.items(
    ):
        kernel_range = merge_ranges(
            device_time_ranges[TracerEventType.Kernel],
            kernel_range,
            is_sorted=True)
    kernel_time = sum_ranges(kernel_range)
    row_values = [
        '  Kernel', format_time(
            kernel_time, unit=time_unit, indent=2), format_ratio(
                float(kernel_time) / total_time, indent=2)
    ]
    append(row_format.format(*row_values))
    memcpy_range = []
    for device_id, device_time_ranges in statistic_data.time_range_summary.GPUTimeRange.items(
    ):
        memcpy_range = merge_ranges(
            device_time_ranges[TracerEventType.Memcpy],
            memcpy_range,
            is_sorted=True)
    memcpy_time = sum_ranges(memcpy_range)
    row_values = [
        '  Memcpy', format_time(
            memcpy_time, unit=time_unit, indent=2), format_ratio(
                float(memcpy_time) / total_time, indent=2)
    ]
    append(row_format.format(*row_values))
    memset_range = []
    for device_id, device_time_ranges in statistic_data.time_range_summary.GPUTimeRange.items(
    ):
        memset_range = merge_ranges(
            device_time_ranges[TracerEventType.Memset],
            memset_range,
            is_sorted=True)
    memset_time = sum_ranges(memset_range)
    row_values = [
        '  Memset', format_time(
            memset_time, unit=time_unit, indent=2), format_ratio(
                float(memset_time) / total_time, indent=2)
    ]
    append(row_format.format(*row_values))
    # overhead_time = total_time - operator_time - dataloader_time
    # row_values = ['Framework overhead', format_time(overhead_time), format_ratio(float(overhead_time)/total_time)]
    # append(row_format.format(*row_values))
    append(header_sep)
    append(
        "Note:\nKernel,Memcpy,Memset are events on GPU, which is asynchronous.\n"
        "Although, we calculate ratio = event time / elapsed time(total time) here.\n"
    )
    append('-' * line_length)
    append('')
    append('')
    ###### Print Model Summary Report ######
    headers = [
        'Name',
        'Time',
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
    append(add_title(line_length, "Model Summary"))
    append('Time unit: {}'.format(time_unit))
    append(header_sep)
    append(row_format.format(*headers))
    append(header_sep)
    dataloader_time = statistic_data.time_range_summary.get_cpu_range_sum(
        TracerEventType.Dataloader)
    row_values = [
        'DataLoader', format_time(
            dataloader_time, unit=time_unit, indent=2),
        format_ratio(float(dataloader_time) / total_time)
    ]
    append(row_format.format(*row_values))
    forward_time = statistic_data.time_range_summary.get_cpu_range_sum(
        TracerEventType.Forward)
    row_values = [
        'Forward', format_time(
            forward_time, unit=time_unit, indent=2),
        format_ratio(float(forward_time) / total_time)
    ]
    append(row_format.format(*row_values))
    backward_time = statistic_data.time_range_summary.get_cpu_range_sum(
        TracerEventType.Backward)
    row_values = [
        'Backward', format_time(
            backward_time, unit=time_unit, indent=2),
        format_ratio(float(backward_time) / total_time)
    ]
    append(row_format.format(*row_values))
    optimization_time = statistic_data.time_range_summary.get_cpu_range_sum(
        TracerEventType.Optimization)
    row_values = [
        'Optimization', format_time(
            optimization_time, unit=time_unit, indent=2),
        format_ratio(float(optimization_time) / total_time)
    ]
    append(row_format.format(*row_values))
    other_time = total_time - dataloader_time - forward_time - backward_time - optimization_time
    row_values = [
        'Others', format_time(
            other_time, unit=time_unit, indent=2),
        format_ratio(float(other_time) / total_time)
    ]
    append(row_format.format(*row_values))
    append(header_sep)
    append('')
    append('')

    ###### Print Distribution Summary Report ######
    headers = [
        'Name',
        'Time',
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
        "Overlap time: Communication time intersect with computation time.")
    append('-' * line_length)
    append('')
    append('')

    ###### Print Operator Summary Report ######
    headers = [
        'Name',
        'Calls',
        'Total Time',
        'Avg Time',
        'MAX Time',
        'Min Time',
        'Ratio (%)',
    ]
    row_format_list = [""]
    header_sep_list = [""]
    line_length_list = [-SPACING_SIZE]
    name_column_width = 20
    add_column(name_column_width)
    DEFAULT_COLUMN_WIDTH = 10
    for _ in headers[1:]:
        add_column(DEFAULT_COLUMN_WIDTH)

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
        thread_items = statistic_data.operator_summary.thread_items
    else:
        thread_items = {
            'All threads merged': statistic_data.operator_summary.items
        }
    for thread_id, items in thread_items.items():
        append(add_title(line_length, "Thread: {}".format(thread_id)))
        if sorted_by == SortedKeys.OpTotal:
            sorted_items = sorted(
                items.items(), key=lambda x: x[1].cpu_time, reverse=True)
        elif sorted_by == SortedKeys.OpAvg:
            sorted_items = sorted(
                items.items(), key=lambda x: x[1].avg_cpu_time, reverse=True)
        elif sorted_by == SortedKeys.OpMax:
            sorted_items = sorted(
                items.items(), key=lambda x: x[1].max_cpu_time, reverse=True)
        elif sorted_by == SortedKeys.OpMin:
            sorted_items = sorted(
                items.items(), key=lambda x: x[1].min_cpu_time)
        elif sorted_by == SortedKeys.KernelTotal:
            sorted_items = sorted(
                items.items(), key=lambda x: x[1].gpu_time, reverse=True)
        elif sorted_by == SortedKeys.KernelAvg:
            sorted_items = sorted(
                items.items(), key=lambda x: x[1].avg_gpu_time, reverse=True)
        elif sorted_by == SortedKeys.KernelMax:
            sorted_items = sorted(
                items.items(), key=lambda x: x[1].max_gpu_time, reverse=True)
        elif sorted_by == SortedKeys.KernelMin:
            sorted_items = sorted(
                items.items(), key=lambda x: x[1].min_gpu_time)

        total_op_time = 0
        total_gpu_time = 0
        for name, item in sorted_items:
            total_op_time += item.cpu_time
            total_gpu_time += item.gpu_time
        for name, item in sorted_items:
            row_values = [
                name, item.call, format_time(
                    item.cpu_time, unit=time_unit), format_time(
                        item.avg_cpu_time, unit=time_unit), format_time(
                            item.max_cpu_time, unit=time_unit), format_time(
                                item.min_cpu_time, unit=time_unit),
                format_ratio(float(item.cpu_time) / total_op_time)
            ]
            append(row_format.format(*row_values))
            if op_detail:
                if format_time(item.min_prepare_data_time) == '-':
                    row_values = [
                        '  Prepare Data', '  -', '  -', '  -', '  -', '  -',
                        '  -'
                    ]
                else:
                    row_values = [
                        '  Prepare Data', '  {}'.format(item.call), format_time(
                            item.prepare_data_time, unit=time_unit,
                            indent=2), format_time(
                                item.avg_prepare_data_time,
                                unit=time_unit,
                                indent=2), format_time(
                                    item.max_prepare_data_time,
                                    unit=time_unit,
                                    indent=2), format_time(
                                        item.min_prepare_data_time,
                                        unit=time_unit,
                                        indent=2),
                        format_ratio(
                            float(item.prepare_data_time) / item.cpu_time,
                            indent=2)
                    ]
                append(row_format.format(*row_values))
                row_values = [
                    '  Infer Shape', '  {}'.format(item.call), format_time(
                        item.infer_shape_time, unit=time_unit,
                        indent=2), format_time(
                            item.avg_infer_shape_time, unit=time_unit,
                            indent=2), format_time(
                                item.max_infer_shape_time,
                                unit=time_unit,
                                indent=2), format_time(
                                    item.min_infer_shape_time,
                                    unit=time_unit,
                                    indent=2),
                    format_ratio(
                        float(item.infer_shape_time) / item.cpu_time, indent=2)
                ]
                append(row_format.format(*row_values))
                row_values = [
                    '  Compute', '  {}'.format(item.call), format_time(
                        item.computation_time, unit=time_unit,
                        indent=2), format_time(
                            item.avg_computation_time, unit=time_unit,
                            indent=2), format_time(
                                item.max_computation_time,
                                unit=time_unit,
                                indent=2), format_time(
                                    item.min_computation_time,
                                    unit=time_unit,
                                    indent=2),
                    format_ratio(
                        float(item.computation_time) / item.cpu_time, indent=2)
                ]
                append(row_format.format(*row_values))

            sorted_device_items = sorted(
                item.devices.items(), key=lambda x: x[1].gpu_time, reverse=True)
            for device_activity_name, devicenode in sorted_device_items:
                row_values = [
                    '  {}...(Device)'.format(device_activity_name[:10]),
                    '  {}'.format(devicenode.call), format_time(
                        devicenode.gpu_time, unit=time_unit,
                        indent=2), format_time(
                            devicenode.avg_gpu_time, unit=time_unit,
                            indent=2), format_time(
                                devicenode.max_gpu_time,
                                unit=time_unit,
                                indent=2), format_time(
                                    devicenode.min_gpu_time,
                                    unit=time_unit,
                                    indent=2), format_ratio(
                                        float(devicenode.gpu_time) /
                                        total_gpu_time,
                                        indent=2)
                ]
                append(row_format.format(*row_values))
        append(header_sep)

    append(
        "Note:\n- means unavailable.\nKernel(Device) time is the computation time of the op on GPU, which is asynchronous.\n"
        "Except kernel time, all other time is collected on host(cpu).\n"
        "Op Ratio = (Op Total Time) / (sum(All Op Total Time)).\nOp::Prepare Data Ratio = (Op::Prepare Data Total Time) / (Op Total Time).\n"
        "Op::Infer Shape Ratio = (Op::Infer Shape Total Time) / (Op Total Time).\nOp::Compute Ratio = (Op::Compute Total Time) / (Op Total Time).\n"
        "Op::Kernel Ratio = (Op::Kernel Total Time) / (sum(All Op Kernel Total Time))."
    )
    append('-' * line_length)
    return ''.join(result)
