#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import paddle
import paddle.profiler as profiler
import paddle.profiler.profiler_statistic as profiler_statistic


class HostPythonNode:

    def __init__(self, name, type, start_ns, end_ns, process_id, thread_id):
        self.name = name
        self.type = type
        self.start_ns = start_ns
        self.end_ns = end_ns
        self.process_id = process_id
        self.thread_id = thread_id
        self.children_node = []
        self.runtime_node = []
        self.device_node = []
        self.mem_node = []


class DevicePythonNode:

    def __init__(self, name, type, start_ns, end_ns, device_id, context_id,
                 stream_id):
        self.name = name
        self.type = type
        self.start_ns = start_ns
        self.end_ns = end_ns
        self.device_id = device_id
        self.context_id = context_id
        self.stream_id = stream_id


class MemPythonNode:
    def __init__(self, timestamp_ns, addr, type, process_id, thread_id, increase_bytes, place, current_allocated, \
        current_reserved, peak_allocated, peak_reserved):
        self.timestamp_ns = timestamp_ns
        self.addr = addr
        self.type = type
        self.process_id = process_id
        self.thread_id = thread_id
        self.increase_bytes = increase_bytes
        self.place = place
        self.current_allocated = current_allocated
        self.current_reserved = current_reserved
        self.peak_allocated = peak_allocated
        self.peak_reserved = peak_reserved


class TestProfilerStatistic(unittest.TestCase):

    def test_statistic_case1(self):
        root_node = HostPythonNode('Root Node',
                                   profiler.TracerEventType.UserDefined, 0,
                                   float('inf'), 1000, 1001)
        profilerstep_node = HostPythonNode('ProfileStep#1',
                                           profiler.TracerEventType.ProfileStep,
                                           0, 400, 1000, 1001)
        dataloader_node = HostPythonNode('Dataloader',
                                         profiler.TracerEventType.Dataloader, 5,
                                         15, 1000, 1001)
        mobilenet_node = HostPythonNode('MobileNet',
                                        profiler.TracerEventType.Forward, 20,
                                        50, 1000, 1001)
        yolonet_node = HostPythonNode('Yolov3Net',
                                      profiler.TracerEventType.Forward, 50, 110,
                                      1000, 1001)

        userdefined_node = HostPythonNode(
            'Communication Time', profiler.TracerEventType.PythonUserDefined,
            100, 110, 1000, 1001)

        communication_node = HostPythonNode(
            'Communication', profiler.TracerEventType.Communication, 105, 110,
            1000, 1001)
        backward_node = HostPythonNode('Gradient Backward',
                                       profiler.TracerEventType.Backward, 120,
                                       200, 1000, 1001)
        optimization_node = HostPythonNode(
            'Optimization', profiler.TracerEventType.Optimization, 220, 300,
            1000, 1001)
        conv2d_node = HostPythonNode('conv2d',
                                     profiler.TracerEventType.Operator, 25, 40,
                                     1000, 1001)
        sync_batch_norm_node = HostPythonNode('sync_batch_norm',
                                              profiler.TracerEventType.Operator,
                                              60, 100, 1000, 1001)
        conv2d_infer_shape = HostPythonNode(
            'conv2d::infer_shape', profiler.TracerEventType.OperatorInner, 25,
            30, 1000, 1001)
        conv2d_compute = HostPythonNode('conv2d::compute',
                                        profiler.TracerEventType.OperatorInner,
                                        30, 40, 1000, 1001)
        conv2d_compute.mem_node.append(
            MemPythonNode(33, 0, profiler_statistic.TracerMemEventType.Allocate,
                          1000, 1001, 20, 'place(gpu:0)', 200, 200, 800, 800))
        conv2d_launchkernel = HostPythonNode(
            'cudalaunchkernel', profiler.TracerEventType.CudaRuntime, 30, 35,
            1000, 1001)
        conv2d_MemCpy = HostPythonNode('AsyncMemcpy',
                                       profiler.TracerEventType.UserDefined, 35,
                                       40, 1000, 1001)
        conv2d_cudaMemCpy = HostPythonNode('cudaMemcpy',
                                           profiler.TracerEventType.CudaRuntime,
                                           35, 40, 1000, 1001)
        conv2d_kernel = DevicePythonNode('conv2d_kernel',
                                         profiler.TracerEventType.Kernel, 35,
                                         50, 0, 0, 0)
        conv2d_memcpy = DevicePythonNode('conv2d_memcpy',
                                         profiler.TracerEventType.Memcpy, 50,
                                         60, 0, 0, 0)
        sync_batch_norm_infer_shape = HostPythonNode(
            'sync_batch_norm::infer_shape',
            profiler.TracerEventType.OperatorInner, 60, 70, 1000, 1001)
        sync_batch_norm_compute = HostPythonNode(
            'sync_batch_norm::compute', profiler.TracerEventType.OperatorInner,
            80, 100, 1000, 1001)
        sync_batch_norm_launchkernel = HostPythonNode(
            'cudalaunchkernel', profiler.TracerEventType.CudaRuntime, 80, 90,
            1000, 1001)
        sync_batch_norm_MemCpy = HostPythonNode(
            'AsyncMemcpy', profiler.TracerEventType.UserDefined, 90, 100, 1000,
            1001)
        sync_batch_norm_cudaMemCpy = HostPythonNode(
            'cudaMemcpy', profiler.TracerEventType.CudaRuntime, 90, 100, 1000,
            1001)
        sync_batch_norm_kernel = DevicePythonNode(
            'sync_batch_norm_kernel', profiler.TracerEventType.Kernel, 95, 155,
            0, 0, 0)
        sync_batch_norm_memcpy = DevicePythonNode(
            'sync_batch_norm_memcpy', profiler.TracerEventType.Memcpy, 150, 200,
            0, 0, 1)
        root_node.children_node.append(profilerstep_node)
        profilerstep_node.children_node.extend([
            dataloader_node, mobilenet_node, yolonet_node, backward_node,
            optimization_node
        ])
        mobilenet_node.children_node.append(conv2d_node)
        yolonet_node.children_node.extend(
            [sync_batch_norm_node, userdefined_node])
        userdefined_node.children_node.append(communication_node)
        conv2d_node.children_node.extend(
            [conv2d_infer_shape, conv2d_compute, conv2d_MemCpy])
        conv2d_compute.runtime_node.append(conv2d_launchkernel)
        conv2d_MemCpy.runtime_node.append(conv2d_cudaMemCpy)
        conv2d_launchkernel.device_node.append(conv2d_kernel)
        conv2d_cudaMemCpy.device_node.append(conv2d_memcpy)
        sync_batch_norm_node.children_node.extend([
            sync_batch_norm_infer_shape, sync_batch_norm_compute,
            sync_batch_norm_MemCpy
        ])
        sync_batch_norm_compute.runtime_node.append(
            sync_batch_norm_launchkernel)
        sync_batch_norm_MemCpy.runtime_node.append(sync_batch_norm_cudaMemCpy)
        sync_batch_norm_launchkernel.device_node.append(sync_batch_norm_kernel)
        sync_batch_norm_cudaMemCpy.device_node.append(sync_batch_norm_memcpy)
        thread_tree = {'thread1001': root_node}
        extra_info = {
            'Process Cpu Utilization': '1.02',
            'System Cpu Utilization': '0.68'
        }
        statistic_data = profiler.profiler_statistic.StatisticData(
            thread_tree, extra_info)
        time_range_summary = statistic_data.time_range_summary
        event_summary = statistic_data.event_summary

        self.assertEqual(
            time_range_summary.get_cpu_range_sum(
                profiler.TracerEventType.ProfileStep), 400)
        self.assertEqual(
            time_range_summary.get_cpu_range_sum(
                profiler.TracerEventType.Forward), 90)
        self.assertEqual(
            time_range_summary.get_cpu_range_sum(
                profiler.TracerEventType.Backward), 80)
        self.assertEqual(
            time_range_summary.get_cpu_range_sum(
                profiler.TracerEventType.Optimization), 80)
        self.assertEqual(
            time_range_summary.get_cpu_range_sum(
                profiler.TracerEventType.Operator), 55)
        self.assertEqual(
            time_range_summary.get_cpu_range_sum(
                profiler.TracerEventType.OperatorInner), 45)
        self.assertEqual(
            time_range_summary.get_cpu_range_sum(
                profiler.TracerEventType.CudaRuntime), 30)
        self.assertEqual(
            time_range_summary.get_gpu_range_sum(
                0, profiler.TracerEventType.Kernel), 75)
        self.assertEqual(
            time_range_summary.get_gpu_range_sum(
                0, profiler.TracerEventType.Memcpy), 60)
        self.assertEqual(
            time_range_summary.get_cpu_range_sum(
                profiler.TracerEventType.UserDefined), 15)
        self.assertEqual(
            time_range_summary.get_cpu_range_sum(
                profiler.TracerEventType.Communication), 5)
        self.assertEqual(len(event_summary.items), 2)
        self.assertEqual(len(event_summary.userdefined_items), 1)
        self.assertEqual(len(event_summary.model_perspective_items), 5)
        self.assertEqual(len(event_summary.memory_manipulation_items), 1)
        self.assertEqual(event_summary.items['conv2d'].cpu_time, 15)
        self.assertEqual(event_summary.items['conv2d'].general_gpu_time, 25)
        self.assertEqual(
            event_summary.model_perspective_items['Forward'].cpu_time, 90)
        self.assertEqual(
            event_summary.model_perspective_items['Forward'].general_gpu_time,
            135)
        self.assertEqual(
            event_summary.model_perspective_items['Backward'].general_gpu_time,
            0)
        self.assertEqual(
            event_summary.memory_manipulation_items['AsyncMemcpy'].cpu_time, 15)
        self.assertEqual(
            event_summary.memory_manipulation_items['AsyncMemcpy'].
            general_gpu_time, 60)
        self.assertEqual(
            statistic_data.memory_summary.allocated_items['place(gpu:0)']
            ['conv2d'].allocation_count, 1)
        self.assertEqual(
            statistic_data.memory_summary.allocated_items['place(gpu:0)']
            ['conv2d'].allocation_size, 20)
        self.assertEqual(
            statistic_data.memory_summary.allocated_items['place(gpu:0)']
            ['conv2d'].increase_size, 20)
        self.assertEqual(
            statistic_data.memory_summary.allocated_items['place(gpu:0)']
            ['conv2d'].increase_size, 20)
        self.assertEqual(
            statistic_data.memory_summary.
            peak_allocation_values['place(gpu:0)'], 800)
        self.assertEqual(
            statistic_data.memory_summary.peak_reserved_values['place(gpu:0)'],
            800)
        print(
            profiler.profiler_statistic._build_table(
                statistic_data,
                sorted_by=profiler.SortedKeys.CPUTotal,
                op_detail=True,
                thread_sep=False,
                time_unit='ms'))

    def test_statistic_case2(self):
        root_node = HostPythonNode('Root Node',
                                   profiler.TracerEventType.UserDefined, 0,
                                   float('inf'), 1000, 1001)
        profilerstep_node = HostPythonNode('ProfileStep#1',
                                           profiler.TracerEventType.ProfileStep,
                                           0, 400, 1000, 1001)

        dataloader_node = HostPythonNode('Dataloader',
                                         profiler.TracerEventType.Dataloader, 5,
                                         15, 1000, 1001)

        mobilenet_node = HostPythonNode('MobileNet',
                                        profiler.TracerEventType.Forward, 20,
                                        50, 1000, 1001)
        yolonet_node = HostPythonNode('Yolov3Net',
                                      profiler.TracerEventType.Forward, 50, 110,
                                      1000, 1001)

        userdefined_node = HostPythonNode(
            'Communication Time', profiler.TracerEventType.PythonUserDefined,
            100, 110, 1000, 1001)
        allreduce_launchkernel0 = HostPythonNode(
            'cudalaunchkernel', profiler.TracerEventType.CudaRuntime, 102, 104,
            1000, 1001)

        nccl_allreduce_kernel0 = DevicePythonNode(
            'nccl_allreduce_kernel', profiler.TracerEventType.Kernel, 105, 120,
            0, 0, 2)

        communication_node = HostPythonNode(
            'Communication', profiler.TracerEventType.Communication, 105, 110,
            1000, 1001)

        allreduce_op1 = HostPythonNode('allreduce_op1',
                                       profiler.TracerEventType.Operator, 105,
                                       108, 1000, 1001)
        allreduce_op1_infershape = HostPythonNode(
            'allreduce_op1::infershape', profiler.TracerEventType.OperatorInner,
            105, 106, 1000, 1001)

        allreduce_launchkernel1 = HostPythonNode(
            'cudalaunchkernel', profiler.TracerEventType.CudaRuntime, 106, 107,
            1000, 1001)

        nccl_allreduce_kernel1 = DevicePythonNode(
            'nccl_allreduce_kernel', profiler.TracerEventType.Kernel, 130, 150,
            0, 0, 2)

        backward_node = HostPythonNode('Gradient Backward',
                                       profiler.TracerEventType.Backward, 120,
                                       200, 1000, 1001)
        optimization_node = HostPythonNode(
            'Optimization', profiler.TracerEventType.Optimization, 220, 300,
            1000, 1001)
        conv2d_node = HostPythonNode('conv2d',
                                     profiler.TracerEventType.Operator, 25, 40,
                                     1000, 1001)
        sync_batch_norm_node = HostPythonNode('sync_batch_norm',
                                              profiler.TracerEventType.Operator,
                                              60, 100, 1000, 1001)
        conv2d_infer_shape = HostPythonNode(
            'conv2d::infer_shape', profiler.TracerEventType.OperatorInner, 25,
            30, 1000, 1001)
        conv2d_compute = HostPythonNode('conv2d::compute',
                                        profiler.TracerEventType.OperatorInner,
                                        30, 40, 1000, 1001)
        conv2d_launchkernel = HostPythonNode(
            'cudalaunchkernel', profiler.TracerEventType.CudaRuntime, 30, 35,
            1000, 1001)
        conv2d_MemCpy = HostPythonNode('AsyncMemcpy',
                                       profiler.TracerEventType.UserDefined, 35,
                                       40, 1000, 1001)
        conv2d_cudaMemCpy = HostPythonNode('cudaMemcpy',
                                           profiler.TracerEventType.CudaRuntime,
                                           35, 40, 1000, 1001)
        conv2d_kernel = DevicePythonNode('conv2d_kernel',
                                         profiler.TracerEventType.Kernel, 35,
                                         50, 0, 0, 0)
        conv2d_memcpy = DevicePythonNode('conv2d_memcpy',
                                         profiler.TracerEventType.Memcpy, 50,
                                         60, 0, 0, 0)
        sync_batch_norm_infer_shape = HostPythonNode(
            'sync_batch_norm::infer_shape',
            profiler.TracerEventType.OperatorInner, 60, 70, 1000, 1001)
        sync_batch_norm_compute = HostPythonNode(
            'sync_batch_norm::compute', profiler.TracerEventType.OperatorInner,
            80, 100, 1000, 1001)
        sync_batch_norm_launchkernel = HostPythonNode(
            'cudalaunchkernel', profiler.TracerEventType.CudaRuntime, 80, 90,
            1000, 1001)
        sync_batch_norm_MemCpy = HostPythonNode(
            'AsyncMemcpy', profiler.TracerEventType.UserDefined, 90, 100, 1000,
            1001)
        sync_batch_norm_cudaMemCpy = HostPythonNode(
            'cudaMemcpy', profiler.TracerEventType.CudaRuntime, 90, 100, 1000,
            1001)
        sync_batch_norm_kernel = DevicePythonNode(
            'sync_batch_norm_kernel', profiler.TracerEventType.Kernel, 95, 300,
            0, 0, 0)
        sync_batch_norm_memcpy = DevicePythonNode(
            'sync_batch_norm_memcpy', profiler.TracerEventType.Memcpy, 150, 200,
            0, 0, 1)

        allreduce_node2 = HostPythonNode('allreduce',
                                         profiler.TracerEventType.Operator, 230,
                                         250, 1000, 1001)

        allreduce_node2_infershape = HostPythonNode(
            'allreduce_node2::infershape',
            profiler.TracerEventType.OperatorInner, 231, 232, 1000, 1001)
        allreduce_launchkernel2 = HostPythonNode(
            'cudalaunchkernel', profiler.TracerEventType.CudaRuntime, 235, 240,
            1000, 1001)

        nccl_allreduce_kernel2 = DevicePythonNode(
            'nccl_allreduce_kernel', profiler.TracerEventType.Kernel, 250, 280,
            0, 0, 2)

        root_node.children_node.append(profilerstep_node)
        profilerstep_node.children_node.extend([
            dataloader_node, mobilenet_node, yolonet_node, backward_node,
            optimization_node
        ])
        mobilenet_node.children_node.append(conv2d_node)
        yolonet_node.children_node.extend(
            [sync_batch_norm_node, userdefined_node])
        userdefined_node.children_node.append(communication_node)
        userdefined_node.runtime_node.append(allreduce_launchkernel0)
        allreduce_launchkernel0.device_node.append(nccl_allreduce_kernel0)
        communication_node.children_node.append(allreduce_op1)
        allreduce_op1.children_node.append(allreduce_op1_infershape)
        allreduce_op1.runtime_node.append(allreduce_launchkernel1)
        allreduce_launchkernel1.device_node.append(nccl_allreduce_kernel1)
        conv2d_node.children_node.extend(
            [conv2d_infer_shape, conv2d_compute, conv2d_MemCpy])
        conv2d_compute.runtime_node.append(conv2d_launchkernel)
        conv2d_MemCpy.runtime_node.append(conv2d_cudaMemCpy)
        conv2d_launchkernel.device_node.append(conv2d_kernel)
        conv2d_cudaMemCpy.device_node.append(conv2d_memcpy)
        sync_batch_norm_node.children_node.extend([
            sync_batch_norm_infer_shape, sync_batch_norm_compute,
            sync_batch_norm_MemCpy
        ])
        sync_batch_norm_compute.runtime_node.append(
            sync_batch_norm_launchkernel)
        sync_batch_norm_MemCpy.runtime_node.append(sync_batch_norm_cudaMemCpy)
        sync_batch_norm_launchkernel.device_node.append(sync_batch_norm_kernel)
        sync_batch_norm_cudaMemCpy.device_node.append(sync_batch_norm_memcpy)
        optimization_node.children_node.append(allreduce_node2)
        allreduce_node2.children_node.append(allreduce_node2_infershape)
        allreduce_node2.runtime_node.append(allreduce_launchkernel2)
        allreduce_launchkernel2.device_node.append(nccl_allreduce_kernel2)
        thread_tree = {'thread1001': root_node}
        extra_info = {
            'Process Cpu Utilization': '1.02',
            'System Cpu Utilization': '0.68'
        }
        statistic_data = profiler.profiler_statistic.StatisticData(
            thread_tree, extra_info)
        time_range_summary = statistic_data.time_range_summary
        event_summary = statistic_data.event_summary
        distributed_summary = statistic_data.distributed_summary

        self.assertEqual(
            time_range_summary.get_cpu_range_sum(
                profiler.TracerEventType.ProfileStep), 400)
        self.assertEqual(
            time_range_summary.get_cpu_range_sum(
                profiler.TracerEventType.Forward), 90)
        self.assertEqual(
            time_range_summary.get_cpu_range_sum(
                profiler.TracerEventType.Backward), 80)
        self.assertEqual(
            time_range_summary.get_cpu_range_sum(
                profiler.TracerEventType.Optimization), 80)
        self.assertEqual(
            time_range_summary.get_cpu_range_sum(
                profiler.TracerEventType.Operator), 78)
        self.assertEqual(
            time_range_summary.get_cpu_range_sum(
                profiler.TracerEventType.OperatorInner), 47)
        self.assertEqual(
            time_range_summary.get_cpu_range_sum(
                profiler.TracerEventType.CudaRuntime), 38)
        self.assertEqual(
            time_range_summary.get_gpu_range_sum(
                0, profiler.TracerEventType.Kernel), 220)
        self.assertEqual(
            time_range_summary.get_gpu_range_sum(
                0, profiler.TracerEventType.Memcpy), 60)
        self.assertEqual(
            time_range_summary.get_cpu_range_sum(
                profiler.TracerEventType.UserDefined), 15)
        self.assertEqual(
            time_range_summary.get_cpu_range_sum(
                profiler.TracerEventType.Communication), 5)
        self.assertEqual(
            profiler.statistic_helper.sum_ranges(
                distributed_summary.cpu_communication_range), 25)
        self.assertEqual(
            profiler.statistic_helper.sum_ranges(
                distributed_summary.gpu_communication_range), 65)
        self.assertEqual(
            profiler.statistic_helper.sum_ranges(
                distributed_summary.communication_range), 85)
        self.assertEqual(
            profiler.statistic_helper.sum_ranges(
                distributed_summary.computation_range), 220)
        self.assertEqual(
            profiler.statistic_helper.sum_ranges(
                distributed_summary.overlap_range), 85)
        self.assertEqual(len(event_summary.items), 4)
        self.assertEqual(len(event_summary.userdefined_items), 1)
        self.assertEqual(len(event_summary.model_perspective_items), 5)
        self.assertEqual(len(event_summary.memory_manipulation_items), 1)
        self.assertEqual(event_summary.items['conv2d'].cpu_time, 15)
        self.assertEqual(event_summary.items['conv2d'].general_gpu_time, 25)
        self.assertEqual(
            event_summary.model_perspective_items['Forward'].cpu_time, 90)
        self.assertEqual(
            event_summary.model_perspective_items['Forward'].general_gpu_time,
            315)
        self.assertEqual(
            event_summary.model_perspective_items['Backward'].general_gpu_time,
            0)
        self.assertEqual(
            event_summary.memory_manipulation_items['AsyncMemcpy'].cpu_time, 15)
        self.assertEqual(
            event_summary.memory_manipulation_items['AsyncMemcpy'].
            general_gpu_time, 60)
        print(
            profiler.profiler_statistic._build_table(
                statistic_data,
                sorted_by=profiler.SortedKeys.CPUTotal,
                op_detail=True,
                thread_sep=False,
                time_unit='ms'))

    def test_statistic_case3(self):
        # for coverage, test all time is 0
        root_node = HostPythonNode('Root Node',
                                   profiler.TracerEventType.UserDefined, 0,
                                   float('inf'), 1000, 1001)
        profilerstep_node = HostPythonNode('ProfileStep#1',
                                           profiler.TracerEventType.ProfileStep,
                                           0, 400, 1000, 1001)
        dataloader_node = HostPythonNode('Dataloader',
                                         profiler.TracerEventType.Dataloader, 5,
                                         15, 1000, 1001)
        mobilenet_node = HostPythonNode('MobileNet',
                                        profiler.TracerEventType.Forward, 20,
                                        50, 1000, 1001)

        backward_node = HostPythonNode('Gradient Backward',
                                       profiler.TracerEventType.Backward, 120,
                                       200, 1000, 1001)
        optimization_node = HostPythonNode(
            'Optimization', profiler.TracerEventType.Optimization, 220, 300,
            1000, 1001)
        userdefined_node = HostPythonNode(
            'Communication Time', profiler.TracerEventType.PythonUserDefined,
            60, 70, 1000, 1001)

        conv2d_node = HostPythonNode('conv2d',
                                     profiler.TracerEventType.Operator, 25, 25,
                                     1000, 1001)

        conv2d_infer_shape = HostPythonNode(
            'conv2d::infer_shape', profiler.TracerEventType.OperatorInner, 25,
            25, 1000, 1001)
        conv2d_compute = HostPythonNode('conv2d::compute',
                                        profiler.TracerEventType.OperatorInner,
                                        25, 25, 1000, 1001)
        conv2d_launchkernel = HostPythonNode(
            'cudalaunchkernel', profiler.TracerEventType.CudaRuntime, 25, 25,
            1000, 1001)

        conv2d_kernel = DevicePythonNode('conv2d_kernel',
                                         profiler.TracerEventType.Kernel, 35,
                                         35, 0, 0, 0)
        another_kernel = DevicePythonNode(
            'void phi::funcs::VectorizedBroadcastKernel<float, float, phi::funcs::AddFunctor<float>, phi::funcs::AddFunctor<float>>()',
            profiler.TracerEventType.Kernel, 35, 35, 0, 0, 0)
        root_node.children_node.append(profilerstep_node)
        profilerstep_node.children_node.extend([
            dataloader_node, mobilenet_node, userdefined_node, backward_node,
            optimization_node
        ])
        mobilenet_node.children_node.append(conv2d_node)
        conv2d_node.children_node.extend([conv2d_infer_shape, conv2d_compute])
        conv2d_compute.runtime_node.append(conv2d_launchkernel)
        conv2d_launchkernel.device_node.append(conv2d_kernel)
        conv2d_launchkernel.device_node.append(another_kernel)
        thread_tree = {'thread1001': root_node}
        extra_info = {
            'Process Cpu Utilization': '1.02',
            'System Cpu Utilization': '0.68'
        }
        statistic_data = profiler.profiler_statistic.StatisticData(
            thread_tree, extra_info)
        time_range_summary = statistic_data.time_range_summary
        event_summary = statistic_data.event_summary

        self.assertEqual(event_summary.items['conv2d'].cpu_time, 0)
        self.assertEqual(event_summary.items['conv2d'].general_gpu_time, 0)
        self.assertEqual(
            event_summary.userdefined_items['Communication Time'].
            general_gpu_time, 0)
        for sort_key in [
                profiler.SortedKeys.CPUTotal, profiler.SortedKeys.CPUMax,
                profiler.SortedKeys.CPUMin, profiler.SortedKeys.CPUAvg,
                profiler.SortedKeys.GPUTotal, profiler.SortedKeys.GPUMax,
                profiler.SortedKeys.GPUMin, profiler.SortedKeys.GPUAvg
        ]:
            print(
                profiler.profiler_statistic._build_table(statistic_data,
                                                         sorted_by=sort_key,
                                                         op_detail=True,
                                                         thread_sep=False,
                                                         time_unit='ms'))


if __name__ == '__main__':
    unittest.main()
