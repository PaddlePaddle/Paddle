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


class TestProfilerStatistic(unittest.TestCase):
    def test_statistic_case1(self):
        root_node = HostPythonNode('Root Node',
                                   profiler.TracerEventType.UserDefined, 0,
                                   float('inf'), 1000, 1001)
        profilerstep_node = HostPythonNode('ProfileStep#1',
                                           profiler.TracerEventType.ProfileStep,
                                           0, 400, 1000, 1001)
        dataloader_node = HostPythonNode(
            'Dataloader', profiler.TracerEventType.Forward, 5, 15, 1000, 1001)
        mobilenet_node = HostPythonNode(
            'MobileNet', profiler.TracerEventType.Forward, 20, 50, 1000, 1001)
        yolonet_node = HostPythonNode(
            'Yolov3Net', profiler.TracerEventType.Forward, 50, 110, 1000, 1001)

        userdefined_node = HostPythonNode('Communication Time',
                                          profiler.TracerEventType.UserDefined,
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
        conv2d_node = HostPythonNode(
            'conv2d', profiler.TracerEventType.Operator, 25, 40, 1000, 1001)
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
        conv2d_kernel = DevicePythonNode(
            'conv2d_kernel', profiler.TracerEventType.Kernel, 35, 50, 0, 0, 0)
        conv2d_memcpy = DevicePythonNode(
            'conv2d_memcpy', profiler.TracerEventType.Memcpy, 50, 60, 0, 0, 0)
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
        statistic_data = profiler.profiler_statistic.StatisticData(thread_tree,
                                                                   extra_info)
        time_range_summary = statistic_data.time_range_summary
        event_summary = statistic_data.event_summary

        self.assertEqual(
            time_range_summary.get_cpu_range_sum(
                profiler.TracerEventType.ProfileStep), 400)
        self.assertEqual(
            time_range_summary.get_cpu_range_sum(
                profiler.TracerEventType.Forward), 100)
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
                profiler.TracerEventType.UserDefined), 25)
        self.assertEqual(
            time_range_summary.get_cpu_range_sum(
                profiler.TracerEventType.Communication), 5)
        self.assertEqual(len(event_summary.items), 2)
        self.assertEqual(len(event_summary.userdefined_items), 1)
        self.assertEqual(len(event_summary.model_perspective_items), 3)
        self.assertEqual(len(event_summary.memory_manipulation_items), 1)
        self.assertEqual(event_summary.items['conv2d'].cpu_time, 15)
        self.assertEqual(event_summary.items['conv2d'].gpu_time, 25)
        self.assertEqual(
            event_summary.model_perspective_items['Forward'].cpu_time, 100)
        self.assertEqual(
            event_summary.model_perspective_items['Forward'].gpu_time, 135)
        self.assertEqual(
            event_summary.model_perspective_items['Backward'].gpu_time, 0)
        self.assertEqual(
            event_summary.memory_manipulation_items['AsyncMemcpy'].cpu_time, 15)
        self.assertEqual(
            event_summary.memory_manipulation_items['AsyncMemcpy'].gpu_time, 60)
        print(
            profiler.profiler_statistic._build_table(
                statistic_data,
                sorted_by=profiler.SortedKeys.CPUTotal,
                op_detail=True,
                thread_sep=False,
                time_unit='ms'))


if __name__ == '__main__':
    unittest.main()
