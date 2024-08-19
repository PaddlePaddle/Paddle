# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from dataclasses import dataclass
from enum import Enum

from paddle import profiler
from paddle.base.core import TracerEventType

from ..utils.envs import ENV_ENABLE_SOT_STEP_PROFILER

EVENT_TYPE_NAME_MAPPING = {
    TracerEventType.Operator: "Operator",
    TracerEventType.Dataloader: "Dataloader",
    TracerEventType.ProfileStep: "ProfileStep",
    TracerEventType.CudaRuntime: "CudaRuntime",
    TracerEventType.Kernel: "Kernel",
    TracerEventType.Memcpy: "Memcpy",
    TracerEventType.Memset: "Memset",
    TracerEventType.UserDefined: "UserDefined",
    TracerEventType.OperatorInner: "OperatorInner",
    TracerEventType.Forward: "Forward",
    TracerEventType.Backward: "Backward",
    TracerEventType.Optimization: "Optimization",
    TracerEventType.Communication: "Communication",
    TracerEventType.PythonOp: "PythonOp",
    TracerEventType.PythonUserDefined: "PythonUserDefined",
    TracerEventType.DygraphKernelCall: "DygraphKernelCall",
    TracerEventType.StaticKernelCall: "StaticKernelCall",
}


class EventVisitor:
    def visit(self, event_node):
        event_type_name = EVENT_TYPE_NAME_MAPPING[event_node.type]
        visit_method_name = f"visit_{event_type_name}"
        if not hasattr(self, visit_method_name):
            self.generic_visit(event_node)
            return
        getattr(self, visit_method_name)(event_node)

    def generic_visit(self, event_node):
        for child in event_node.children_node:
            self.visit(child)

    def __call__(self, events):
        for event_node in events.values():
            self.visit(event_node)


class KernelRunMode(Enum):
    Dygraph = 1
    Static = 2


@dataclass
class KernelInfo:
    # name: str # TODO: Add name field to KernelInfo
    run_mode: KernelRunMode
    duration: float


class KernelStatsVisitor(EventVisitor):
    def __init__(self):
        self.kernels = []

    def calc_kernel_count(self, mode):
        return len(
            [kernel for kernel in self.kernels if kernel.run_mode == mode]
        )

    def calc_kernel_duration(self, mode):
        return sum(
            [
                kernel.duration
                for kernel in self.kernels
                if kernel.run_mode == mode
            ]
        )

    def visit_DygraphKernelCall(self, event_node):
        duration = event_node.end_ns - event_node.start_ns
        self.kernels.append(KernelInfo(KernelRunMode.Dygraph, duration))
        self.generic_visit(event_node)

    def visit_StaticKernelCall(self, event_node):
        duration = event_node.end_ns - event_node.start_ns
        self.kernels.append(KernelInfo(KernelRunMode.Static, duration))
        self.generic_visit(event_node)

    def print_summary(self):
        static_kernel_duration = self.calc_kernel_duration(KernelRunMode.Static)
        dygraph_kernel_duration = self.calc_kernel_duration(
            KernelRunMode.Dygraph
        )
        static_kernel_count = self.calc_kernel_count(KernelRunMode.Static)
        dygraph_kernel_count = self.calc_kernel_count(KernelRunMode.Dygraph)

        percentage_static_kernel_count = static_kernel_count / (
            static_kernel_count + dygraph_kernel_count
        )
        percentage_static_kernel_duration = static_kernel_duration / (
            static_kernel_duration + dygraph_kernel_duration
        )
        print(f"dygraph kernel count: {dygraph_kernel_count}")
        print(f"static kernel count: {static_kernel_count}")
        print(
            f"percentage dygraph kernel count: {percentage_static_kernel_count:.2%}"
        )
        print(
            f"dygraph kernel duration: {dygraph_kernel_duration / 1000:.2f} ms"
        )
        print(f"static kernel duration: {static_kernel_duration / 1000:.2f} ms")
        print(
            f"percentage dygraph kernel duration: {percentage_static_kernel_duration:.2%}"
        )


class SotStepProfilerGuard:
    def __init__(self):
        self.started = False

    def _kernel_stats(self, prof):
        kernel_stats_visitor = KernelStatsVisitor()
        kernel_stats_visitor(prof.profiler_result.get_data())
        kernel_stats_visitor.print_summary()

    def on_trace_ready(self, prof):
        self._kernel_stats(prof)

    def start(self):
        if ENV_ENABLE_SOT_STEP_PROFILER.get():
            self.profiler = profiler.Profiler(
                targets=[
                    profiler.ProfilerTarget.CPU,
                    profiler.ProfilerTarget.GPU,
                ],
                on_trace_ready=self.on_trace_ready,
            )
            self.profiler.start()
            self.started = True

    def stop(self):
        if self.started:
            self.profiler.stop()

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
