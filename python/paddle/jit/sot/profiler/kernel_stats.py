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

from __future__ import annotations

import atexit
import re
from dataclasses import dataclass
from enum import Enum

from paddle import profiler
from paddle.base.core import tracer_event_type_to_string

from ..utils.envs import ENV_ENABLE_SOT_STEP_PROFILER


class EventVisitor:
    def visit(self, event_node):
        event_type_name = tracer_event_type_to_string(event_node.type)
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


def safe_divide(a, b):
    # Avoid division by zero
    return a / b if b != 0 else 0


@dataclass
class KernelInfo:
    name: str
    run_mode: KernelRunMode
    duration: float
    cuda_kernels: list[CudaKernelInfo]


@dataclass
class CudaKernelInfo:
    name: str
    duration: float


# TODO(SigureMo): Split into multiple files by visitor type
class KernelStatsVisitor(EventVisitor):
    SKIP_KERNEL_NAMES = {"full", "full_int_array", "shadow_feed"}
    KERNEL_NAME_REGEX = re.compile("(?P<kernel_name>.+) kernel launch")

    def __init__(self):
        self.kernels = []

    def get_kernel_name(self, event_name):
        if match_obj := self.KERNEL_NAME_REGEX.match(event_name):
            return match_obj.group("kernel_name")
        raise ValueError(f"Unexpected event name: {event_name}")

    def calc_kernel_count(self, mode):
        return len(
            [
                kernel
                for kernel in self.kernels
                if (
                    kernel.run_mode == mode
                    and kernel.name not in KernelStatsVisitor.SKIP_KERNEL_NAMES
                )
            ]
        )

    def calc_kernel_duration(self, mode):
        return sum(
            [
                kernel.duration
                for kernel in self.kernels
                if (
                    kernel.run_mode == mode
                    and kernel.name not in KernelStatsVisitor.SKIP_KERNEL_NAMES
                )
            ]
        )

    def find_all_cuda_kernels(self, host_event):
        # TODO(SigureMo): Find a better way to find all CUDA kernels
        return [
            CudaKernelInfo(
                device_event.name, device_event.end_ns - device_event.start_ns
            )
            for runtime_event in host_event.runtime_node
            for device_event in runtime_event.device_node
        ]

    def visit_DygraphKernelLaunch(self, event_node):
        duration = event_node.end_ns - event_node.start_ns
        kernel_name = self.get_kernel_name(event_node.name)
        all_cuda_kernels = self.find_all_cuda_kernels(event_node)
        self.kernels.append(
            KernelInfo(
                kernel_name, KernelRunMode.Dygraph, duration, all_cuda_kernels
            )
        )
        self.generic_visit(event_node)

    def visit_StaticKernelLaunch(self, event_node):
        duration = event_node.end_ns - event_node.start_ns
        kernel_name = self.get_kernel_name(event_node.name)
        all_cuda_kernels = self.find_all_cuda_kernels(event_node)
        self.kernels.append(
            KernelInfo(
                kernel_name, KernelRunMode.Static, duration, all_cuda_kernels
            )
        )
        self.generic_visit(event_node)

    def summary(self) -> str:
        static_kernel_duration = self.calc_kernel_duration(KernelRunMode.Static)
        dygraph_kernel_duration = self.calc_kernel_duration(
            KernelRunMode.Dygraph
        )
        static_kernel_count = self.calc_kernel_count(KernelRunMode.Static)
        dygraph_kernel_count = self.calc_kernel_count(KernelRunMode.Dygraph)

        percentage_static_kernel_count = safe_divide(
            static_kernel_count, static_kernel_count + dygraph_kernel_count
        )
        percentage_static_kernel_duration = safe_divide(
            static_kernel_duration,
            static_kernel_duration + dygraph_kernel_duration,
        )
        step_summary = ""
        step_summary += f"dygraph kernel count: {dygraph_kernel_count}\n"
        step_summary += f"static kernel count: {static_kernel_count}\n"
        step_summary += f"percentage static kernel count: {percentage_static_kernel_count:.2%}\n"

        step_summary += f"dygraph kernel duration: {dygraph_kernel_duration / 1000000:.2f} ms\n"
        step_summary += f"static kernel duration: {static_kernel_duration / 1000000:.2f} ms\n"
        step_summary += f"percentage static kernel duration: {percentage_static_kernel_duration:.2%}\n"
        return step_summary


class SotStepProfilerGuard:
    EXPORT_CHROME_TRACING_PATH = "./sot-chrome-tracing/"
    STEP_CNT = 0
    LAST_INFO_SUMMARY = None

    def __init__(self, enable_kernel_stats=True, enable_chrome_tracing=False):
        self.enable_kernel_stats = enable_kernel_stats
        self.enable_chrome_tracing = enable_chrome_tracing
        self.started = False
        self.record_event = None
        self.summary = None

    def _kernel_stats(self, prof) -> str:
        kernel_stats_visitor = KernelStatsVisitor()
        kernel_stats_visitor(prof.profiler_result.get_data())
        return kernel_stats_visitor.summary()

    def collect_step_info_summary(self, prof) -> str:
        summary = ""
        if self.enable_kernel_stats:
            summary += self._kernel_stats(prof)
        if self.enable_chrome_tracing:
            # If you want to export chrome tracing, you can enable this flag
            # and view the tracing result in https://ui.perfetto.dev/#!/viewer
            profiler.export_chrome_tracing(
                SotStepProfilerGuard.EXPORT_CHROME_TRACING_PATH,
                f"step_{SotStepProfilerGuard.STEP_CNT:03d}",
            )(prof)
        return summary

    def on_trace_ready(self, prof):
        summary = self.collect_step_info_summary(prof)
        if SotStepProfilerGuard.STEP_CNT == 0:
            coldstart_title = f"SOT step profiler info summary (ColdStart, step#{SotStepProfilerGuard.STEP_CNT}):"
            coldstart_report = f"{coldstart_title}\n{summary}"
            print(coldstart_report)
            self.summary = coldstart_report
        else:
            warmup_title = f"SOT step profiler info summary (Warmup, step#{SotStepProfilerGuard.STEP_CNT}):"
            warmup_report = f"{warmup_title}\n{summary}"
            if SotStepProfilerGuard.LAST_INFO_SUMMARY is None:
                atexit.register(
                    lambda: print(SotStepProfilerGuard.LAST_INFO_SUMMARY)
                )
            SotStepProfilerGuard.LAST_INFO_SUMMARY = warmup_report
            self.summary = warmup_report

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
            assert self.profiler is not None
            self.profiler.stop()
            self.profiler = None  # Avoid to hold the profiler instance
        SotStepProfilerGuard.STEP_CNT += 1

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
