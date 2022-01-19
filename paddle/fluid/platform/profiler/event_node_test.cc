// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "gtest/gtest.h"

#include "paddle/fluid/platform/profiler/chrometracing_logger.h"
#include "paddle/fluid/platform/profiler/event_node.h"

using paddle::platform::ChromeTracingLogger;
using paddle::platform::NodeTrees;
using paddle::platform::HostTraceEventNode;
using paddle::platform::CudaRuntimeTraceEventNode;
using paddle::platform::DeviceTraceEventNode;
using paddle::platform::HostTraceEvent;
using paddle::platform::RuntimeTraceEvent;
using paddle::platform::DeviceTraceEvent;
using paddle::platform::TracerEventType;
using paddle::platform::KernelEventInfo;
TEST(test_node_tree, nodetreelog) {
  std::list<HostTraceEvent> host_events;
  std::list<RuntimeTraceEvent> runtime_events;
  std::list<DeviceTraceEvent> device_events;
  host_events.push_back(HostTraceEvent(
      std::string("op1"), TracerEventType::Operator, 10, 100, 10, 10));
  host_events.push_back(HostTraceEvent(
      std::string("op2"), TracerEventType::Operator, 30, 70, 10, 10));
  host_events.push_back(HostTraceEvent(
      std::string("op3"), TracerEventType::Operator, 2, 120, 10, 11));
  runtime_events.push_back(
      RuntimeTraceEvent(std::string("cudalaunch1"), 15, 25, 10, 10, 1, 0));
  runtime_events.push_back(
      RuntimeTraceEvent(std::string("cudalaunch2"), 35, 45, 10, 10, 2, 0));
  runtime_events.push_back(
      RuntimeTraceEvent(std::string("cudalaunch3"), 2, 55, 10, 11, 3, 0));
  device_events.push_back(DeviceTraceEvent(std::string("kernel1"),
                                           TracerEventType::Kernel, 40, 55, 10,
                                           10, 10, 1, KernelEventInfo()));
  device_events.push_back(DeviceTraceEvent(std::string("kernel2"),
                                           TracerEventType::Kernel, 70, 95, 10,
                                           10, 10, 2, KernelEventInfo()));
  device_events.push_back(DeviceTraceEvent(std::string("kernel3"),
                                           TracerEventType::Kernel, 60, 75, 10,
                                           10, 11, 3, KernelEventInfo()));
  ChromeTracingLogger logger("testlog.json");
  NodeTrees tree(host_events, runtime_events, device_events);
  tree.LogMe(&logger);
}

TEST(test_node_tree, nodetreehandle) {
  std::list<HostTraceEvent> host_events;
  std::list<RuntimeTraceEvent> runtime_events;
  std::list<DeviceTraceEvent> device_events;
  host_events.push_back(HostTraceEvent(
      std::string("op1"), TracerEventType::Operator, 10, 100, 10, 10));
  host_events.push_back(HostTraceEvent(
      std::string("op2"), TracerEventType::Operator, 30, 70, 10, 10));
  host_events.push_back(HostTraceEvent(
      std::string("op3"), TracerEventType::Operator, 2, 120, 10, 11));
  runtime_events.push_back(
      RuntimeTraceEvent(std::string("cudalaunch1"), 15, 25, 10, 10, 1, 0));
  runtime_events.push_back(
      RuntimeTraceEvent(std::string("cudalaunch2"), 35, 45, 10, 10, 2, 0));
  runtime_events.push_back(
      RuntimeTraceEvent(std::string("cudalaunch3"), 2, 55, 10, 11, 3, 0));
  device_events.push_back(DeviceTraceEvent(std::string("kernel1"),
                                           TracerEventType::Kernel, 40, 55, 10,
                                           10, 10, 1, KernelEventInfo()));
  device_events.push_back(DeviceTraceEvent(std::string("kernel2"),
                                           TracerEventType::Kernel, 70, 95, 10,
                                           10, 10, 2, KernelEventInfo()));
  device_events.push_back(DeviceTraceEvent(std::string("kernel3"),
                                           TracerEventType::Kernel, 60, 75, 10,
                                           10, 10, 3, KernelEventInfo()));
  ChromeTracingLogger logger("testlog_handle.json");
  NodeTrees tree(host_events, runtime_events, device_events);
  std::function<void(HostTraceEventNode*)> host_event_node_handle(
      [&](HostTraceEventNode* a) { logger.LogHostTraceEventNode(*a); });
  std::function<void(CudaRuntimeTraceEventNode*)> runtime_event_node_handle([&](
      CudaRuntimeTraceEventNode* a) { logger.LogRuntimeTraceEventNode(*a); });
  std::function<void(DeviceTraceEventNode*)> device_event_node_handle(
      [&](DeviceTraceEventNode* a) { logger.LogDeviceTraceEventNode(*a); });
  tree.HandleTrees(host_event_node_handle, runtime_event_node_handle,
                   device_event_node_handle);
}
