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

#include "paddle/fluid/platform/profiler/dump/deserialization_reader.h"
#include "paddle/fluid/platform/profiler/dump/serialization_logger.h"
#include "paddle/fluid/platform/profiler/event_node.h"
#include "paddle/fluid/platform/profiler/event_python.h"

using paddle::platform::SerializationLogger;
using paddle::platform::DeserializationReader;
using paddle::platform::NodeTrees;
using paddle::platform::HostTraceEventNode;
using paddle::platform::CudaRuntimeTraceEventNode;
using paddle::platform::DeviceTraceEventNode;
using paddle::platform::HostTraceEvent;
using paddle::platform::RuntimeTraceEvent;
using paddle::platform::DeviceTraceEvent;
using paddle::platform::TracerEventType;
using paddle::platform::KernelEventInfo;
using paddle::platform::MemcpyEventInfo;
using paddle::platform::MemsetEventInfo;
using paddle::platform::ProfilerResult;

TEST(SerializationLoggerTest, dump_case0) {
  std::list<HostTraceEvent> host_events;
  std::list<RuntimeTraceEvent> runtime_events;
  std::list<DeviceTraceEvent> device_events;
  host_events.push_back(HostTraceEvent(std::string("dataloader#1"),
                                       TracerEventType::Dataloader, 1000, 10000,
                                       10, 10));
  host_events.push_back(HostTraceEvent(
      std::string("op1"), TracerEventType::Operator, 11000, 20000, 10, 10));
  host_events.push_back(HostTraceEvent(
      std::string("op2"), TracerEventType::Operator, 21000, 30000, 10, 10));
  host_events.push_back(HostTraceEvent(
      std::string("op3"), TracerEventType::Operator, 31000, 40000, 10, 11));
  runtime_events.push_back(RuntimeTraceEvent(std::string("cudalaunch1"), 15000,
                                             17000, 10, 10, 1, 0));
  runtime_events.push_back(RuntimeTraceEvent(std::string("cudalaunch2"), 25000,
                                             35000, 10, 10, 2, 0));
  runtime_events.push_back(RuntimeTraceEvent(std::string("cudalaunch3"), 33000,
                                             37000, 10, 11, 3, 0));
  runtime_events.push_back(RuntimeTraceEvent(std::string("cudaMemcpy1"), 18000,
                                             19000, 10, 10, 4, 0));
  runtime_events.push_back(RuntimeTraceEvent(std::string("cudaMemset1"), 38000,
                                             39000, 10, 11, 5, 0));
  device_events.push_back(
      DeviceTraceEvent(std::string("kernel1"), TracerEventType::Kernel, 40000,
                       55000, 0, 10, 10, 1, KernelEventInfo()));
  device_events.push_back(
      DeviceTraceEvent(std::string("kernel2"), TracerEventType::Kernel, 70000,
                       95000, 0, 10, 10, 2, KernelEventInfo()));
  device_events.push_back(
      DeviceTraceEvent(std::string("kernel3"), TracerEventType::Kernel, 60000,
                       65000, 0, 10, 11, 3, KernelEventInfo()));
  device_events.push_back(
      DeviceTraceEvent(std::string("memcpy1"), TracerEventType::Memcpy, 56000,
                       59000, 0, 10, 10, 4, MemcpyEventInfo()));
  device_events.push_back(
      DeviceTraceEvent(std::string("memset1"), TracerEventType::Memset, 66000,
                       69000, 0, 10, 11, 5, MemsetEventInfo()));
  SerializationLogger logger("test_serialization_logger_case0.pb");
  NodeTrees tree(host_events, runtime_events, device_events);
  std::map<uint64_t, std::vector<HostTraceEventNode*>> nodes =
      tree.Traverse(true);
  EXPECT_EQ(nodes[10].size(), 4u);
  EXPECT_EQ(nodes[11].size(), 2u);
  std::vector<HostTraceEventNode*> thread1_nodes = nodes[10];
  std::vector<HostTraceEventNode*> thread2_nodes = nodes[11];
  for (auto it = thread1_nodes.begin(); it != thread1_nodes.end(); it++) {
    if ((*it)->Name() == "root node") {
      EXPECT_EQ((*it)->GetChildren().size(), 3u);
    }
    if ((*it)->Name() == "op1") {
      EXPECT_EQ((*it)->GetChildren().size(), 0u);
      EXPECT_EQ((*it)->GetRuntimeTraceEventNodes().size(), 2u);
    }
  }
  for (auto it = thread2_nodes.begin(); it != thread2_nodes.end(); it++) {
    if ((*it)->Name() == "op3") {
      EXPECT_EQ((*it)->GetChildren().size(), 0u);
      EXPECT_EQ((*it)->GetRuntimeTraceEventNodes().size(), 2u);
    }
  }
  tree.LogMe(&logger);
}

TEST(SerializationLoggerTest, dump_case1) {
  std::list<HostTraceEvent> host_events;
  std::list<RuntimeTraceEvent> runtime_events;
  std::list<DeviceTraceEvent> device_events;
  runtime_events.push_back(RuntimeTraceEvent(std::string("cudalaunch1"), 15000,
                                             17000, 10, 10, 1, 0));
  runtime_events.push_back(RuntimeTraceEvent(std::string("cudalaunch2"), 25000,
                                             35000, 10, 10, 2, 0));
  runtime_events.push_back(RuntimeTraceEvent(std::string("cudalaunch3"), 33000,
                                             37000, 10, 11, 3, 0));
  runtime_events.push_back(RuntimeTraceEvent(std::string("cudaMemcpy1"), 18000,
                                             19000, 10, 10, 4, 0));
  runtime_events.push_back(RuntimeTraceEvent(std::string("cudaMemset1"), 38000,
                                             39000, 10, 11, 5, 0));
  device_events.push_back(
      DeviceTraceEvent(std::string("kernel1"), TracerEventType::Kernel, 40000,
                       55000, 0, 10, 10, 1, KernelEventInfo()));
  device_events.push_back(
      DeviceTraceEvent(std::string("kernel2"), TracerEventType::Kernel, 70000,
                       95000, 0, 10, 10, 2, KernelEventInfo()));
  device_events.push_back(
      DeviceTraceEvent(std::string("kernel3"), TracerEventType::Kernel, 60000,
                       65000, 0, 10, 11, 3, KernelEventInfo()));
  device_events.push_back(
      DeviceTraceEvent(std::string("memcpy1"), TracerEventType::Memcpy, 56000,
                       59000, 0, 10, 10, 4, MemcpyEventInfo()));
  device_events.push_back(
      DeviceTraceEvent(std::string("memset1"), TracerEventType::Memset, 66000,
                       69000, 0, 10, 11, 5, MemsetEventInfo()));
  SerializationLogger logger("test_serialization_logger_case1.pb");
  NodeTrees tree(host_events, runtime_events, device_events);
  std::map<uint64_t, std::vector<HostTraceEventNode*>> nodes =
      tree.Traverse(true);
  EXPECT_EQ(nodes[10].size(), 1u);
  EXPECT_EQ(nodes[11].size(), 1u);
  std::vector<HostTraceEventNode*> thread1_nodes = nodes[10];
  std::vector<HostTraceEventNode*> thread2_nodes = nodes[11];
  for (auto it = thread1_nodes.begin(); it != thread1_nodes.end(); it++) {
    if ((*it)->Name() == "root node") {
      EXPECT_EQ((*it)->GetRuntimeTraceEventNodes().size(), 3u);
    }
  }
  for (auto it = thread2_nodes.begin(); it != thread2_nodes.end(); it++) {
    if ((*it)->Name() == "root node") {
      EXPECT_EQ((*it)->GetChildren().size(), 0u);
      EXPECT_EQ((*it)->GetRuntimeTraceEventNodes().size(), 2u);
    }
  }
  tree.LogMe(&logger);
}

TEST(DeserializationReaderTest, restore_case0) {
  DeserializationReader reader("test_serialization_logger_case0.pb");
  auto profiler_result = reader.Parse();
  auto tree = profiler_result->GetNodeTrees();
  std::map<uint64_t, std::vector<HostTraceEventNode*>> nodes =
      tree->Traverse(true);
  EXPECT_EQ(nodes[10].size(), 4u);
  EXPECT_EQ(nodes[11].size(), 2u);
  std::vector<HostTraceEventNode*> thread1_nodes = nodes[10];
  std::vector<HostTraceEventNode*> thread2_nodes = nodes[11];
  for (auto it = thread1_nodes.begin(); it != thread1_nodes.end(); it++) {
    if ((*it)->Name() == "root node") {
      EXPECT_EQ((*it)->GetChildren().size(), 3u);
    }
    if ((*it)->Name() == "op1") {
      EXPECT_EQ((*it)->GetChildren().size(), 0u);
      EXPECT_EQ((*it)->GetRuntimeTraceEventNodes().size(), 2u);
    }
  }
  for (auto it = thread2_nodes.begin(); it != thread2_nodes.end(); it++) {
    if ((*it)->Name() == "op3") {
      EXPECT_EQ((*it)->GetChildren().size(), 0u);
      EXPECT_EQ((*it)->GetRuntimeTraceEventNodes().size(), 2u);
    }
  }
}

TEST(DeserializationReaderTest, restore_case1) {
  DeserializationReader reader("test_serialization_logger_case1.pb");
  auto profiler_result = reader.Parse();
  auto tree = profiler_result->GetNodeTrees();
  std::map<uint64_t, std::vector<HostTraceEventNode*>> nodes =
      tree->Traverse(true);
  EXPECT_EQ(nodes[10].size(), 1u);
  EXPECT_EQ(nodes[11].size(), 1u);
  std::vector<HostTraceEventNode*> thread1_nodes = nodes[10];
  std::vector<HostTraceEventNode*> thread2_nodes = nodes[11];
  for (auto it = thread1_nodes.begin(); it != thread1_nodes.end(); it++) {
    if ((*it)->Name() == "root node") {
      EXPECT_EQ((*it)->GetRuntimeTraceEventNodes().size(), 3u);
    }
  }
  for (auto it = thread2_nodes.begin(); it != thread2_nodes.end(); it++) {
    if ((*it)->Name() == "root node") {
      EXPECT_EQ((*it)->GetChildren().size(), 0u);
      EXPECT_EQ((*it)->GetRuntimeTraceEventNodes().size(), 2u);
    }
  }
}
