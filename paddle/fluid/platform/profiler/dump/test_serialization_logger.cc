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
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/platform/profiler/dump/deserialization_reader.h"
#include "paddle/fluid/platform/profiler/dump/serialization_logger.h"
#include "paddle/fluid/platform/profiler/event_node.h"
#include "paddle/fluid/platform/profiler/event_python.h"

using paddle::framework::AttributeMap;
using paddle::platform::DeserializationReader;
using paddle::platform::DeviceTraceEvent;
using paddle::platform::HostTraceEvent;
using paddle::platform::HostTraceEventNode;
using paddle::platform::KernelEventInfo;
using paddle::platform::MemcpyEventInfo;
using paddle::platform::MemsetEventInfo;
using paddle::platform::MemTraceEvent;
using paddle::platform::NodeTrees;
using paddle::platform::OperatorSupplementEvent;
using paddle::platform::RuntimeTraceEvent;
using paddle::platform::SerializationLogger;
using phi::TracerEventType;
using phi::TracerMemEventType;

TEST(SerializationLoggerTest, dump_case0) {
  std::list<HostTraceEvent> host_events;
  std::list<RuntimeTraceEvent> runtime_events;
  std::list<DeviceTraceEvent> device_events;
  std::list<MemTraceEvent> mem_events;
  std::list<OperatorSupplementEvent> op_supplement_events;
  host_events.emplace_back(std::string("dataloader#1"),
                           TracerEventType::Dataloader,
                           1000,
                           10000,
                           10,
                           10);
  host_events.emplace_back(
      std::string("op1"), TracerEventType::Operator, 11000, 20000, 10, 10);
  host_events.emplace_back(
      std::string("op2"), TracerEventType::Operator, 21000, 30000, 10, 10);
  host_events.emplace_back(
      std::string("op3"), TracerEventType::Operator, 31000, 40000, 10, 11);
  mem_events.emplace_back(11500,
                          0x1000,
                          TracerMemEventType::Allocate,
                          10,
                          10,
                          50,
                          "GPU:0",
                          50,
                          50,
                          100,
                          100);
  mem_events.emplace_back(11900,
                          0x1000,
                          TracerMemEventType::Free,
                          10,
                          10,
                          -50,
                          "GPU:0",
                          0,
                          50,
                          100,
                          100);
  std::map<std::string, std::vector<std::vector<int64_t>>> input_shapes;
  std::map<std::string, std::vector<std::string>> dtypes;
  input_shapes[std::string("X")].push_back(std::vector<int64_t>{1, 2, 3});
  input_shapes[std::string("X")].push_back(std::vector<int64_t>{4, 5, 6, 7});
  dtypes[std::string("X")].emplace_back("int8");
  dtypes[std::string("X")].emplace_back("float32");
  AttributeMap attrs;
  op_supplement_events.emplace_back(
      11600, "op1", input_shapes, dtypes, "op1()", attrs, 0, 10, 10);
  runtime_events.emplace_back(
      std::string("cudalaunch1"), 15000, 17000, 10, 10, 1, 0);
  runtime_events.emplace_back(
      std::string("cudalaunch2"), 25000, 35000, 10, 10, 2, 0);
  runtime_events.emplace_back(
      std::string("cudalaunch3"), 33000, 37000, 10, 11, 3, 0);
  runtime_events.emplace_back(
      std::string("cudaMemcpy1"), 18000, 19000, 10, 10, 4, 0);
  runtime_events.emplace_back(
      std::string("cudaMemset1"), 38000, 39000, 10, 11, 5, 0);
  device_events.emplace_back(std::string("kernel1"),
                             TracerEventType::Kernel,
                             40000,
                             55000,
                             0,
                             10,
                             10,
                             1,
                             KernelEventInfo());
  device_events.emplace_back(std::string("kernel2"),
                             TracerEventType::Kernel,
                             70000,
                             95000,
                             0,
                             10,
                             10,
                             2,
                             KernelEventInfo());
  device_events.emplace_back(std::string("kernel3"),
                             TracerEventType::Kernel,
                             60000,
                             65000,
                             0,
                             10,
                             11,
                             3,
                             KernelEventInfo());
  device_events.emplace_back(std::string("memcpy1"),
                             TracerEventType::Memcpy,
                             56000,
                             59000,
                             0,
                             10,
                             10,
                             4,
                             MemcpyEventInfo());
  device_events.emplace_back(std::string("memset1"),
                             TracerEventType::Memset,
                             66000,
                             69000,
                             0,
                             10,
                             11,
                             5,
                             MemsetEventInfo());
  SerializationLogger logger("test_serialization_logger_case0.pb");
  logger.LogMetaInfo(std::string("1.0.2"), 0);
  NodeTrees tree(host_events,
                 runtime_events,
                 device_events,
                 mem_events,
                 op_supplement_events);
  std::map<uint64_t, std::vector<HostTraceEventNode*>> nodes =
      tree.Traverse(true);
  EXPECT_EQ(nodes[10].size(), 4u);
  EXPECT_EQ(nodes[11].size(), 2u);
  std::vector<HostTraceEventNode*> thread1_nodes = nodes[10];
  std::vector<HostTraceEventNode*> thread2_nodes = nodes[11];
  for (auto& thread1_node : thread1_nodes) {
    if (thread1_node->Name() == "root node") {
      EXPECT_EQ(thread1_node->GetChildren().size(), 3u);
    }
    if (thread1_node->Name() == "op1") {
      EXPECT_EQ(thread1_node->GetChildren().size(), 0u);
      EXPECT_EQ(thread1_node->GetRuntimeTraceEventNodes().size(), 2u);
      EXPECT_EQ(thread1_node->GetMemTraceEventNodes().size(), 2u);
      EXPECT_NE(thread1_node->GetOperatorSupplementEventNode(), nullptr);
    }
  }
  for (auto& thread2_node : thread2_nodes) {
    if (thread2_node->Name() == "op3") {
      EXPECT_EQ(thread2_node->GetChildren().size(), 0u);
      EXPECT_EQ(thread2_node->GetRuntimeTraceEventNodes().size(), 2u);
    }
  }
  tree.LogMe(&logger);
  logger.LogExtraInfo(std::unordered_map<std::string, std::string>());
}

TEST(SerializationLoggerTest, dump_case1) {
  std::list<HostTraceEvent> host_events;
  std::list<RuntimeTraceEvent> runtime_events;
  std::list<DeviceTraceEvent> device_events;
  std::list<MemTraceEvent> mem_events;
  std::list<OperatorSupplementEvent> op_supplement_events;
  runtime_events.emplace_back(
      std::string("cudalaunch1"), 15000, 17000, 10, 10, 1, 0);
  runtime_events.emplace_back(
      std::string("cudalaunch2"), 25000, 35000, 10, 10, 2, 0);
  runtime_events.emplace_back(
      std::string("cudalaunch3"), 33000, 37000, 10, 11, 3, 0);
  runtime_events.emplace_back(
      std::string("cudaMemcpy1"), 18000, 19000, 10, 10, 4, 0);
  runtime_events.emplace_back(
      std::string("cudaMemset1"), 38000, 39000, 10, 11, 5, 0);
  device_events.emplace_back(std::string("kernel1"),
                             TracerEventType::Kernel,
                             40000,
                             55000,
                             0,
                             10,
                             10,
                             1,
                             KernelEventInfo());
  device_events.emplace_back(std::string("kernel2"),
                             TracerEventType::Kernel,
                             70000,
                             95000,
                             0,
                             10,
                             10,
                             2,
                             KernelEventInfo());
  device_events.emplace_back(std::string("kernel3"),
                             TracerEventType::Kernel,
                             60000,
                             65000,
                             0,
                             10,
                             11,
                             3,
                             KernelEventInfo());
  device_events.emplace_back(std::string("memcpy1"),
                             TracerEventType::Memcpy,
                             56000,
                             59000,
                             0,
                             10,
                             10,
                             4,
                             MemcpyEventInfo());
  device_events.emplace_back(std::string("memset1"),
                             TracerEventType::Memset,
                             66000,
                             69000,
                             0,
                             10,
                             11,
                             5,
                             MemsetEventInfo());
  SerializationLogger logger("test_serialization_logger_case1.pb");
  logger.LogMetaInfo(std::string("1.0.2"), 0);
  NodeTrees tree(host_events,
                 runtime_events,
                 device_events,
                 mem_events,
                 op_supplement_events);
  std::map<uint64_t, std::vector<HostTraceEventNode*>> nodes =
      tree.Traverse(true);
  EXPECT_EQ(nodes[10].size(), 1u);
  EXPECT_EQ(nodes[11].size(), 1u);
  std::vector<HostTraceEventNode*> thread1_nodes = nodes[10];
  std::vector<HostTraceEventNode*> thread2_nodes = nodes[11];
  for (auto& thread1_node : thread1_nodes) {
    if (thread1_node->Name() == "root node") {
      EXPECT_EQ(thread1_node->GetRuntimeTraceEventNodes().size(), 3u);
    }
  }
  for (auto& thread2_node : thread2_nodes) {
    if (thread2_node->Name() == "root node") {
      EXPECT_EQ(thread2_node->GetChildren().size(), 0u);
      EXPECT_EQ(thread2_node->GetRuntimeTraceEventNodes().size(), 2u);
    }
  }
  tree.LogMe(&logger);
  logger.LogExtraInfo(std::unordered_map<std::string, std::string>());
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
  for (auto& thread1_node : thread1_nodes) {
    if (thread1_node->Name() == "root node") {
      EXPECT_EQ(thread1_node->GetChildren().size(), 3u);
    }
    if (thread1_node->Name() == "op1") {
      EXPECT_EQ(thread1_node->GetChildren().size(), 0u);
      EXPECT_EQ(thread1_node->GetRuntimeTraceEventNodes().size(), 2u);
      EXPECT_EQ(thread1_node->GetMemTraceEventNodes().size(), 2u);
      EXPECT_NE(thread1_node->GetOperatorSupplementEventNode(), nullptr);
    }
  }
  for (auto& thread2_node : thread2_nodes) {
    if (thread2_node->Name() == "op3") {
      EXPECT_EQ(thread2_node->GetChildren().size(), 0u);
      EXPECT_EQ(thread2_node->GetRuntimeTraceEventNodes().size(), 2u);
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
  for (auto& thread1_node : thread1_nodes) {
    if (thread1_node->Name() == "root node") {
      EXPECT_EQ(thread1_node->GetRuntimeTraceEventNodes().size(), 3u);
    }
  }
  for (auto& thread2_node : thread2_nodes) {
    if (thread2_node->Name() == "root node") {
      EXPECT_EQ(thread2_node->GetChildren().size(), 0u);
      EXPECT_EQ(thread2_node->GetRuntimeTraceEventNodes().size(), 2u);
    }
  }
}
