/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <popart/devicemanager.hpp>
#include <popart/names.hpp>
#include <popart/tensorinfo.hpp>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/device/ipu/ipu_strategy.h"
#include "paddle/phi/core/platform/timer.h"

namespace paddle {
namespace framework {
class ExecutionContext;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace platform {
namespace ipu {

class IpuStrategy;
class Compiler;
class Executor;

class IpuBackend {
 public:
  static IpuBackend *GetInstance();

 public:
  IpuBackend();
  ~IpuBackend();

  // What compile method does:
  // Convert paddle ops to popart ops;
  // Construct a popart graph, which is a onnx compute graph;
  // Load the graph and weights to ipu.
  void Compile(framework::ir::Graph *graph,
               const std::vector<std::string> &feed_list,
               const std::vector<std::string> &fetch_list);

  // Run the compiled graph on ipu
  void Run(const std::vector<const phi::DenseTensor *> &inputs,
           const std::vector<phi::DenseTensor *> &outputs,
           const framework::ExecutionContext &ctx);

  // Sync weights from IPU while training
  void WeightsToHost();

  // Detach IPU manually
  void Detach();

  // Reset manually
  // Call it before destruct works
  void Reset();

  void SetScope(const framework::Scope &scope);
  const framework::Scope *GetScope() { return scope_; }
  void SetIpuStrategy(const IpuStrategy &strategy);
  const IpuStrategy *GetIpuStrategy() { return ipu_strategy_; }

  // Save compiled model to onnx
  void SaveModelProto(const std::string &path);

 private:
  // Not own
  const framework::Scope *scope_ = nullptr;
  const IpuStrategy *ipu_strategy_ = nullptr;

  // Own
  std::unique_ptr<Compiler> compiler_;
  std::unique_ptr<Executor> executor_;
  std::unique_ptr<Timer> timer_;

  bool is_compiled_ = false;

  DISABLE_COPY_AND_ASSIGN(IpuBackend);
};

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
