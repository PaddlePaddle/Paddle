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

#include <popart/dataflow.hpp>
#include <popart/half.hpp>
#include <popart/names.hpp>
#include <popart/patterns/patterns.hpp>
#include <popart/session.hpp>
#include <popart/tensorinfo.hpp>
#include <popdist/popdist_poplar.hpp>

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/device/ipu/ipu_compiler.h"
#include "paddle/fluid/platform/device/ipu/ipu_names.h"
#include "paddle/fluid/platform/device/ipu/ipu_strategy.h"
#include "paddle/fluid/platform/device/ipu/ipu_utils.h"

namespace paddle {
namespace platform {
namespace ipu {

struct ExecutorResources {
  // map<tensor_id, paddle_var_ptr>
  popart::WeightsIO weights_io;
  // <popart_var, paddle_var> pairs, include weights and optimizer states
  std::vector<std::pair<popart::TensorId, std::string>> weights_and_opt_state;
};

class Executor {
 public:
  Executor() = default;
  ~Executor();

  // build popart session
  void Prepare(const std::string &proto);

  // run popart session
  void Run(const std::vector<const Tensor *> &inputs,
           const std::vector<Tensor *> &outputs,
           const framework::ExecutionContext &ctx);

  // sync weights from popart to paddle
  void WeightsToHost();

  // detach IPU
  void Detach();

  // Scope
  void SetScope(const Scope *scope) { scope_ = scope; }

  // Strategy
  void SetIpuStrategy(const IpuStrategy &strategy) {
    ipu_strategy_ = &strategy;
  }

  // CompilerResources
  void SetCompilerResources(CompilerResources *resources) {
    compiler_resources_ = resources;
  }

  // Save model to onnx
  void SaveModelToHost(const std::string &path);

 private:
  void AcquireDevice();
  void SetWeightsIO();
  void ConvertWeights(bool);
  void WeightsFromPaddle();
  void WeightsToPaddle();

 private:
  // not own
  const Scope *scope_ = nullptr;
  const IpuStrategy *ipu_strategy_ = nullptr;
  CompilerResources *compiler_resources_ = nullptr;

  // deviceinfo for popart session
  std::shared_ptr<popart::DeviceInfo> device_;
  // popart session, where graph running
  std::unique_ptr<popart::Session> session_;
  // one OneSession means a graph
  std::unique_ptr<ExecutorResources> executor_resources_;
};

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
