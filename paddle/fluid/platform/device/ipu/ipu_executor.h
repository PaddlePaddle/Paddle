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

#include <mutex>

#include <model_runtime/ModelRunner.hpp>
#include <model_runtime/Tensor.hpp>
#include <popart/dataflow.hpp>
#include <popart/half.hpp>
#include <popart/names.hpp>
#include <popart/patterns/patterns.hpp>
#include <popart/session.hpp>
#include <popart/stepio.hpp>
#include <popart/tensorinfo.hpp>

#include "paddle/fluid/platform/device/ipu/ipu_utils.h"

namespace paddle {
namespace framework {
class ExecutionContext;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace platform {
namespace ipu {

struct CompilerResources;
class IpuStrategy;

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

  // Build popart session
  void Prepare(const std::string &proto);

  // Run popart session
  void Run(const std::vector<const Tensor *> &inputs,
           const std::vector<Tensor *> &outputs,
           const framework::ExecutionContext &ctx);

  // Run popef session
  void RunPopef(const std::vector<const Tensor *> &inputs,
                const std::vector<Tensor *> &outputs,
                const framework::ExecutionContext &ctx);

  // Sync weights from popart to paddle
  void WeightsToHost();

  // Detach IPU
  void Detach();

  // Reset session
  void Reset();

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
  void PreparePopefSession();
  void ResetPopef();

 private:
  // Not own
  const Scope *scope_ = nullptr;
  const IpuStrategy *ipu_strategy_ = nullptr;
  CompilerResources *compiler_resources_ = nullptr;
  model_runtime::QueueManager *queue_manager_ = nullptr;

  // Deviceinfo for popart session
  std::shared_ptr<popart::DeviceInfo> device_;
  // Popart session, where graph running
  std::unique_ptr<popart::Session> session_;
  // A ExecutorResources corresponds to a graph
  std::unique_ptr<ExecutorResources> executor_resources_;
  // mutex to lock session run.
  mutable std::mutex mutex_;
  // popef session
  std::unique_ptr<model_runtime::Session> popef_session_;
  // popef execution threads
  std::thread main_program_;
  // mutex to lock popef queue
  std::mutex queue_mutex_;

  bool compile_only_ = false;
  // indicate if popart/popef is used. Do not use the ipu_strategy which may
  // deconstruct before executor and cause undefined behavior.
  bool enable_model_runtime_executor_ = false;
  std::atomic_bool stop_ = {false};
};

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
