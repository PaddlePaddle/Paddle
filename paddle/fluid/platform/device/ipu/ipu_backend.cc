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

#include "paddle/fluid/platform/device/ipu/ipu_backend.h"

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/device/ipu/ipu_compiler.h"
#include "paddle/fluid/platform/device/ipu/ipu_executor.h"

namespace paddle {
namespace platform {
namespace ipu {

IpuBackend* IpuBackend::GetInstance() {
  static IpuBackend instance;
  return &instance;
}

IpuBackend::IpuBackend() {
  compiler_ = std::make_unique<Compiler>();
  executor_ = std::make_unique<Executor>();
  timer_ = std::make_unique<platform::Timer>();
}

IpuBackend::~IpuBackend() {
  compiler_.reset();
  executor_.reset();
}

void IpuBackend::Compile(framework::ir::Graph* graph,
                         const std::vector<std::string>& feed_list,
                         const std::vector<std::string>& fetch_list) {
  VLOG(10) << "enter IpuBackend::Compile";
  is_compiled_ = false;
  compiler_->Prepare(graph);
  compiler_->InitInputs(feed_list);
  compiler_->LowerConstants(scope_);
  compiler_->LowerWeights(scope_);
  compiler_->LowerBody();
  compiler_->InitOutputs(fetch_list);
  if (ipu_strategy_->is_training) {
    compiler_->LowerOptimizer(scope_);
  }
  if (!ipu_strategy_->onnx_dump_path.empty()) {
    SaveModelProto(ipu_strategy_->onnx_dump_path);
  }
  executor_->SetCompilerResources(compiler_->GetResources());
  executor_->Prepare(compiler_->GetModelProto());
  is_compiled_ = true;
  VLOG(10) << "leave IpuBackend::Compile";
}

void IpuBackend::Run(const std::vector<const phi::DenseTensor*>& inputs,
                     const std::vector<phi::DenseTensor*>& outputs,
                     const framework::ExecutionContext& ctx) {
  timer_->Start();
  executor_->Run(inputs, outputs, ctx);
  timer_->Pause();
  VLOG(10) << "[IPU Run]: " << timer_->ElapsedMS() << " (ms)";
}

void IpuBackend::WeightsToHost() { executor_->WeightsToHost(); }

void IpuBackend::Detach() { executor_->Detach(); }

void IpuBackend::Reset() { executor_->Reset(); }

void IpuBackend::SetScope(const framework::Scope& scope) {
  scope_ = &scope;
  executor_->SetScope(&scope);
}

void IpuBackend::SetIpuStrategy(const IpuStrategy& strategy) {
  ipu_strategy_ = &strategy;
  compiler_->SetIpuStrategy(strategy);
  executor_->SetIpuStrategy(strategy);
  if (!strategy.custom_ops.empty()) {
    compiler_->SetCustomOps(strategy.custom_ops);
  }
}

void IpuBackend::SaveModelProto(const std::string& path) {
  if (ipu_strategy_->is_training && is_compiled_) {
    executor_->SaveModelToHost(path);
  } else {
    compiler_->SaveModelProtoNoCheck(path);
  }
}

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
