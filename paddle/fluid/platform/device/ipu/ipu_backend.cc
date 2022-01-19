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
#include "paddle/fluid/platform/device/ipu/ipu_utils.h"

#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/node.h"

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
}

IpuBackend::~IpuBackend() {
  compiler_.reset();
  executor_.reset();
}

void IpuBackend::Compile(Graph* graph,
                         const std::vector<std::string>& feed_list,
                         const std::vector<std::string>& fetch_list) {
  VLOG(10) << "enter IpuBackend::Compile";
  compiler_->Prepare();
  executor_->SetCompilerResources(compiler_->GetResources());

  compiler_->InitInputs(graph, feed_list);
  compiler_->LowerConstants(graph, scope_);
  compiler_->LowerWeights(graph, scope_);
  compiler_->LowerBody(graph);
  compiler_->InitOutputs(fetch_list);
  if (ipu_strategy_->is_training) {
    compiler_->LowerOptimier(graph, scope_);
  }
  is_compiled_ = true;
  // when call compile, means a new graph
  is_prepared_ = false;
  VLOG(10) << "leave IpuBackend::Compile";
}

void IpuBackend::Run(const std::vector<const Tensor*>& inputs,
                     const std::vector<Tensor*>& outputs,
                     const framework::ExecutionContext& ctx) {
  Prepare();
  timer_->Start();
  executor_->Run(inputs, outputs, ctx);
  timer_->Pause();
  VLOG(10) << "[IPU Run]: " << timer_->ElapsedMS() << " (ms)";
}

void IpuBackend::Prepare() {
  if (!is_prepared_) {
    executor_->Prepare(compiler_->GetModelProto());
    timer_.reset(new platform::Timer());
    is_prepared_ = true;
  }
}

void IpuBackend::Detach() { executor_->Detach(); }

void IpuBackend::Reset() {
  executor_->Detach();
  compiler_.reset();
  executor_.reset();
}

void IpuBackend::SetScope(const Scope& scope) {
  scope_ = &scope;
  executor_->SetScope(&scope);
}

void IpuBackend::SetIpuStrategy(const IpuStrategy& strategy) {
  ipu_strategy_ = &strategy;
  compiler_->SetIpuStrategy(strategy);
  executor_->SetIpuStrategy(strategy);
}

void IpuBackend::SetCustomOps(
    const std::vector<IpuCustomOpIdentifier>& custom_ops) {
  compiler_->SetCustomOps(custom_ops);
}

void IpuBackend::SaveMoldeProto(const std::string& path) {
  if (ipu_strategy_->is_training && is_prepared_) {
    executor_->SaveModelToHost(path);
  } else if (is_compiled_) {
    compiler_->SaveModelProtoNoCheck(path);
  } else {
    LOG(WARNING) << "Model is empty";
  }
}

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
