// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/jit/engine/pe_engine.h"

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/details/build_strategy.h"
#include "paddle/fluid/framework/details/execution_strategy.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/parallel_executor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace jit {

static ExecutionStrategy GetExecutionStrategy(const platform::Place &place) {
  ExecutionStrategy execution_strategy;

  auto device_type = platform::Place2DeviceType(place);
  switch (device_type) {
    case platform::DeviceType::CPU: {
      execution_strategy.num_threads_ = 1;
      break;
    }
    case platform::DeviceType::CUDA: {
      // NOTE: According experiments, one thread is faster in
      // most model training.
      execution_strategy.num_threads_ = 1;
      break;
    }
    case platform::DeviceType::XPU: {
      execution_strategy.num_threads_ = 1;
      break;
    }
    case platform::DeviceType::IPU: {
      execution_strategy.num_threads_ = 1;
      break;
    }
    default:
      PADDLE_THROW(platform::errors::Unavailable("Unsupported Device type %d.",
                                                 device_type));
  }
  execution_strategy.use_device_ = device_type;

  return execution_strategy;
}

PEEngine::PEEngine(const std::shared_ptr<FunctionInfo> &info,
                   const VariableMap &params_dict,
                   const phi::Place &place)
    : info_(info), place_(place) {
  info_->RemoveDescFeedFetch();
  PADDLE_ENFORCE_GT(
      static_cast<int64_t>(info_->ProgramDesc().Block(0).OpSize()),
      0,
      platform::errors::PreconditionNotMet(
          "There is no operator in ProgramDesc."));
  utils::ShareParamsIntoScope(info_->ParamNames(), params_dict, &scope_);
  VLOG(6) << framework::GenScopeTreeDebugInfo(&scope_);
  CreateGraphAndPE();
}

void PEEngine::CreateGraphAndPE() {
  framework::details::BuildStrategy build_strategy;
  build_strategy.enable_inference_pass_ = true;  // use pe to inference
  auto execution_strategy = GetExecutionStrategy(place_);

  auto &program_desc = info_->ProgramDesc();
  const framework::BlockDesc &global_block = program_desc.Block(0);
  int64_t start_op_index = 0;
  int64_t end_op_index = static_cast<int64_t>(global_block.OpSize());

  graph_ = std::make_shared<Graph>(program_desc, start_op_index, end_op_index);
  inner_pe_ = std::make_shared<ParallelExecutor>(
      place_, &scope_, execution_strategy, build_strategy, graph_.get());
  inner_pe_->PrepareVariables(&scope_);
  inner_pe_->SkipMemoryReuse(/*scope_idx=*/0, info_->InputArgNames());
}

std::vector<Tensor> PEEngine::operator()(const std::vector<Tensor> &inputs) {
  auto dense_tensors = utils::ToDenseTensors(inputs);
  return utils::ToTensors(this->operator()(dense_tensors));
}

std::vector<DenseTensor> PEEngine::operator()(
    const std::vector<DenseTensor> &inputs) {
  utils::ShareIntoScope(info_->InputArgNames(), inputs, &scope_);

  // update op_handle scope_map in pe->executor_->Graph
  std::unordered_map<framework::Scope *, framework::Scope *> scope_map = {
      {inner_pe_->GetLocalScopes().front(), &scope_}};
  inner_pe_->ResetOpHandleScopeMapOfGraphs(scope_map);
  // need to recreate tmp variables in new scope
  inner_pe_->PrepareVariables(&scope_);

  inner_pe_->RunWithoutFetch(info_->OutputArgNames());

  std::vector<DenseTensor> outputs;
  utils::FetchOuts(info_->OutputArgNames(), scope_, &outputs);
  scope_.DropKids();
  return outputs;
}

const std::shared_ptr<FunctionInfo> &PEEngine::Info() const { return info_; }

}  // namespace jit
}  // namespace paddle
