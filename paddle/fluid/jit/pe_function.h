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

#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/details/build_strategy.h"
#include "paddle/fluid/framework/details/execution_strategy.h"
#include "paddle/fluid/framework/executor_cache.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/phi/core/enforce.h"

#include "paddle/fluid/jit/base_function.h"
#include "paddle/fluid/jit/function_schema.h"
#include "paddle/fluid/jit/function_utils.h"

namespace paddle {
namespace jit {

using ExecutionStrategy = framework::details::ExecutionStrategy;
using ParallelExecutor = framework::ParallelExecutor;
using Graph = framework::ir::Graph;

class PEFunction : public BaseFunction {
 public:
  PEFunction(const std::shared_ptr<FunctionInfo> &info,
             const Name2VariableMap &params_dict,
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

  ~PEFunction() noexcept {}

  static ExecutionStrategy GetExecutionStrategy(const platform::Place &place) {
    ExecutionStrategy execution_strategy;

    auto device_type = platform::Place2DeviceType(place);
    switch (device_type) {
      case platform::DeviceType::CPU: {
        execution_strategy.num_threads_ = 2;
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
        PADDLE_THROW(platform::errors::Unavailable(
            "Unsupported Device type %d.", device_type));
    }
    execution_strategy.use_device_ = device_type;

    return execution_strategy;
  }

  void CreateGraphAndPE() {
    framework::details::BuildStrategy build_strategy;
    auto execution_strategy = GetExecutionStrategy(place_);

    auto &program_desc = info_->ProgramDesc();
    const framework::BlockDesc &global_block = program_desc.Block(0);
    int64_t start_op_index = 0;
    int64_t end_op_index = static_cast<int64_t>(global_block.OpSize());

    PADDLE_ENFORCE_GT(end_op_index,
                      start_op_index,
                      platform::errors::PreconditionNotMet(
                          "There is no operator in ProgramDesc."));

    graph_ =
        std::make_shared<Graph>(program_desc, start_op_index, end_op_index);
    inner_pe_ = std::make_shared<ParallelExecutor>(
        place_, &scope_, execution_strategy, build_strategy, graph_.get());
    inner_pe_->PrepareVariables(&scope_);
    inner_pe_->SkipMemoryReuse(/*scope_idx=*/0, info_->InputArgNames());
  }

  std::vector<Tensor> operator()(const std::vector<Tensor> &inputs) {
    auto dense_tensors = utils::ToDenseTensors(inputs);
    return utils::ToTensors(this->operator()(dense_tensors));
  }

  std::vector<DenseTensor> operator()(const std::vector<DenseTensor> &inputs) {
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

  const std::shared_ptr<FunctionInfo> &Info() const { return info_; }

 private:
  std::shared_ptr<FunctionInfo> info_;
  framework::Scope scope_;
  phi::Place place_;
  std::shared_ptr<ParallelExecutor> inner_pe_;
  std::shared_ptr<Graph> graph_;
};

}  // namespace jit
}  // namespace paddle
