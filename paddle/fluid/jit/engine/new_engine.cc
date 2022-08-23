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

#include "paddle/fluid/jit/engine/new_engine.h"

#include "paddle/fluid/framework/block_desc.h"
// #include "paddle/fluid/framework/details/build_strategy.h"
// #include "paddle/fluid/framework/details/execution_strategy.h"
// #include "paddle/fluid/framework/ir/graph.h"
// #include "paddle/fluid/framework/parallel_executor.h"
#include "paddle/fluid/framework/new_executor/standalone_executor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace jit {

NewEngine::NewEngine(const std::shared_ptr<FunctionInfo> &info,
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

void NewEngine::CreateGraphAndPE() {
  //   framework::details::BuildStrategy build_strategy;
  //   build_strategy.fuse_bn_act_ops_ = true;
  //   auto execution_strategy = GetExecutionStrategy(place_);

  auto &program_desc = info_->ProgramDesc();

  //   const framework::BlockDesc &global_block = program_desc.Block(0);
  //   int64_t start_op_index = 0;
  //   int64_t end_op_index = static_cast<int64_t>(global_block.OpSize());

  //   graph_ = std::make_shared<Graph>(program_desc, start_op_index,
  //   end_op_index); inner_pe_ = std::make_shared<ParallelExecutor>(
  //       place_, &scope_, execution_strategy, build_strategy, graph_.get());
  //   inner_pe_->PrepareVariables(&scope_);
  //   inner_pe_->SkipMemoryReuse(/*scope_idx=*/0, info_->InputArgNames());

  inner_pe_ = std::make_shared<StandaloneExecutor>(place_, program_desc);
}

std::vector<Tensor> NewEngine::operator()(const std::vector<Tensor> &inputs) {
  auto dense_tensors = utils::ToDenseTensors(inputs);
  return utils::ToTensors(this->operator()(dense_tensors));
}

std::vector<DenseTensor> NewEngine::operator()(
    const std::vector<DenseTensor> &inputs) {
  utils::ShareIntoScope(info_->InputArgNames(), inputs, &scope_);

  // the latter can be moved to python side.
  auto &feed_names = info_->InputArgNames();
  auto &fetch_names = info_->OutputArgNames();
  paddle::framework::FetchList outs =
      inner_pe_->Run(&scope_, feed_names, fetch_names);
  scope_.DropKids();
  std::vector<DenseTensor> outputs;
  for (auto &item : outs) {
    auto &o = PADDLE_GET(DenseTensor, item);
    outputs.emplace_back(o);
  }
  return outputs;
}

const std::shared_ptr<FunctionInfo> &NewEngine::Info() const { return info_; }

}  // namespace jit
}  // namespace paddle
