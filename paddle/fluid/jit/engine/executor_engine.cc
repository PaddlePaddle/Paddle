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

#include "paddle/fluid/jit/engine/executor_engine.h"

#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace jit {

ExecutorEngine::ExecutorEngine(const std::shared_ptr<FunctionInfo> &info,
                               const VariableMap &params_dict,
                               const phi::Place &place)
    : info_(info), place_(place), inner_exe_(place_) {
  info_->RemoveDescFeedFetch();
  PADDLE_ENFORCE_GT(
      static_cast<int64_t>(info_->ProgramDesc().Block(0).OpSize()),
      0,
      platform::errors::PreconditionNotMet(
          "There is no operator in ProgramDesc."));
  utils::ShareParamsIntoScope(info_->ParamNames(), params_dict, &scope_);
  VLOG(6) << framework::GenScopeTreeDebugInfo(&scope_);
}

std::vector<Tensor> ExecutorEngine::operator()(
    const std::vector<Tensor> &inputs) {
  auto dense_tensors = utils::ToDenseTensors(inputs);
  return utils::ToTensors(this->operator()(dense_tensors));
}

std::vector<DenseTensor> ExecutorEngine::operator()(
    const std::vector<DenseTensor> &inputs) {
  utils::ShareIntoScope(info_->InputArgNames(), inputs, &scope_);
  const auto out_names = info_->OutputArgNames();
  inner_exe_.Run(info_->ProgramDesc(),
                 &scope_,
                 /*blockID=*/0,
                 false,
                 true,
                 out_names);
  std::vector<DenseTensor> outputs;
  utils::FetchOuts(out_names, scope_, &outputs);
  // Erase output vars to avoid data rewriting.
  scope_.EraseVars(out_names);
  return outputs;
}

const std::shared_ptr<FunctionInfo> &ExecutorEngine::Info() const {
  return info_;
}

}  // namespace jit
}  // namespace paddle
