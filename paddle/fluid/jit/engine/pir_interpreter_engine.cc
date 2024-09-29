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

#include "paddle/fluid/jit/engine/pir_interpreter_engine.h"
#include "paddle/fluid/jit/engine/interpreter_engine.h"

#include "paddle/fluid/framework/new_executor/interpretercore.h"
#include "paddle/fluid/pir/transforms/pd_op_to_kernel_pass.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/value.h"

namespace paddle {
namespace jit {

PirInterpreterEngine::PirInterpreterEngine(
    const std::shared_ptr<PirFunctionInfo> &info,
    const std::shared_ptr<VariableMap> &params_dict,
    const phi::Place &place,
    const std::shared_ptr<pir::Program> &prog)
    : info_(info), params_dict_(params_dict), place_(place), prog_(prog) {
  PADDLE_ENFORCE_GT(static_cast<int64_t>(info_->Program()->block()->size()),
                    0,
                    common::errors::PreconditionNotMet(
                        "There is no operator in ProgramDesc."));
  utils::ShareParamsIntoScope(info_->ParamNames(), params_dict_, &scope_);
  CreateInterpreterCore();
}

void PirInterpreterEngine::CreateInterpreterCore() {
  framework::interpreter::ExecutionConfig execution_config;
  execution_config.create_local_scope = false;
  execution_config.used_for_jit = true;

  auto in_names = info_->InputArgNames();
  auto out_names = info_->OutputArgNames();

  execution_config.skip_gc_vars.insert(in_names.begin(), in_names.end());
  execution_config.skip_gc_vars.insert(out_names.begin(), out_names.end());
  inner_interpreter_ = std::make_shared<PirInterpreter>(
      place_, out_names, prog_->block(), &scope_, execution_config);
}

std::vector<Tensor> PirInterpreterEngine::operator()(
    const std::vector<Tensor> &inputs) {
  auto dense_tensors = utils::ToDenseTensors(inputs);
  return utils::ToTensors(this->operator()(dense_tensors));
}

std::vector<DenseTensor> PirInterpreterEngine::operator()(
    const std::vector<DenseTensor> &inputs) {
  prog_ = paddle::dialect::PdOpLowerToKernelPass(prog_.get(), place_);
  utils::ShareIntoScope(info_->InputArgNames(), inputs, &scope_);

  // the latter can be moved to python side.
  auto &feed_names = info_->InputArgNames();
  paddle::framework::FetchList outs = inner_interpreter_->Run(feed_names);

  std::vector<DenseTensor> outputs;
  utils::FetchOuts(info_->OutputArgNames(), scope_, &outputs);
  scope_.DropKids();

  return outputs;
}

const std::shared_ptr<PirFunctionInfo> &PirInterpreterEngine::Info() const {
  return info_;
}

std::unique_ptr<BaseEngine> PirInterpreterEngine::Clone(void *stream) {
  auto *x = new PirInterpreterEngine(info_, params_dict_, place_, prog_);
  return std::unique_ptr<BaseEngine>(x);
}

}  // namespace jit
}  // namespace paddle
