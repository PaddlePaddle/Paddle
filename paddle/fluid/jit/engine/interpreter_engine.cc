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

#include "paddle/fluid/jit/engine/interpreter_engine.h"

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/new_executor/interpretercore.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace jit {

InterpreterEngine::InterpreterEngine(const std::shared_ptr<FunctionInfo> &info,
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
  CreateInterpreterCore();
}

void InterpreterEngine::CreateInterpreterCore() {
  auto &program_desc = info_->ProgramDesc();

  // apply inference pass
  framework::ir::Graph graph{program_desc};
  auto pass =
      framework::ir::PassRegistry::Instance().Get("delete_dropout_op_x_pass");
  pass->Apply(&graph);
#ifdef PADDLE_WITH_MKLDNN
  auto mkldnn_pass =
      framework::ir::PassRegistry::Instance().Get("mkldnn_placement_pass");
  mkldnn_pass->Set("mkldnn_enabled_op_types",
                   new std::unordered_set<std::string>({}));
  mkldnn_pass->Apply(&graph);
#endif

  GraphToProgram(graph, &converted_prog_, nullptr);

  auto in_names = info_->InputArgNames();
  auto out_names = info_->OutputArgNames();
  std::set<std::string> skip_gc_vars;
  skip_gc_vars.insert(in_names.begin(), in_names.end());
  skip_gc_vars.insert(out_names.begin(), out_names.end());

  inner_interpreter_ =
      std::make_shared<InterpreterCore>(place_,
                                        converted_prog_.Block(0),
                                        /*skip_gc_vars=*/skip_gc_vars,
                                        &scope_,
                                        /*used_for_jit=*/true);
}

std::vector<Tensor> InterpreterEngine::operator()(
    const std::vector<Tensor> &inputs) {
  auto dense_tensors = utils::ToDenseTensors(inputs);
  return utils::ToTensors(this->operator()(dense_tensors));
}

std::vector<DenseTensor> InterpreterEngine::operator()(
    const std::vector<DenseTensor> &inputs) {
  utils::ShareIntoScope(info_->InputArgNames(), inputs, &scope_);

  // the latter can be moved to python side.
  auto &feed_names = info_->InputArgNames();
  auto &fetch_names = info_->OutputArgNames();
  paddle::framework::FetchList outs = inner_interpreter_->Run(feed_names);

  std::vector<DenseTensor> outputs;
  utils::FetchOuts(info_->OutputArgNames(), scope_, &outputs);
  scope_.DropKids();

  return outputs;
}

const std::shared_ptr<FunctionInfo> &InterpreterEngine::Info() const {
  return info_;
}

}  // namespace jit
}  // namespace paddle
