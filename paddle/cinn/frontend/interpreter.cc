// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/frontend/interpreter.h"

#include "paddle/cinn/auto_schedule/auto_tuner.h"
#include "paddle/cinn/auto_schedule/tuning.h"
#include "paddle/cinn/frontend/optimize.h"
#include "paddle/cinn/frontend/syntax.h"
#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/graph_compiler_util.h"
#include "paddle/cinn/hlir/framework/pass.h"
#include "paddle/cinn/hlir/op/use_ops.h"
#include "paddle/cinn/hlir/pass/use_pass.h"
#include "paddle/cinn/runtime/flags.h"

PD_DECLARE_bool(enable_auto_tuner);

namespace cinn::frontend {

struct Interpreter::Impl {
  Impl(const std::vector<std::string>& input_names,
       const std::vector<hlir::framework::shape_t>& input_shapes)
      : scope_(std::make_shared<hlir::framework::Scope>()),
        input_names_(input_names),
        input_shapes_(input_shapes) {}

  /**
   * Build the model.
   * @param input_names The name of input variables.
   * @param input_shapes The input shapes.
   */
  void Build(const Target& target, const std::string& model_name = "");

 private:
  friend class Interpreter;

  std::vector<std::string> input_names_;
  absl::flat_hash_set<std::string> fetch_names_;
  std::vector<hlir::framework::shape_t> input_shapes_;

  std::shared_ptr<hlir::framework::Scope> scope_;
  std::unique_ptr<frontend::Program> program_;
  std::unique_ptr<hlir::framework::GraphCompiler> graph_compiler_;

  absl::flat_hash_map<std::string, Variable> var_map_;
  absl::flat_hash_map<std::string, std::string> var_map_paddle_to_cinn_;
  absl::flat_hash_map<std::string, std::string> var_map_cinn_to_paddle_;

  std::unique_ptr<hlir::framework::Program> runtime_program_;
  std::unique_ptr<hlir::framework::Program> prerun_program_;
};

void Interpreter::LoadPaddleModel(const std::string& model_dir,
                                  const Target& target,
                                  bool params_combined,
                                  const std::string& model_name) {
  std::unordered_map<std::string, std::vector<int>> input_shape_map;
  CHECK_EQ(impl_->input_names_.size(), impl_->input_shapes_.size());
  for (int idx = 0; idx < impl_->input_names_.size(); ++idx) {
    input_shape_map[impl_->input_names_[idx]] = impl_->input_shapes_[idx];
  }
  auto programTuple = LoadPaddleProgram(
      model_dir, impl_->scope_.get(), input_shape_map, params_combined, target);
  auto& program = std::get<0>(programTuple);
  auto& var_map = std::get<1>(programTuple);
  auto& var_map_paddle_to_program = std::get<2>(programTuple);
  auto& fetch_names = std::get<3>(programTuple);
  impl_->program_.reset(program.release());
  impl_->var_map_ = var_map;
  impl_->var_map_paddle_to_cinn_ = var_map_paddle_to_program;
  impl_->fetch_names_ = fetch_names;

  impl_->Build(target, model_name);
}

frontend::Program Interpreter::GetProgram() {
  frontend::Program* res = impl_->program_.get();
  return *res;
}

void Interpreter::Run() { impl_->runtime_program_->Execute(); }

hlir::framework::Tensor Interpreter::GetTensor(const std::string& name) {
  if (impl_->scope_->FindVar(name)) return impl_->scope_->GetTensor(name);

  auto it = impl_->var_map_paddle_to_cinn_.find(name);
  if (it == impl_->var_map_paddle_to_cinn_.end()) {
    LOG(FATAL) << "No variable called [" << name
               << "] found in executor\nThe existing vars: "
               << utils::Join(impl_->scope_->var_names(), ", ");
  }
  return impl_->scope_->GetTensor(it->second);
}

void Interpreter::Impl::Build(const Target& target,
                              const std::string& model_name) {
  CHECK(!var_map_.empty());
  VLOG(3) << "Program:\n" << *program_;
  // applay frontend pass
  std::unordered_set<std::string> fetch_var_ids;
  for (auto& name : fetch_names_) {
    CHECK(var_map_.count(name)) << "var_map finds no fetch var " << name;
    fetch_var_ids.insert(var_map_.at(name)->id);
  }

  auto graph = Optimize(program_.get(), fetch_var_ids, target);
  // auto graph                 =
  // std::make_shared<hlir::framework::Graph>(*program_, target);
  graph->attrs["model_name"] = std::make_shared<absl::any>(model_name);
  scope_ = hlir::framework::BuildScope(target, graph, scope_);

  hlir::framework::CompilationContext context(graph, scope_, target);
  context.with_instantiate_variables = true;
  if (FLAGS_enable_auto_tuner) {
    VLOG(4) << "Compile with auto-tune";
    auto_schedule::AutoTuner auto_tuner(target, graph.get());
    auto_tuner.Initialize(auto_schedule::AutoTuner::Config(),
                          graph_compiler_.get());
    auto_schedule::TuningOptions tuning_options;
    auto_schedule::TuningResult tuning_result = auto_tuner.Tune(tuning_options);
    context.ApplyTuningResult(tuning_result);
  }
  graph_compiler_ = std::make_unique<hlir::framework::GraphCompiler>(context);
  runtime_program_ = graph_compiler_->Build();
  runtime_program_->PreRun();
}

std::shared_ptr<hlir::framework::Scope> Interpreter::GetScope() {
  CHECK(impl_->scope_);
  return impl_->scope_;
}

Interpreter::Interpreter(
    const std::vector<std::string>& input_names,
    const std::vector<hlir::framework::shape_t>& input_shapes)
    : impl_(new Impl(input_names, input_shapes)) {}

}  // namespace cinn::frontend

cinn::frontend::Interpreter::~Interpreter() = default;
