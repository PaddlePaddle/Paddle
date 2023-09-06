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

#include "paddle/cinn/frontend/paddle_model_convertor.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "paddle/cinn/frontend/decomposer/test_helper.h"
#include "paddle/cinn/runtime/use_extern_funcs.h"

PD_DEFINE_string(model_dir, "", "");

namespace cinn {
namespace frontend {

template <typename T>
void RandomInput(const Target& target,
                 hlir::framework::Tensor tensor,
                 T low = static_cast<T>(0),
                 T high = static_cast<T>(1)) {
  std::vector<T> vec;
  InitRandomVector<T>(&vec, tensor->shape().numel(), low, high);
  CopyFromVector<T>(vec, tensor, target);
}

template <>
void RandomInput<bool>(const Target& target,
                       hlir::framework::Tensor tensor,
                       bool low,
                       bool high) {
  std::vector<int> vec_int;
  InitRandomVector<int>(&vec_int, tensor->shape().numel(), 0, 1);

  std::vector<bool> vec(vec_int.size());
  for (int i = 0; i < vec_int.size(); ++i) {
    vec[i] = static_cast<bool>(vec_int[i]);
  }
  CopyFromVector<bool>(vec, tensor, target);
}

void RunProgram(const Target& target, Program* prog) {
  const auto& inputs = prog->GetInputs();
  std::vector<std::string> input_names;
  for (const auto& var : inputs) {
    input_names.emplace_back(var->id);
  }

  LOG(INFO) << "The Program's inputs are ["
            << cinn::utils::Join(input_names, ", ") << "]";

  auto passes = DefaultTrainingOptimizeOptions();

  frontend::ProgramPass::Apply(prog, {}, target, passes.program_passes);

  auto graph = std::make_shared<hlir::framework::Graph>(*prog, target);
  hlir::framework::ApplyPasses(graph.get(), passes.graph_passes);

  auto scope = BuildScope(target, graph);

  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  for (size_t i = 0; i < input_names.size(); ++i) {
    scope->Var<hlir::framework::Tensor>(input_names[i]);
    auto tensor = scope->GetTensor(input_names[i]);

    if (inputs[i]->type.is_float(32)) {
      RandomInput<float>(target, tensor);
    } else if (inputs[i]->type.is_int(32)) {
      RandomInput<int>(target, tensor);
    } else if (inputs[i]->type.is_bool()) {
      RandomInput<bool>(target, tensor, 0, inputs[i]->shape[0]);
    } else {
      LOG(FATAL) << "Only support float/int/bool! Please check.";
    }
  }

  runtime_program->Execute();
}

TEST(PaddleModelConvertor, basic) {
  auto target = common::DefaultTarget();

  PaddleModelConvertor model_transform(target);
  model_transform.LoadModel(FLAGS_model_dir);
  auto program = model_transform();

  const auto& var_map = model_transform.var_map();
  const auto& var_model_to_program_map =
      model_transform.var_model_to_program_map();

  ASSERT_FALSE(var_map.empty());
  ASSERT_FALSE(var_model_to_program_map.empty());
  ASSERT_FALSE(model_transform.GetFetchList().empty());
  ASSERT_GT(program.size(), 0);

  RunProgram(target, &program);
}

}  // namespace frontend
}  // namespace cinn
