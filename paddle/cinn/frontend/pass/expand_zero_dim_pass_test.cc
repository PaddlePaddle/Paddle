// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include <gtest/gtest.h>

#include <cfloat>

#include "paddle/cinn/cinn.h"
#include "paddle/cinn/frontend/decomposer/test_helper.h"
#include "paddle/cinn/frontend/net_builder.h"
#include "paddle/cinn/frontend/optimize.h"
#include "paddle/cinn/frontend/pass/use_program_pass.h"
#include "paddle/cinn/frontend/program_pass.h"
#include "paddle/cinn/frontend/syntax.h"
#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/graph_compiler.h"
#include "paddle/cinn/hlir/framework/pass.h"
#include "paddle/cinn/hlir/op/use_ops.h"
#include "paddle/cinn/hlir/pass/use_pass.h"
#include "paddle/cinn/utils/data_util.h"

namespace cinn {
namespace frontend {

int GetSize(const std::vector<int>& shape) {
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

std::unordered_map<std::string, std::vector<float>> GetInputRandom(
    const std::vector<Variable>&& inputs) {
  std::unordered_map<std::string, std::vector<float>> input_data;
  for (auto input : inputs) {
    input_data[input->id] = std::vector<float>(GetSize(input->shape));
    InitRandomVector<float>(
        &input_data[input->id], input_data[input->id].size(), 0.0f, 1.0f, 1e-3);
  }

  return input_data;
}

std::unordered_map<std::string, hlir::framework::Tensor> RunWithProgram(
    const Program& program,
    const Target& target,
    const std::unordered_map<std::string, std::vector<float>>& input_data,
    const std::unordered_set<std::string>& fetch_ids) {
  auto graph =
      std::make_shared<hlir::framework::Graph>(program, fetch_ids, target);
  auto scope = hlir::framework::BuildScope(target, graph);

  hlir::framework::ApplyPasses(graph.get(), {"InferShape"});
  hlir::framework::ApplyPasses(graph.get(), DefaultOpFusionPasses());
  VLOG(1) << "graph:\n" << graph->Visualize();
  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();
  for (auto& data : input_data) {
    scope->Var<hlir::framework::Tensor>(data.first);
    auto tensor = scope->GetTensor(data.first);
    CopyFromVector(data.second, tensor, target);
  }
  runtime_program->Execute();

  std::unordered_map<std::string, hlir::framework::Tensor> outputs;
  for (auto id : fetch_ids) {
    auto tensor = scope->GetTensor(id);
    outputs[id] = tensor;
  }
  return outputs;
}

TEST(ExpandZeroDimPass, expand_zero_dim_1) {
  NetBuilder builder("expand_zero_dim_1");
  auto x = builder.CreateInput(Float(32), {}, "x");
  auto y = builder.CreateInput(Float(32), {}, "y");
  auto out = builder.Add(x, y);
  auto program = builder.Build();
  auto target = common::DefaultTarget();

  size_t origin_size = program.size();
  VLOG(1) << "Program Before ExpandZeroDimPass:\n" << program;
  /*
    Program {
      Var(var_1: shape=[], dtype=float32)
      Var(y: shape=[], dtype=float32)
      Var(x: shape=[], dtype=float32)

      var_1 = elementwise_add(x, y, axis=-1)
    }
  */
  ProgramPass::Apply(&program, {}, target, {"ExpandZeroDim"});
  size_t pass_size = program.size();
  VLOG(1) << "Program after ExpandZeroDimPass:\n" << program;
  /*
    Program {
      Var(var_1: shape=[1], dtype=float32)
      Var(y: shape=[1], dtype=float32)
      Var(x: shape=[1], dtype=float32)

      var_1 = elementwise_add(x, y, axis=-1)
    }
  */
  auto input_data = GetInputRandom({x, y});
  auto fetch_ids = {out->id};
  auto outputs = RunWithProgram(program, target, input_data, fetch_ids);
  for (auto iter : outputs) {
    // output var_1: shape=[1]
    ASSERT_EQ(iter.second->shape().data().size(), 1);
  }
}

TEST(ExpandZeroDimPass, expand_zero_dim_2) {
  NetBuilder builder("expand_zero_dim_1");
  auto x = builder.CreateInput(Float(32), {3, 5}, "x");
  auto y = builder.CreateInput(Float(32), {}, "y");
  auto out = builder.Add(x, y);
  auto program = builder.Build();
  auto target = common::DefaultTarget();

  size_t origin_size = program.size();
  VLOG(1) << "Program Before ExpandZeroDimPass:\n" << program;
  /*
    Program {
      Var(var_1: shape=[3, 5], dtype=float32)
      Var(y: shape=[], dtype=float32)
      Var(x: shape=[3, 5], dtype=float32)

      var_1 = elementwise_add(x, y, axis=-1)
    }
  */
  ProgramPass::Apply(&program, {}, target, {"ExpandZeroDim"});
  size_t pass_size = program.size();
  VLOG(1) << "Program after ExpandZeroDimPass:\n" << program;
  /*
    Program {
      Var(var_1: shape=[3, 5], dtype=float32)
      Var(y: shape=[1], dtype=float32)
      Var(x: shape=[3, 5], dtype=float32)

      var_1 = elementwise_add(x, y, axis=-1)
    }
  */
  auto input_data = GetInputRandom({x, y});
  auto fetch_ids = {out->id};
  auto outputs = RunWithProgram(program, target, input_data, fetch_ids);
  for (auto iter : outputs) {
    // output var_1: shape=[3, 5]
    ASSERT_EQ(iter.second->shape().data().size(), 2);
  }
}

}  // namespace frontend
}  // namespace cinn
