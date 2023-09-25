// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include <gtest/gtest.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>
#ifdef CINN_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include "paddle/cinn/common/target.h"
#include "paddle/cinn/frontend/optimize.h"
#include "paddle/cinn/frontend/pass/use_program_pass.h"
#include "paddle/cinn/frontend/program_pass.h"
#include "paddle/cinn/frontend/syntax.h"
#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/graph_compiler.h"
#include "paddle/cinn/hlir/framework/graph_compiler_util.h"
#include "paddle/cinn/hlir/framework/pass.h"
#include "paddle/cinn/hlir/framework/tensor.h"
#include "paddle/cinn/hlir/op/use_ops.h"
#include "paddle/cinn/hlir/pass/use_pass.h"
#include "paddle/cinn/utils/data_util.h"
#include "paddle/utils/flags.h"

PD_DECLARE_bool(cinn_use_op_fusion);

namespace cinn {
namespace frontend {

inline void PrintMatrix(const std::vector<float>& mat, int bs, int m, int n) {
  if (!VLOG_IS_ON(5)) {
    return;
  }
  const auto min_max = std::minmax_element(mat.begin(), mat.end());
  int min = static_cast<int>(*min_max.first);
  int max = static_cast<int>(*min_max.second);
  auto ele_width =
      std::max(std::to_string(min).length(), std::to_string(max).length());
  std::cout << "\n" << std::string((ele_width + 2) * n - 1, '-') << "\n";
  for (int b = 0; b < bs; b++) {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        std::cout << std::setw(ele_width) << mat[b * m * n + i * n + j] << ", ";
      }
      std::cout << "\n";
    }
    if (b != bs - 1) {
      std::cout << std::string((ele_width + 2) * n - 1, '*') << "\n";
    }
  }
  std::cout << std::string((ele_width + 2) * n - 1, '-') << "\n\n";
}

inline void RunGraph(std::shared_ptr<hlir::framework::Graph> graph,
                     const common::Target& target,
                     const std::shared_ptr<hlir::framework::Scope>& scope,
                     const std::vector<std::string>& output_ids,
                     const std::vector<std::string>& graph_passes) {
  hlir::framework::ApplyPasses(graph.get(), graph_passes);
  VLOG(3) << "Graph Viz:\n" << graph->Visualize();
  BuildScope(target, graph, scope);
  hlir::framework::CompilationContext context(graph, scope, target);
  context.attached_source_code = "";
  context.with_instantiate_variables = true;
  context.fetch_var_ids.insert(output_ids.begin(), output_ids.end());
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();
  runtime_program->Execute();
}

inline std::vector<float> RunProgram(
    const Program& program,
    const common::Target& target,
    const std::vector<std::string>& input_ids,
    const std::vector<std::string>& output_ids,
    const std::vector<std::string>& graph_passes,
    int seed = -1,
    bool print_tensor = false) {
  std::unordered_set<std::string> outputs_set{output_ids.begin(),
                                              output_ids.end()};
  auto graph =
      std::make_shared<hlir::framework::Graph>(program, outputs_set, target);
  auto scope = hlir::framework::BuildScope(target, graph);
  for (auto& input_id : input_ids) {
    scope->Var<hlir::framework::Tensor>(input_id);
    auto input_tensor = scope->GetTensor(input_id);
    SetRandData<int>(input_tensor, target, seed);
    if (print_tensor) {
      auto tensor_data = GetTensorData<float>(input_tensor, target);
      if (input_tensor->shape().data().size() == 2) {
        PrintMatrix(tensor_data,
                    1,
                    input_tensor->shape().data()[0],
                    input_tensor->shape().data()[1]);
      } else if (input_tensor->shape().data().size() == 3) {
        PrintMatrix(tensor_data,
                    input_tensor->shape().data()[0],
                    input_tensor->shape().data()[1],
                    input_tensor->shape().data()[2]);
      }
    }
  }

  RunGraph(graph, target, scope, output_ids, graph_passes);

  auto output_tensor = scope->GetTensor(output_ids.front());
  auto output_data = GetTensorData<float>(output_tensor, target);
  if (print_tensor) {
    if (output_tensor->shape().data().size() == 2) {
      PrintMatrix(output_data,
                  1,
                  output_tensor->shape().data()[0],
                  output_tensor->shape().data()[1]);
    } else if (output_tensor->shape().data().size() == 3) {
      PrintMatrix(output_data,
                  output_tensor->shape().data()[0],
                  output_tensor->shape().data()[1],
                  output_tensor->shape().data()[2]);
    }
  }
  return output_data;
}

struct OptimizeConfig {
  struct PassGroup;
  explicit OptimizeConfig(const PassGroup& program_passes)
      : program_passes{program_passes} {
    if (FLAGS_cinn_use_op_fusion) {
      graph_passes = {{"OpFusionPass", "FusionMergePass"},
                      {"OpFusionPass", "FusionMergePass"}};
    }
  }
  OptimizeConfig(const PassGroup& program_passes, const PassGroup& graph_passes)
      : program_passes{program_passes}, graph_passes{graph_passes} {}

  OptimizeConfig(const std::pair<std::vector<std::string>,
                                 std::vector<std::string>>& program_passes) {
    this->program_passes.ctrl = program_passes.first;
    this->program_passes.exp = program_passes.second;

    if (FLAGS_cinn_use_op_fusion) {
      graph_passes = {
          {"TransToCustomCallPass", "OpFusionPass", "FusionMergePass"},
          {"TransToCustomCallPass", "OpFusionPass", "FusionMergePass"}};
    }
  }

  struct PassGroup {
    // control group
    std::vector<std::string> ctrl;
    // experimental group
    std::vector<std::string> exp;
  };
  PassGroup program_passes;
  PassGroup graph_passes;
};

inline void CompareResult(Program* program,
                          const common::Target& target,
                          const std::vector<std::string>& input_ids,
                          const std::vector<std::string>& output_ids,
                          size_t size_diff,
                          const OptimizeConfig& passes,
                          int seed = -1,
                          bool print_tensor = false) {
  std::unordered_set<std::string> fetch_ids(output_ids.begin(),
                                            output_ids.end());
  // apply common passes
  ProgramPass::Apply(program, fetch_ids, target, passes.program_passes.ctrl);
  // get original program size
  auto origin_size = program->size();
  // get original output
  auto origin_out = RunProgram(*program,
                               target,
                               input_ids,
                               output_ids,
                               passes.graph_passes.ctrl,
                               seed,
                               print_tensor);

  // apply fused passes
  ProgramPass::Apply(program, fetch_ids, target, passes.program_passes.exp);

  // get fused program size
  auto fused_size = program->size();
  ASSERT_EQ(size_diff, origin_size - fused_size);
  // get fused output
  auto fused_out = RunProgram(*program,
                              target,
                              input_ids,
                              output_ids,
                              passes.graph_passes.exp,
                              seed,
                              print_tensor);

  ASSERT_EQ(origin_out.size(), fused_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_FLOAT_EQ(origin_out[i], fused_out[i]) << " i is " << i;
  }
}

inline bool CompareProgramPassResult(
    Program* program,
    const common::Target& target,
    const std::unordered_set<std::string>& fetch_ids,
    const size_t size_diff,
    const OptimizeConfig& passes) {
  // apply common passes
  ProgramPass::Apply(program, fetch_ids, target, passes.program_passes.ctrl);
  // get original program size
  auto origin_size = program->size();

  // apply fused passes
  ProgramPass::Apply(program, fetch_ids, target, passes.program_passes.exp);

  // get fused program size
  auto fused_size = program->size();
  return size_diff == (origin_size - fused_size);
}

}  // namespace frontend
}  // namespace cinn
