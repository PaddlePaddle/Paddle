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

#include <gtest/gtest.h>

#include <cfloat>

#include "paddle/cinn/cinn.h"
#include "paddle/cinn/frontend/net_builder.h"
#include "paddle/cinn/frontend/pass/pass_test_helper.h"
#include "paddle/cinn/frontend/pass/use_program_pass.h"
#include "paddle/cinn/frontend/program_pass.h"
#include "paddle/cinn/frontend/syntax.h"
#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/graph_compiler.h"
#include "paddle/cinn/hlir/framework/graph_compiler_util.h"
#include "paddle/cinn/hlir/framework/pass.h"
#include "paddle/cinn/hlir/op/use_ops.h"
#include "paddle/cinn/hlir/pass/use_pass.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/cinn/utils/data_util.h"

namespace cinn::frontend {

void RunWithProgram(const Program& program,
                    const Target& target,
                    const std::shared_ptr<hlir::framework::Scope>& scope) {
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPasses(graph.get(), {"OpFusionPass"});
  VLOG(1) << "graph:\n" << graph->Visualize();
  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();
  runtime_program->Execute();
}

TEST(TransposeFoldingInput, FoldIntoDotBatchedCase1) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {4, 5, 3}, "X");
  auto y = builder.CreateInput(Float(32), {4, 5, 6}, "Y");
  auto transpose_x = builder.Transpose(x, {0, 2, 1});
  auto out = builder.Matmul(transpose_x, y);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{x.id(), y.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  std::pair<std::vector<std::string>, std::vector<std::string>> passes{
      {"Decomposer", "RemoveIdentity"},
      {"TransposeFoldingInput",
       "GemmRewriter",
       "TransposeFoldingOutput",
       "GemmRewriter"}};
  CompareResult(&program, target, input_ids, {out->id}, 1, passes, 123, true);
}

TEST(TransposeFoldingInput, FoldIntoDotBachedCase2) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {4, 3, 5}, "X");
  auto y = builder.CreateInput(Float(32), {4, 6, 5}, "Y");
  auto transpose_y = builder.Transpose(y, {0, 2, 1});
  auto out = builder.Matmul(x, transpose_y);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{x.id(), y.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  std::pair<std::vector<std::string>, std::vector<std::string>> passes{
      {"Decomposer", "RemoveIdentity"},
      {"TransposeFoldingInput",
       "GemmRewriter",
       "TransposeFoldingOutput",
       "GemmRewriter"}};
  CompareResult(&program, target, input_ids, {out->id}, 1, passes, 123, true);
}

TEST(TransposeFoldingInput, FoldIntoDotBachedCase3) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {4, 5, 3}, "X");
  auto y = builder.CreateInput(Float(32), {4, 6, 5}, "Y");
  auto transpose_x = builder.Transpose(x, {0, 2, 1});
  auto transpose_y = builder.Transpose(y, {0, 2, 1});
  auto out = builder.Matmul(transpose_x, transpose_y);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{x.id(), y.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  std::pair<std::vector<std::string>, std::vector<std::string>> passes{
      {"Decomposer", "RemoveIdentity"},
      {"TransposeFoldingInput",
       "GemmRewriter",
       "TransposeFoldingOutput",
       "GemmRewriter"}};
  CompareResult(&program, target, input_ids, {out->id}, 2, passes, 123, true);
}

TEST(TransposeFoldingInput, FoldIntoDotCase1) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {2, 3}, "X");
  auto y = builder.CreateInput(Float(32), {2, 3}, "Y");
  auto transpose_y = builder.Transpose(y, {1, 0});
  auto out = builder.Matmul(x, transpose_y);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{x.id(), y.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  std::pair<std::vector<std::string>, std::vector<std::string>> passes{
      {"Decomposer", "RemoveIdentity"},
      {"TransposeFoldingInput",
       "GemmRewriter",
       "TransposeFoldingOutput",
       "GemmRewriter"}};
  CompareResult(&program, target, input_ids, {out->id}, 1, passes, 123, true);
}

TEST(TransposeFoldingInput, FoldIntoDotCase2) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a = builder.FillConstant<float>({2, 20}, 2.0f, "A");
  auto b = builder.Transpose(a, {1, 0});
  auto c = builder.CreateInput(Float(32), {121, 20}, "C");
  auto d = builder.Matmul(c, b);
  auto x = builder.FillConstant<float>({2, 20}, 1.0f, "X");
  auto y = builder.Transpose(x, {1, 0});
  auto z = builder.CreateInput(Float(32), {121, 20}, "Z");
  auto q = builder.Matmul(z, y);
  auto out = builder.Add(d, q);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{c.id(), z.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  std::pair<std::vector<std::string>, std::vector<std::string>> passes{
      {"Decomposer", "RemoveIdentity"},
      {"TransposeFoldingInput",
       "GemmRewriter",
       "TransposeFoldingOutput",
       "GemmRewriter"}};
  CompareResult(&program, target, input_ids, {out->id}, 2, passes, 123, true);
}

TEST(TransposeFoldingInput, TransposeOutInFetchIds) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {2, 3}, "X");
  auto y = builder.CreateInput(Float(32), {2, 3}, "Y");
  auto transpose_y = builder.Transpose(y, {1, 0});
  auto out = builder.Matmul(x, transpose_y);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{x.id(), y.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  std::pair<std::vector<std::string>, std::vector<std::string>> passes{
      {"Decomposer", "RemoveIdentity"},
      {"TransposeFoldingInput",
       "GemmRewriter",
       "TransposeFoldingOutput",
       "GemmRewriter"}};
  CompareResult(&program,
                target,
                input_ids,
                {out->id, transpose_y->id},
                0,
                passes,
                123,
                true);
}

TEST(TransposeFoldingInput, TransposeOutUsedByOtherInstrs) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {2, 2}, "X");
  auto y = builder.CreateInput(Float(32), {2, 2}, "Y");
  auto transpose_y = builder.Transpose(y, {1, 0});
  auto dot = builder.Matmul(x, transpose_y);
  auto out = builder.Add(transpose_y, dot);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{x.id(), y.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  std::pair<std::vector<std::string>, std::vector<std::string>> passes{
      {"Decomposer", "RemoveIdentity"},
      {"TransposeFoldingInput",
       "GemmRewriter",
       "TransposeFoldingOutput",
       "GemmRewriter"}};
  CompareResult(&program, target, input_ids, {out->id}, 0, passes, 123, true);
}

TEST(TransposeFoldingInput, TransposeTwiceWithMatmul) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {2, 20}, "X");
  auto y = builder.CreateInput(Float(32), {10201, 20}, "Y");
  auto z = builder.CreateInput(Float(32), {10201, 2}, "Z");

  auto x_t = builder.Transpose(x, {1, 0});
  auto x_t_t = builder.Transpose(x_t, {1, 0});
  auto dot1 = builder.Matmul(y, x_t);
  auto dot2 = builder.Matmul(z, x_t_t);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{x.id(), y.id(), z.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  std::pair<std::vector<std::string>, std::vector<std::string>> passes{
      {"Decomposer", "RemoveIdentity"},
      {"TransposeFoldingInput",
       "GemmRewriter",
       "TransposeFoldingOutput",
       "GemmRewriter"}};
  CompareResult(
      &program, target, input_ids, {dot1->id, dot2->id}, 1, passes, 123, true);
}

TEST(TransposeFoldingInput, TransposeWithMultiMamtul) {
  if (!cinn::runtime::IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {2, 2}, "X");
  auto y = builder.CreateInput(Float(32), {2, 2}, "Y");
  auto transpose_y = builder.Transpose(y, {1, 0});
  auto dot1 = builder.Matmul(x, transpose_y);
  auto dot2 = builder.Matmul(transpose_y, x);
  auto out = builder.Add(dot1, dot2);
  auto program = builder.Build();

  common::Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{x.id(), y.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  std::pair<std::vector<std::string>, std::vector<std::string>> passes{
      {"Decomposer", "RemoveIdentity"},
      {"TransposeFoldingInput",
       "GemmRewriter",
       "TransposeFoldingOutput",
       "GemmRewriter"}};
  CompareResult(&program, target, input_ids, {out->id}, 1, passes, 123, true);
}

}  // namespace cinn::frontend
