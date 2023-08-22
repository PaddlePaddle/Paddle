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

namespace cinn::frontend {

std::vector<float> RunWithProgram(const Program& program,
                                  const Target& target,
                                  Variable out) {
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope = hlir::framework::BuildScope(target, graph);

  hlir::framework::ApplyPasses(graph.get(), {"InferShape"});
  hlir::framework::ApplyPasses(graph.get(), DefaultOpFusionPasses());
  VLOG(1) << "graph:\n" << graph->Visualize();
  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();
  runtime_program->Execute();

  return GetTensorData<float>(scope->GetTensor(out->id), target);
}

TEST(TransposeFolding, FoldTwoFillConstant) {
  NetBuilder builder("net_builder");
  auto x = builder.FillConstant<float>({32, 32}, 1.0f, "x");
  auto y = builder.FillConstant<float>({32, 32}, 1.0f, "y");
  auto transpose_x = builder.Transpose(x, {1, 0});
  auto transpose_y = builder.Transpose(y, {1, 0});
  auto out = builder.Add(transpose_x, transpose_y);
  auto program = builder.Build();
  auto target = common::DefaultTarget();

  size_t origin_size = program.size();
  VLOG(1) << "Program Before FillConstantFolding:\n" << program;
  // Program {
  //   x = fill_constant(value=1, dtype=float32, force_cpu=false, shape=[32,32])
  //   y = fill_constant(dtype=float32, shape=[32,32], value=1, force_cpu=false)
  //   var_1 = transpose(x, axis=[1,0])
  //   var_2 = transpose(y, axis=[1,0])
  //   var_3 = elementwise_add(var_1, var_2)
  // }

  auto origin_out = RunWithProgram(program, target, out);

  ProgramPass::Apply(&program, {}, target, {"FillConstantFolding"});
  size_t folded_size = program.size();
  VLOG(1) << "Program after FillConstantFolding:\n" << program;
  // y was removed
  // Program {
  //   x = fill_constant(value=1, dtype=float32, force_cpu=false, shape=[32,32])
  //   var_1 = transpose(x, axis=[1,0])
  //   var_2 = transpose(x, axis=[1,0])
  //   var_3 = elementwise_add(var_1, var_2)
  // }

  auto folded_out = RunWithProgram(program, target, out);

  ASSERT_EQ(origin_size, folded_size + 1);
  ASSERT_EQ(origin_out.size(), folded_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_FLOAT_EQ(origin_out[i], folded_out[i]);
  }
}

TEST(TransposeFolding, FoldTwoFillConstantWithSameOuput) {
  NetBuilder builder("net_builder");
  auto x = builder.FillConstant<float>({32, 32}, 1.0f, "x");
  auto y = builder.FillConstant<float>({32, 32}, 1.0f, "y");
  auto transpose_x = builder.Transpose(x, {1, 0});
  auto out = builder.Add(y, y);
  auto program = builder.Build();
  auto target = common::DefaultTarget();

  size_t origin_size = program.size();
  VLOG(1) << "Program Before FillConstantFolding:\n" << program;
  // Program {
  //   x = fill_constant(dtype=float32, shape=[32,32], value=1, force_cpu=false)
  //   y = fill_constant(shape=[32,32], dtype=float32, value=1, force_cpu=false)
  //   var_6 = transpose(x, axis=[1,0])
  //   var_7 = elementwise_add(y, y)
  // }

  auto origin_out = RunWithProgram(program, target, out);

  ProgramPass::Apply(&program, {}, target, {"FillConstantFolding"});
  size_t folded_size = program.size();
  VLOG(1) << "Program after FillConstantFolding:\n" << program;
  // Program {
  //   x = fill_constant(dtype=float32, shape=[32,32], value=1, force_cpu=false)
  //   var_6 = transpose(x, axis=[1,0])
  //   var_7 = elementwise_add(x, x)
  // }

  auto folded_out = RunWithProgram(program, target, out);

  ASSERT_EQ(origin_size, folded_size + 1);
  ASSERT_EQ(origin_out.size(), folded_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_FLOAT_EQ(origin_out[i], folded_out[i]);
  }
}

TEST(TransposeFolding, FoldThreeFillConstant) {
  NetBuilder builder("net_builder");
  auto x = builder.FillConstant<float>({32, 32}, 1.0f, "x");
  auto y = builder.FillConstant<float>({32, 32}, 1.0f, "y");
  auto z = builder.FillConstant<float>({32, 32}, 1.0f, "z");
  auto transpose_x = builder.Transpose(x, {1, 0});
  auto out = builder.Add(y, z);
  auto program = builder.Build();
  auto target = common::DefaultTarget();
  size_t origin_size = program.size();
  VLOG(1) << "Program Before FillConstantFolding:\n" << program;
  // Program {
  //   x = fill_constant(dtype=float32, shape=[32,32], value=1, force_cpu=false)
  //   y = fill_constant(dtype=float32, shape=[32,32], value=1, force_cpu=false)
  //   z = fill_constant(force_cpu=false, shape=[32,32], dtype=float32, value=1)
  //   var_10 = transpose(x, axis=[1,0])
  //   var_11 = elementwise_add(y, z)
  // }

  auto origin_out = RunWithProgram(program, target, out);

  ProgramPass::Apply(&program, {}, target, {"FillConstantFolding"});
  size_t folded_size = program.size();
  VLOG(1) << "Program after FillConstantFolding:\n" << program;
  // Program {
  //   x = fill_constant(dtype=float32, shape=[32,32], value=1, force_cpu=false)
  //   var_10 = transpose(x, axis=[1,0])
  //   var_11 = elementwise_add(x, x)
  // }

  auto folded_out = RunWithProgram(program, target, out);

  ASSERT_EQ(origin_size, folded_size + 2);
  ASSERT_EQ(origin_out.size(), folded_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_FLOAT_EQ(origin_out[i], folded_out[i]);
  }
}

TEST(TransposeFolding, FoldThreeFillConstantWithOneDiff) {
  NetBuilder builder("net_builder");
  auto x = builder.FillConstant<float>({32, 32}, 1.0f, "x");
  auto y = builder.FillConstant<float>({32, 32}, 1.0f, "y");
  auto z = builder.FillConstant<float>({32, 32}, 0.0f, "z");
  auto transpose_x = builder.Transpose(x, {1, 0});
  auto out = builder.Add(y, z);
  auto program = builder.Build();
  auto target = common::DefaultTarget();
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope = hlir::framework::BuildScope(target, graph);

  size_t origin_size = program.size();
  VLOG(1) << "Program Before FillConstantFolding:\n" << program;
  // Program {
  //   x = fill_constant(dtype=float32, shape=[32,32], value=1, force_cpu=false)
  //   y = fill_constant(force_cpu=false, shape=[32,32], dtype=float32, value=1)
  //   z = fill_constant(force_cpu=false, shape=[32,32], value=0, dtype=float32)
  //   var_15 = transpose(x, axis=[1,0])
  //   var_16 = elementwise_add(y, z)
  // }

  auto origin_out = RunWithProgram(program, target, out);

  ProgramPass::Apply(&program, {}, target, {"FillConstantFolding"});
  size_t folded_size = program.size();
  VLOG(1) << "Program after FillConstantFolding:\n" << program;
  // Program {
  //   x = fill_constant(dtype=float32, shape=[32,32], value=1, force_cpu=false)
  //   z = fill_constant(force_cpu=false, shape=[32,32], value=0, dtype=float32)
  //   var_15 = transpose(x, axis=[1,0])
  //   var_16 = elementwise_add(z, x)
  // }

  auto folded_out = RunWithProgram(program, target, out);

  ASSERT_EQ(origin_size, folded_size + 1);
  ASSERT_EQ(origin_out.size(), folded_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_FLOAT_EQ(origin_out[i], folded_out[i]);
  }
}

}  // namespace cinn::frontend
