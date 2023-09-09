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
#include "paddle/cinn/frontend/pass/use_program_pass.h"
#include "paddle/cinn/frontend/program_pass.h"
#include "paddle/cinn/frontend/syntax.h"
#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/graph_compiler.h"
#include "paddle/cinn/hlir/framework/graph_compiler_util.h"
#include "paddle/cinn/hlir/framework/pass.h"
#include "paddle/cinn/hlir/op/use_ops.h"
#include "paddle/cinn/hlir/pass/use_pass.h"
#include "paddle/cinn/utils/data_util.h"

namespace cinn::frontend {

void SetInputData(const hlir::framework::Tensor& tensor, Target target) {
  auto* data = tensor->mutable_data<float>(target);
  std::vector<float> host_memory(tensor->shape().numel(), 0);
  for (size_t i = 0; i < tensor->shape().numel(); ++i) {
    host_memory[i] = static_cast<float>(i);
  }
#ifdef CINN_WITH_CUDA
  if (target == common::DefaultNVGPUTarget()) {
    cudaMemcpy(data,
               host_memory.data(),
               tensor->shape().numel() * sizeof(float),
               cudaMemcpyHostToDevice);
    return;
  }
#endif
  CHECK(target == common::DefaultHostTarget());
  std::copy(host_memory.begin(), host_memory.end(), data);
}
std::vector<std::vector<float>> RunWithProgram(
    const Program& program,
    const Target& target,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& out_ids) {
  std::unordered_set<std::string> fetch_list;
  for (auto id : out_ids) {
    fetch_list.insert(id);
  }
  auto graph =
      std::make_shared<hlir::framework::Graph>(program, fetch_list, target);
  auto scope = hlir::framework::BuildScope(target, graph);

  for (const auto& in_name : input_names) {
    scope->Var<hlir::framework::Tensor>(in_name);
    SetInputData(scope->GetTensor(in_name), target);
  }

  hlir::framework::ApplyPasses(graph.get(), {"InferShape", "OpFusionPass"});
  VLOG(1) << "graph:\n" << graph->Visualize();
  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();
  runtime_program->Execute();

  std::vector<std::vector<float>> outputs;
  for (const auto& out_id : out_ids) {
    outputs.emplace_back(
        GetTensorData<float>(scope->GetTensor(out_id), target));
  }
  return outputs;
}

TEST(TransposeCollapsing, FuseTwoTranspose) {
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {4, 5, 3}, "X");
  auto x_t = builder.Transpose(x, {0, 2, 1});
  auto out = builder.Transpose(x_t, {2, 1, 0});
  auto program = builder.Build();
  auto target = common::DefaultTarget();

  std::initializer_list<std::string> fetch_list = {out->id};

  size_t origin_size = program.size();
  VLOG(1) << "Program before pass:\n" << program;
  // Program {
  //   var_0 = transpose(X, axis=[0,2,1])
  //   var_1 = transpose(var_0, axis=[2,1,0])
  // }

  auto origin_out = RunWithProgram(program, target, {"X"}, fetch_list);

  ProgramPass::Apply(&program, fetch_list, target, {"TransposeCollapsing"});
  size_t folded_size = program.size();
  VLOG(1) << "Program after pass:\n" << program;
  // Program {
  //   var_1 = transpose(X, axis=[1,2,0])
  // }

  auto folded_out = RunWithProgram(program, target, {"X"}, fetch_list);

  ASSERT_EQ(origin_size, folded_size + 1);
  ASSERT_EQ(origin_out.size(), folded_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_EQ(origin_out[i].size(), folded_out[i].size());
    for (size_t j = 0; j < origin_out[i].size(); ++j) {
      ASSERT_FLOAT_EQ(origin_out[i][j], folded_out[i][j]);
    }
  }
}

TEST(TransposeCollapsing, FuseThreeTranspose) {
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {4, 5, 3}, "X");
  auto x_1t = builder.Transpose(x, {0, 2, 1});
  auto x_2t = builder.Transpose(x_1t, {2, 1, 0});
  auto out = builder.Transpose(x_2t, {1, 2, 0});
  auto program = builder.Build();
  auto target = common::DefaultTarget();

  std::initializer_list<std::string> fetch_list = {out->id};

  size_t origin_size = program.size();
  VLOG(1) << "Program before pass:\n" << program;
  // Program {
  //   var_4 = transpose(X, axis=[0,2,1])
  //   var_5 = transpose(var_4, axis=[2,1,0])
  //   var_6 = transpose(var_5, axis=[1,2,0])
  // }

  auto origin_out = RunWithProgram(program, target, {"X"}, fetch_list);

  ProgramPass::Apply(&program, fetch_list, target, {"TransposeCollapsing"});
  size_t folded_size = program.size();
  VLOG(1) << "Program after pass:\n" << program;
  // Program {
  //   var_6 = transpose(X, axis=[2,0,1])
  // }

  auto folded_out = RunWithProgram(program, target, {"X"}, fetch_list);

  ASSERT_EQ(origin_size, folded_size + 2);
  ASSERT_EQ(origin_out.size(), folded_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_EQ(origin_out[i].size(), folded_out[i].size());
    for (size_t j = 0; j < origin_out[i].size(); ++j) {
      ASSERT_FLOAT_EQ(origin_out[i][j], folded_out[i][j]);
    }
  }
}

TEST(TransposeCollapsing, RemoveUselessTranspose) {
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {4, 5, 3}, "X");
  auto x_t = builder.Transpose(x, {0, 1, 2});
  auto out = builder.Add(x, x_t);
  auto program = builder.Build();
  auto target = common::DefaultTarget();

  std::initializer_list<std::string> fetch_list = {out->id};

  size_t origin_size = program.size();
  VLOG(1) << "Program before pass:\n" << program;
  // Program {
  //   var_9 = transpose(X, axis=[0,1,2])
  //   var_10 = elementwise_add(X, var_9)
  // }

  auto origin_out = RunWithProgram(program, target, {"X"}, fetch_list);

  ProgramPass::Apply(&program, fetch_list, target, {"TransposeCollapsing"});
  size_t folded_size = program.size();
  VLOG(1) << "Program after pass:\n" << program;
  // Program {
  //   var_10 = elementwise_add(X, X)
  // }
  auto folded_out = RunWithProgram(program, target, {"X"}, fetch_list);

  ASSERT_EQ(origin_out.size(), folded_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_EQ(origin_out[i].size(), folded_out[i].size());
    for (size_t j = 0; j < origin_out[i].size(); ++j) {
      ASSERT_FLOAT_EQ(origin_out[i][j], folded_out[i][j]);
    }
  }
}

TEST(TransposeCollapsing, ReplaceUselessTransposeWithIndentity) {
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {4, 5, 3}, "X");
  auto out = builder.Transpose(x, {0, 1, 2});
  auto program = builder.Build();
  auto target = common::DefaultTarget();

  std::initializer_list<std::string> fetch_list = {out->id};

  size_t origin_size = program.size();
  VLOG(1) << "Program before pass:\n" << program;
  // Program {
  //   var_9 = transpose(X, axis=[0,1,2])
  //   var_10 = elementwise_add(X, var_9)
  // }

  auto origin_out = RunWithProgram(program, target, {"X"}, fetch_list);

  ProgramPass::Apply(&program, fetch_list, target, {"TransposeCollapsing"});
  size_t folded_size = program.size();
  VLOG(1) << "Program after pass:\n" << program;
  // Program {
  //   var_10 = elementwise_add(X, X)
  // }

  auto folded_out = RunWithProgram(program, target, {"X"}, fetch_list);

  ASSERT_EQ(origin_size, folded_size);
  ASSERT_EQ(origin_out.size(), folded_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_EQ(origin_out[i].size(), folded_out[i].size());
    for (size_t j = 0; j < origin_out[i].size(); ++j) {
      ASSERT_FLOAT_EQ(origin_out[i][j], folded_out[i][j]);
    }
  }
}

TEST(TransposeCollapsing, FuseTransposeToUseless) {
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {4, 5, 3}, "X");
  auto x_1t = builder.Transpose(x, {0, 2, 1});
  auto x_2t = builder.Transpose(x_1t, {0, 2, 1});
  auto x_3t = builder.Transpose(x_2t, {0, 2, 1});
  auto out = builder.Add(x_3t, x_3t);
  auto program = builder.Build();
  auto target = common::DefaultTarget();

  std::initializer_list<std::string> fetch_list = {out->id};

  size_t origin_size = program.size();
  VLOG(1) << "Program before pass:\n" << program;
  // Program {
  //   var_13 = transpose(X, axis=[0,2,1])
  //   var_14 = transpose(var_13, axis=[0,2,1])
  //   var_15 = transpose(var_14, axis=[0,2,1])
  //   var_16 = elementwise_add(var_15, var_15)
  // }

  auto origin_out = RunWithProgram(program, target, {"X"}, fetch_list);

  ProgramPass::Apply(&program, fetch_list, target, {"TransposeCollapsing"});
  size_t folded_size = program.size();
  VLOG(1) << "Program after pass:\n" << program;
  // Program {
  //   var_15 = transpose(X, axis=[0,2,1])
  //   var_16 = elementwise_add(var_15, var_15)
  // }

  auto folded_out = RunWithProgram(program, target, {"X"}, fetch_list);

  ASSERT_EQ(origin_size, folded_size + 2);
  ASSERT_EQ(origin_out.size(), folded_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_EQ(origin_out[i].size(), folded_out[i].size());
    for (size_t j = 0; j < origin_out[i].size(); ++j) {
      ASSERT_FLOAT_EQ(origin_out[i][j], folded_out[i][j]);
    }
  }
}

TEST(TransposeCollapsing, FuseTransposeWithMultiOutput) {
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {4, 5, 3}, "X");
  auto x_1t = builder.Transpose(x, {0, 2, 1});
  auto x_2t = builder.Transpose(x_1t, {2, 0, 1});
  auto x_3t = builder.Transpose(x_2t, {2, 1, 0});
  auto out1 = builder.Sqrt(x_1t);
  auto out2 = builder.Sqrt(x_2t);
  auto out3 = builder.Sqrt(x_3t);
  auto program = builder.Build();
  auto target = common::DefaultTarget();

  std::initializer_list<std::string> fetch_list = {
      out1->id, out2->id, out3->id};

  size_t origin_size = program.size();
  VLOG(1) << "Program before pass:\n" << program;
  // Program {
  //   var_18 = transpose(X, axis=[0,2,1])
  //   var_19 = transpose(var_18, axis=[2,0,1])
  //   var_20 = transpose(var_19, axis=[1,0,2])
  //   var_21 = sqrt(var_18)
  //   var_22 = sqrt(var_19)
  //   var_23 = sqrt(var_20)
  // }

  auto origin_out = RunWithProgram(program, target, {"X"}, fetch_list);

  ProgramPass::Apply(&program, fetch_list, target, {"TransposeCollapsing"});
  size_t folded_size = program.size();
  VLOG(1) << "Program after pass:\n" << program;
  // Program {
  //   var_18 = transpose(X, axis=[0,2,1])
  //   var_19 = transpose(X, axis=[1,0,2])
  //   var_20 = transpose(X, axis=[0,2,1])
  //   var_21 = sqrt(var_18)
  //   var_22 = sqrt(var_19)
  //   var_23 = sqrt(var_20)
  // }

  auto folded_out = RunWithProgram(program, target, {"X"}, fetch_list);

  ASSERT_EQ(origin_size, folded_size);
  ASSERT_EQ(origin_out.size(), folded_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_EQ(origin_out[i].size(), folded_out[i].size());
    for (size_t j = 0; j < origin_out[i].size(); ++j) {
      ASSERT_FLOAT_EQ(origin_out[i][j], folded_out[i][j]);
    }
  }
}

TEST(TransposeCollapsing, FuseTwoSecTranspose) {
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {4, 5, 3}, "X");
  auto x_1t = builder.Transpose(x, {0, 2, 1});
  auto x_2t = builder.Transpose(x_1t, {2, 1, 0});
  auto out1 = builder.Reshape(x_2t, {5, 3, 4});
  auto x_3t = builder.Transpose(out1, {0, 2, 1});
  auto x_4t = builder.Transpose(x_3t, {2, 1, 0});
  auto out2 = builder.Sqrt(x_4t);
  auto program = builder.Build();
  auto target = common::DefaultTarget();

  std::initializer_list<std::string> fetch_list = {out1->id, out2->id};

  size_t origin_size = program.size();
  VLOG(1) << "Program before pass:\n" << program;
  // Program {
  //   var_26 = transpose(X, axis=[0,2,1])
  //   var_27 = transpose(var_26, axis=[2,0,1])
  //   var_28 = sqrt(var_27)
  //   var_29 = transpose(var_28, axis=[0,2,1])
  //   var_30 = transpose(var_29, axis=[2,0,1])
  //   var_31 = sqrt(var_30)
  // }

  auto origin_out = RunWithProgram(program, target, {"X"}, fetch_list);

  ProgramPass::Apply(&program, fetch_list, target, {"TransposeCollapsing"});
  size_t folded_size = program.size();
  VLOG(1) << "Program after pass:\n" << program;
  // Program {
  //   var_27 = transpose(X, axis=[1,0,2])
  //   var_28 = sqrt(var_27)
  //   var_30 = transpose(var_28, axis=[1,0,2])
  //   var_31 = sqrt(var_30)
  // }

  auto folded_out = RunWithProgram(program, target, {"X"}, fetch_list);

  ASSERT_EQ(origin_size, folded_size + 2);
  ASSERT_EQ(origin_out.size(), folded_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_EQ(origin_out[i].size(), folded_out[i].size());
    for (size_t j = 0; j < origin_out[i].size(); ++j) {
      ASSERT_FLOAT_EQ(origin_out[i][j], folded_out[i][j]);
    }
  }
}

TEST(TransposeCollapsing, FuseTwoHorizontalTranspose) {
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {4, 5, 3}, "X");
  auto y_t1 = builder.Transpose(x, {0, 2, 1});
  auto y_t2 = builder.Transpose(x, {0, 2, 1});
  auto out = builder.Add(y_t1, y_t2);
  auto program = builder.Build();
  auto target = common::DefaultTarget();

  std::initializer_list<std::string> fetch_list = {out->id};

  size_t origin_size = program.size();
  VLOG(1) << "Program before pass:\n" << program;
  // Program {
  //   var_35 = transpose(X, axis=[0,2,1])
  //   var_36 = transpose(X, axis=[0,2,1])
  //   var_37 = elementwise_add(var_35, var_36)
  // }

  auto origin_out = RunWithProgram(program, target, {"X"}, fetch_list);

  ProgramPass::Apply(&program, fetch_list, target, {"TransposeCollapsing"});
  size_t folded_size = program.size();
  VLOG(1) << "Program after pass:\n" << program;
  // Program {
  //   var_36 = transpose(X, axis=[0,2,1])
  //   var_37 = elementwise_add(var_36, var_36)
  // }

  auto folded_out = RunWithProgram(program, target, {"X"}, fetch_list);

  ASSERT_EQ(origin_size - folded_size, 0);
  ASSERT_EQ(origin_out.size(), folded_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_EQ(origin_out[i].size(), folded_out[i].size());
    for (size_t j = 0; j < origin_out[i].size(); ++j) {
      ASSERT_FLOAT_EQ(origin_out[i][j], folded_out[i][j]);
    }
  }
}

TEST(TransposeCollapsing, FuseVerAndHorTranspose) {
  NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {4, 5, 3}, "X");
  auto y_t1 = builder.Transpose(x, {0, 2, 1});
  auto y_t2 = builder.Transpose(y_t1, {2, 1, 0});
  auto y_t3 = builder.Transpose(x, {1, 2, 0});
  auto out = builder.Add(y_t2, y_t3);
  auto program = builder.Build();
  auto target = common::DefaultTarget();

  std::initializer_list<std::string> fetch_list = {out->id};

  size_t origin_size = program.size();
  VLOG(1) << "Program before pass:\n" << program;
  // Program {
  //   var_40 = transpose(X, axis=[0,2,1])
  //   var_41 = transpose(var_40, axis=[2,1,0])
  //   var_42 = transpose(X, axis=[1,2,0])
  //   var_43 = elementwise_add(var_41, var_42)
  // }

  auto origin_out = RunWithProgram(program, target, {"X"}, fetch_list);

  ProgramPass::Apply(&program, fetch_list, target, {"TransposeCollapsing"});
  size_t folded_size = program.size();
  VLOG(1) << "Program after pass:\n" << program;
  // Program {
  //   var_42 = transpose(X, axis=[1,2,0])
  //   var_43 = elementwise_add(var_42, var_42)
  // }

  auto folded_out = RunWithProgram(program, target, {"X"}, fetch_list);

  ASSERT_EQ(origin_size - folded_size, 1);
  ASSERT_EQ(origin_out.size(), folded_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_EQ(origin_out[i].size(), folded_out[i].size());
    for (size_t j = 0; j < origin_out[i].size(); ++j) {
      ASSERT_FLOAT_EQ(origin_out[i][j], folded_out[i][j]);
    }
  }
}

}  // namespace cinn::frontend
