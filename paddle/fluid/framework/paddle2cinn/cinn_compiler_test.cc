// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/paddle2cinn/cinn_compiler.h"

#include <map>
#include <memory>
#include <string>

#include "cinn/common/target.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace paddle2cinn {

using ir::Graph;
using ::cinn::common::Target;

// v1 --
//      | --> mul --> v3 --
// v2 --                   | --> add --> v5 --> relu --> v6
//                    v4 --
std::unique_ptr<Graph> CreateGraph() {
  ProgramDesc prog;
  auto g = std::make_unique<Graph>(prog);

  OpDesc add_op;
  add_op.SetType("add");
  OpDesc mul_op;
  mul_op.SetType("mul");
  OpDesc relu_op;
  relu_op.SetType("relu");

  VarDesc var1("var1");
  VarDesc var2("var2");
  var2.SetPersistable(true);
  var2.SetIsParameter(true);
  VarDesc var3("var3");
  VarDesc var4("var4");
  var4.SetPersistable(true);
  var4.SetIsParameter(true);
  VarDesc var5("var5");
  VarDesc var6("var6");

  ir::Node* add = g->CreateOpNode(&add_op);
  ir::Node* mul = g->CreateOpNode(&mul_op);
  ir::Node* relu = g->CreateOpNode(&relu_op);

  ir::Node* v1 = g->CreateVarNode(&var1);
  ir::Node* v2 = g->CreateVarNode(&var2);
  ir::Node* v3 = g->CreateVarNode(&var3);
  ir::Node* v4 = g->CreateVarNode(&var4);
  ir::Node* v5 = g->CreateVarNode(&var5);
  ir::Node* v6 = g->CreateVarNode(&var6);

  // fill op node
  mul->inputs = {v1, v2};
  mul->outputs = {v3};
  add->inputs = {v3, v4};
  add->outputs = {v5};
  relu->inputs = {v5};
  relu->outputs = {v6};

  // fill variable node
  v1->outputs = {mul};
  v2->outputs = {mul};

  v3->inputs = {mul};
  v3->outputs = {add};

  v4->outputs = {add};

  v5->inputs = {add};
  v5->outputs = {relu};

  v6->inputs = {relu};

  return g;
}

TEST(CinnCompilerTest, TodoTest) {
  auto* cinn_compiler = CinnCompiler::GetInstance();
  std::string compilation_key = cinn_compiler->AddGraph(CreateGraph());
  auto* graph = cinn_compiler->FindGraph(compilation_key);
  ASSERT_NE(graph, nullptr);
  EXPECT_THROW(cinn_compiler->FindGraph("no_existed"),
               paddle::platform::EnforceNotMet);
  LoDTensor tensor1, tensor2, tensor3;
  tensor1.Resize({8, 32});
  tensor2.Resize({32, 64});
  tensor3.Resize({64});
  std::map<std::string, const LoDTensor*> input_tensors = {
      {"input", &tensor1}, {"fc_w", &tensor2}, {"fc_b", &tensor3}};

  auto compile_fn = [&cinn_compiler](const Target& target) {
    auto* compiled_obj = cinn_compiler->Compile(*graph, input_tensors, target);
    ASSERT_NE(compiled_obj->runtime_program.get(), nullptr);
    ASSERT_NE(compiled_obj->scope.get(), nullptr);
    ASSERT_FALSE(compiled_obj->paddle2cinn_varmap.empty());
    cinn_compiler->Compile(compilation_key, input_tensors, target);
    ASSERT_EQ(cinn_compiler->real_compiled_num(), 1);
  };

  compile_fn(::cinn::common::DefaultNVGPUTarget());
  compile_fn(::cinn::common::DefaultHostTarget());
}

}  // namespace paddle2cinn
}  // namespace framework
}  // namespace paddle
