// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/inference/analysis/helper.h"

namespace paddle {
namespace framework {
namespace ir {

void AddVarToScope(Scope* param_scope,
                   const std::string& name,
                   const DDim& dims) {
  auto* tensor = param_scope->Var(name)->GetMutable<phi::DenseTensor>();
  tensor->Resize(dims);
  auto* cpu_ctx = static_cast<phi::CPUContext*>(
      platform::DeviceContextPool::Instance().Get(phi::CPUPlace()));
  cpu_ctx->Alloc<float>(tensor);
}

VarDesc* Data(paddle::framework::BlockDesc* block,
              std::string name,
              std::vector<int64_t> shape = {},
              bool is_persistable = false,
              proto::VarType::Type data_type = proto::VarType::FP32) {
  auto* var = block->Var(name);
  var->SetType(proto::VarType::LOD_TENSOR);
  var->SetDataType(data_type);
  var->SetShape(shape);
  var->SetPersistable(is_persistable);
  return var;
}

TEST(SaveOptimizedModelPass, basic) {
  paddle::framework::ProgramDesc program;
  auto* block = program.MutableBlock(0);
  auto* lookup_table_w = Data(block, "lookup_table_w", {1}, true);
  auto* lookup_table_out = Data(block, "scatter_out", {1});
  OpDesc* lookup_table = block->AppendOp();
  lookup_table->SetType("lookup_table_v2");
  lookup_table->SetInput("W", {lookup_table_w->Name()});
  lookup_table->SetOutput("Out", {lookup_table_out->Name()});

  std::unique_ptr<ir::Graph> graph(new ir::Graph(program));
  auto scope = new Scope();
  AddVarToScope(scope, lookup_table_w->Name(), {1});
  graph->Set("__param_scope__", scope);

  auto save_optimized_model_pass =
      PassRegistry::Instance().Get("save_optimized_model_pass");
  save_optimized_model_pass->Set("save_optimized_model", new bool(true));
  save_optimized_model_pass->Set("model_opt_cache_dir", new std::string(""));
  save_optimized_model_pass->Apply(graph.get());
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(save_optimized_model_pass);
