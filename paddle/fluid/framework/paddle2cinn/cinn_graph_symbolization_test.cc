/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "gtest/gtest.h"

#include "paddle/fluid/framework/paddle2cinn/cinn_graph_symbolization.h"

namespace paddle {
namespace framework {
namespace paddle2cinn {

using ir::Graph;
using ir::Node;
using CinnTensor = ::cinn::hlir::framework::Tensor;
using OpMapperContext = ::cinn::frontend::OpMapperContext;
using CinnOpDesc = ::cinn::frontend::paddle::cpp::OpDesc;

// only used for test CinnGraphSymbolization class
class CinnGraphSymbolizationForTest {
 public:
  void AddFeedInfoIntoContext(CinnGraphSymbolization* cinn_symbol_,
                              OpMapperContext* ctx) const {
    cinn_symbol_->AddFeedInfoIntoContext(ctx);
  }

  std::vector<std::unique_ptr<CinnOpDesc>> TransformAllGraphOpToCinn(
      CinnGraphSymbolization* cinn_symbol_) const {
    return cinn_symbol_->TransformAllGraphOpToCinn();
  }

  void RunOp(CinnGraphSymbolization* cinn_symbol_, const CinnOpDesc& op_desc,
             const OpMapperContext& ctx) const {
    cinn_symbol_->RunOp(op_desc, ctx);
  }

  std::shared_ptr<::cinn::hlir::framework::Scope> TransformPaddleScopeToCinn(
      CinnGraphSymbolization* cinn_symbol_) const {
    return cinn_symbol_->TransformPaddleScopeToCinn();
  }

  void RunGraph(CinnGraphSymbolization* cinn_symbol_,
                const OpMapperContext& ctx) const {
    cinn_symbol_->RunGraph(ctx);
  }
};

std::unique_ptr<Graph> BuildAllOpSupportCinnGraph() {
  ProgramDesc prog;
  auto g = std::make_unique<Graph>(prog);

  // v1 --
  //      | --> mul --> v3 --
  // v2 --                   | --> add --> v5 --> relu --> v6
  //                    v4 --

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

::cinn::common::Target CreateDefaultTarget(bool use_gpu = false) {
#ifdef PADDLE_WITH_CUDA
  if (use_gpu) {
    return ::cinn::common::DefaultNVGPUTarget();
  }
#endif
  return ::cinn::common::DefaultHostTarget();
}

std::unique_ptr<Scope> CreateScope() {
  std::unique_ptr<Scope> scope;
  scope->Var("var2");
  return scope;
}

std::map<std::string, LoDTensor> CreateFeedTarget() {
  std::map<std::string, LoDTensor> feed_targets;

  auto create_tensor = []() {
    LoDTensor tensor;
    DDim dims = {256, 1024, 1024};
    tensor.Resize(dims);
    return tensor;
  };
#define FillFeedList(Name) feed_targets[#Name] = create_tensor();

  FillFeedList(var1) FillFeedList(var3) FillFeedList(var4) FillFeedList(var5)
      FillFeedList(var6)
#undef FillFeedList
}

std::map<std::string, const LoDTensor*> ConvertFeedType(
    const std::map<std::string, LoDTensor>& feed_targets) {
  std::map<std::string, const LoDTensor*> res;
  for (auto& feed_pair : feed_targets) {
    res[feed_pair.first] = &feed_pair.second;
  }
  return res;
}

TEST(CinnGraphSymbolizationTest, basic) {
  auto graph = BuildAllOpSupportCinnGraph();
  auto scope = CreateScope();
  auto target = CreateDefaultTarget();
  auto feed_object = CreateFeedTarget();
  auto feed_targets = ConvertFeedType(feed_object);

  CinnGraphSymbolization symbol(100, *graph, *scope, target, feed_targets);
  ASSERT_NO_THROW(symbol());
  ASSERT_FALSE(symbol.var_map().empty());
  ASSERT_FALSE(symbol.var_model_to_program_map().empty());
}

}  // namespace paddle2cinn
}  // namespace framework
}  // namespace paddle
