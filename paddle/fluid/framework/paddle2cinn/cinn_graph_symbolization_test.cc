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
using ::cinn::frontend::NetBuilder;
using CinnTensor = ::cinn::hlir::framework::Tensor;
using OpMapperContext = ::cinn::frontend::OpMapperContext;
using CinnOpDesc = ::cinn::frontend::paddle::cpp::OpDesc;
using FeedInfoMap = absl::flat_hash_map<std::string, OpMapperContext::FeedInfo>;

// only used for test CinnGraphSymbolization class
class CinnGraphSymbolizationForTest {
 public:
  explicit CinnGraphSymbolizationForTest(CinnGraphSymbolization* cinn_symbol)
      : cinn_symbol_(cinn_symbol) {}

  std::unordered_set<std::string> GetGraphInputParameterNames() {
    return cinn_symbol_->GetGraphInputParameterNames();
  }

  std::shared_ptr<::cinn::hlir::framework::Scope> CreateCinnScope(
      const FeedInfoMap& feed_map) {
    return cinn_symbol_->CreateCinnScope(feed_map);
  }

  OpMapperContext CreateNewContext(NetBuilder* builder,
                                   const FeedInfoMap& feed_map) {
    return OpMapperContext(*cinn_symbol_->CreateCinnScope(feed_map),
                           cinn_symbol_->target_, builder,
                           &cinn_symbol_->var_map_,
                           &cinn_symbol_->var_model_to_program_map_);
  }

  FeedInfoMap GetFeedInfoMapFromInput() {
    return cinn_symbol_->GetFeedInfoMapFromInput();
  }

  std::vector<std::unique_ptr<CinnOpDesc>> TransformAllGraphOpToCinn() {
    return cinn_symbol_->TransformAllGraphOpToCinn();
  }

  void RunOp(const CinnOpDesc& op_desc, const OpMapperContext& ctx) {
    cinn_symbol_->RunOp(op_desc, ctx);
  }

 private:
  CinnGraphSymbolization* cinn_symbol_;
};

class CinnGraphSymbolizationTest : public ::testing::Test {
 public:
  std::unique_ptr<CinnGraphSymbolization> symbol_;
  std::unique_ptr<CinnGraphSymbolizationForTest> test_;

  OpMapperContext CreateNewContext() {
    return test_->CreateNewContext(builder_.get(), feed_map_);
  }

  std::shared_ptr<::cinn::hlir::framework::Scope> CreateCinnScope() {
    return test_->CreateCinnScope(feed_map_);
  }

 protected:
  void SetUp() override {
    int64_t graph_id = 100;
    graph_ = BuildAllOpSupportCinnGraph();
    target_ = CreateDefaultTarget();
    feed_tensors_ = CreateFeedTensors();
    feed_targets_ = ConvertFeedType(feed_tensors_);
    symbol_ = std::make_unique<CinnGraphSymbolization>(graph_id, *graph_,
                                                       target_, feed_targets_);
    builder_ = std::make_unique<NetBuilder>("NetBuilder_of_graph_" +
                                            std::to_string(graph_id));
    test_ = std::make_unique<CinnGraphSymbolizationForTest>(symbol_.get());
    feed_map_ = test_->GetFeedInfoMapFromInput();
  }

 private:
  std::unique_ptr<Graph> graph_;
  ::cinn::common::Target target_;
  std::map<std::string, LoDTensor> feed_tensors_;
  std::map<std::string, const LoDTensor*> feed_targets_;
  std::unique_ptr<NetBuilder> builder_;
  FeedInfoMap feed_map_;

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

  std::map<std::string, LoDTensor> CreateFeedTensors() {
    std::map<std::string, LoDTensor> feed_targets;

    auto create_tensor = []() {
      LoDTensor tensor;
      DDim dims = {256, 1024, 1024};
      tensor.Resize(dims);
      return tensor;
    };
#define FillFeedList(Name) feed_targets[#Name] = create_tensor();
    FillFeedList(var1);
    FillFeedList(var2);
    FillFeedList(var3);
    FillFeedList(var4);
    FillFeedList(var5);
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
};

TEST_F(CinnGraphSymbolizationTest, basic) {
  ASSERT_NO_THROW((*symbol_)());
  ASSERT_FALSE(symbol_->var_map().empty());
  ASSERT_FALSE(symbol_->var_model_to_program_map().empty());
}

TEST_F(CinnGraphSymbolizationTest, feed_map) {
  auto feed_map = test_->GetFeedInfoMapFromInput();
  auto ctx = CreateNewContext();

  ASSERT_TRUE(feed_map.count("var1"));
  ASSERT_TRUE(feed_map.count("var2"));

  auto feed_info = feed_map.at("var1");
  ASSERT_EQ(feed_info.shape, std::vector<int>({256, 1024, 1024}));
  ASSERT_EQ(feed_info.type, ::cinn::common::F32());
}

TEST_F(CinnGraphSymbolizationTest, scope) {
  auto prame_names = test_->GetGraphInputParameterNames();
  ASSERT_EQ(prame_names, std::unordered_set<std::string>({"var2"}));

  auto cinn_scope = CreateCinnScope();

  auto* var1 = cinn_scope->FindVar("var1");
  ASSERT_EQ(var1, nullptr);
  auto* var2 = cinn_scope->FindVar("var2");
  ASSERT_NE(var2, nullptr);

  auto& cinn_tensor = absl::get<CinnTensor>(*var2);
  ASSERT_EQ(cinn_tensor->shape().data(), std::vector<int>({256, 1024, 1024}));
  ASSERT_EQ(cinn_tensor->type(), ::cinn::common::F32());
}

TEST_F(CinnGraphSymbolizationTest, sortgraph) {
  auto cinn_op_descs = test_->TransformAllGraphOpToCinn();
  ASSERT_FALSE(cinn_op_descs.empty());
  std::vector<std::string> sort_names;
  for (auto& desc : cinn_op_descs) {
    sort_names.emplace_back(desc->Type());
  }
  ASSERT_EQ(sort_names, std::vector<std::string>({"mul", "add", "relu"}));
}

TEST_F(CinnGraphSymbolizationTest, runop) {
  auto cinn_op_descs = test_->TransformAllGraphOpToCinn();
  auto ctx = CreateNewContext();
  ASSERT_NO_THROW(test_->RunOp(*cinn_op_descs[0], ctx));

  CinnOpDesc desc;
  desc.SetType("fake");
  ASSERT_ANY_THROW(test_->RunOp(desc, ctx));
}

}  // namespace paddle2cinn
}  // namespace framework
}  // namespace paddle
