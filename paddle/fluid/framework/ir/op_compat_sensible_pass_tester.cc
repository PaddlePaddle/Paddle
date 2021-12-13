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

#include "paddle/fluid/framework/ir/op_compat_sensible_pass.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace framework {
namespace ir {

TEST(OpCompatSensiblePass, compatOp) {
  auto lambda = [](const std::string& str) { return str == "tanh"; };
  OpCompat compat("fc_test");
  compat.AddAttr("in_num_col_dims")
      .IsIntIn({1, 2})
      .IsNumLE(1)
      .End()
      .AddAttr("activation_type")
      .IsStringIn({"tanh", "sigmoid"})
      .IsStringMatch(lambda)
      .End()
      .AddAttr("test_attr")
      .IsBoolEQ(true)
      .End()
      .AddInput("Input")
      .IsTensor()
      .End()
      .AddInput("W")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .IsOptional()
      .End()
      .AddInput("Test")
      .IsOptional()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End();

  OpDesc fc_op;

  std::unordered_map<std::string, Attribute> attr_map;
  attr_map["in_num_col_dims"] = 1;
  attr_map["activation_type"] = std::string("tanh");
  attr_map["test_attr"] = true;

  fc_op.SetAttrMap(attr_map);

  fc_op.SetInput("Input", std::vector<std::string>{"test_input"});
  fc_op.SetInput("W", std::vector<std::string>{"test_input_0"});
  fc_op.SetInput("Bias", std::vector<std::string>{"test_input_1"});
  fc_op.SetOutput("Out", std::vector<std::string>{"test_output"});

  OpInfo info;
  info.proto_ = new proto::OpProto;
  info.proto_->set_type("fc_test");
  info.proto_->set_comment("");
  auto* attr = info.proto_->add_attrs();
  attr->set_name("in_num_col_dims");
  attr = info.proto_->add_attrs();
  attr->set_name("test_attr");
  OpInfoMap::Instance().Insert("fc_test", info);

  EXPECT_STREQ(compat.Name().c_str(), "fc_test");
  EXPECT_TRUE(compat.Judge(fc_op, "test_pass"));

  delete info.proto_;
  OpInfoMap::Instance().mutable_map()->erase("fc_test");
}

TEST(OpCompatSensiblePass, compatOpAttribute) {
  OpCompat compat("fc_test");

  OpDesc fc_op;
  std::unordered_map<std::string, Attribute> attr_map;
  attr_map["in_num_col_dims"] = 1;
  fc_op.SetAttrMap(attr_map);

  OpInfo info;
  info.proto_ = new proto::OpProto;
  info.proto_->set_type("fc_test");
  info.proto_->set_comment("");
  auto* attr = info.proto_->add_attrs();
  attr->set_name("in_num_col_dims");
  info.checker_ = new OpAttrChecker();
  OpInfoMap::Instance().Insert("fc_test", info);
  EXPECT_FALSE(compat.Judge(fc_op, "test_pass"));

  OpCompat compat_1("fc_test");
  info.checker_->AddAttrChecker<int>("in_num_col_dims", nullptr).SetDefault(1);
  EXPECT_TRUE(compat_1.Judge(fc_op, "test_pass"));
  delete info.checker_;
  delete info.proto_;
  OpInfoMap::Instance().mutable_map()->erase("fc_test");
}

TEST(OpCompatSensiblePass, opDefNotFound) {
  OpCompat compat("fc_test");

  OpDesc fc_op;
  OpInfo info;
  info.proto_ = new proto::OpProto;
  info.proto_->set_type("fc_test");
  info.proto_->set_comment("");
  OpInfoMap::Instance().Insert("fc_test", info);
  compat.Judge(fc_op, "test_pass");
  delete info.proto_;
  OpInfoMap::Instance().mutable_map()->erase("fc_test");
}

TEST(OpCompatSensiblePass, compatOpAttributeOptional) {
  OpCompat compat("fc_test");
  compat.AddAttr("activation_type")
      .IsOptional()
      .IsStringIn({"tanh", "sigmoid"});
  OpDesc fc_op;
  OpInfo info;
  info.proto_ = new proto::OpProto;
  info.proto_->set_type("fc_test");
  info.proto_->set_comment("");
  auto* attr = info.proto_->add_attrs();
  attr->set_name("activation_type");
  OpInfoMap::Instance().Insert("fc_test", info);
  EXPECT_TRUE(compat.Judge(fc_op, "test_pass"));
  delete info.proto_;
  OpInfoMap::Instance().mutable_map()->erase("fc_test");
}

TEST(OpCompatSensiblePass, compatOpInput) {
  OpInfo info;
  info.proto_ = new proto::OpProto;
  info.proto_->set_type("fc_test");
  info.proto_->set_comment("");
  OpInfoMap::Instance().Insert("fc_test", info);

  OpCompat compat("fc_test");

  OpDesc fc_op;
  fc_op.SetInput("Input", std::vector<std::string>{"test_input"});

  EXPECT_FALSE(compat.Judge(fc_op, "test_pass"));

  compat.AddInput("Input").IsTensor().End().AddInput("Bias").IsTensor().End();
  EXPECT_FALSE(compat.Judge(fc_op, "test_pass"));

  fc_op.SetInput("Bias", std::vector<std::string>{"test_input", ""});
  EXPECT_FALSE(compat.Judge(fc_op, "test_pass"));

  delete info.proto_;
  OpInfoMap::Instance().mutable_map()->erase("fc_test");
}

TEST(OpCompatSensiblePass, compatOutput) {
  OpInfo info;
  info.proto_ = new proto::OpProto;
  info.proto_->set_type("fc_test");
  info.proto_->set_comment("");
  OpInfoMap::Instance().Insert("fc_test", info);

  OpCompat compat("fc_test");

  OpDesc fc_op;
  fc_op.SetOutput("Output", std::vector<std::string>{"test_output"});

  EXPECT_FALSE(compat.Judge(fc_op, "test_pass"));

  compat.AddOutput("Output")
      .IsTensor()
      .End()
      .AddOutput("Output_2")
      .IsTensor()
      .End();
  EXPECT_FALSE(compat.Judge(fc_op, "test_pass"));

  fc_op.SetOutput("Output_2", std::vector<std::string>{"test_output", ""});
  EXPECT_FALSE(compat.Judge(fc_op, "test_pass"));

  delete info.proto_;
  OpInfoMap::Instance().mutable_map()->erase("fc_test");
}

class OpCompatSensiblePassTest : public OpCompatSensiblePass {
 public:
  OpCompatSensiblePassTest();
  bool TestIsCompat(const OpDesc& op_desc) { return IsCompat(op_desc); }
  bool TestIsCompat(const GraphPatternDetector::subgraph_t& subgraph,
                    Graph* g) {
    return IsCompat(subgraph, g);
  }
};

OpCompatSensiblePassTest::OpCompatSensiblePassTest() {
  AddOpCompat(OpCompat("fc_test"))
      .AddAttr("in_num_col_dims")
      .IsNumLE(1)
      .End()
      .AddAttr("activation_type")
      .IsStringIn({"tanh", "sigmoid"})
      .End()
      .AddInput("Input")
      .IsTensor()
      .End()
      .AddInput("W")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Out")
      .IsTensor();
}

TEST(OpCompatSensiblePass, IsCompat) {
  OpInfo info;
  info.proto_ = new proto::OpProto;
  info.proto_->set_type("fc_test");
  info.proto_->set_comment("");
  auto* attr = info.proto_->add_attrs();
  attr->set_name("in_num_col_dims");
  attr = info.proto_->add_attrs();
  attr->set_name("activation_type");
  OpInfoMap::Instance().Insert("fc_test", info);

  OpCompatSensiblePassTest test;
  OpDesc fc_op;
  fc_op.SetType("fc_test");
  std::unordered_map<std::string, Attribute> attr_map;
  attr_map["in_num_col_dims"] = 1;
  attr_map["activation_type"] = std::string("tanh");

  fc_op.SetAttrMap(attr_map);
  fc_op.SetInput("Input", std::vector<std::string>{"test_input"});
  fc_op.SetInput("W", std::vector<std::string>{"test_input_0"});
  fc_op.SetInput("Bias", std::vector<std::string>{"test_input_1"});
  fc_op.SetOutput("Out", std::vector<std::string>{"test_output"});

  EXPECT_TRUE(test.TestIsCompat(fc_op));

  delete info.proto_;
  OpInfoMap::Instance().mutable_map()->erase("fc_test");
}

TEST(OpCompatSensiblePass, IsCompatFail) {
  OpInfo info;
  info.proto_ = new proto::OpProto;
  info.proto_->set_type("fc_test");
  info.proto_->set_comment("");
  auto* attr = info.proto_->add_attrs();
  attr->set_name("activation_type");
  attr = info.proto_->add_attrs();
  attr->set_name("in_num_col_dims");
  OpInfoMap::Instance().Insert("fc_test", info);
  OpInfoMap::Instance().Insert("op2", info);

  OpCompatSensiblePassTest test;
  GraphPatternDetector::subgraph_t subgraph;
  PDPattern pattern;
  PDNode* pd_node = pattern.NewNode();
  ProgramDesc prog;
  Graph g(prog);
  OpDesc fc_op;
  std::unordered_map<std::string, Attribute> attr_map;
  attr_map["in_num_col_dims"] = 1;
  attr_map["activation_type"] = std::string("tanh");
  fc_op.SetAttrMap(attr_map);
  fc_op.SetType("fc_test");
  subgraph[pd_node] = g.CreateOpNode(&fc_op);
  EXPECT_FALSE(test.TestIsCompat(subgraph, &g));

  fc_op.SetType("op2");
  subgraph[pd_node] = g.CreateOpNode(&fc_op);
  EXPECT_TRUE(test.TestIsCompat(subgraph, &g));

  delete info.proto_;
  OpInfoMap::Instance().mutable_map()->erase("fc_test");
  OpInfoMap::Instance().mutable_map()->erase("op2");
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
