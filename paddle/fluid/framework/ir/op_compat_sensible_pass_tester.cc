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
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace framework {
namespace ir {

TEST(OpCompatSensiblePass, compatOp) {
  auto lambda = [](const std::string& str) { return str == "tanh"; };
  OpCompat compat("FC");
  compat.AddAttr("in_num_col_dims")
      .IsIntIn({1, 2})
      .IsNumLE(1)
      .IsLeftDefault()
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

  EXPECT_STREQ(compat.Name().c_str(), "FC");
  EXPECT_TRUE(compat.Judge(fc_op));
}

class OpCompatSensiblePassTest : public OpCompatSensiblePass {
 public:
  OpCompatSensiblePassTest();
  bool TestIsCompat(const OpDesc& op_desc) { return IsCompat(op_desc); }
};

OpCompatSensiblePassTest::OpCompatSensiblePassTest() {
  AddOpCompat(OpCompat("FC"))
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
  OpCompatSensiblePassTest test;
  OpDesc fc_op;
  fc_op.SetType("FC");
  std::unordered_map<std::string, Attribute> attr_map;
  attr_map["in_num_col_dims"] = 1;
  attr_map["activation_type"] = std::string("tanh");

  fc_op.SetAttrMap(attr_map);
  fc_op.SetInput("Input", std::vector<std::string>{"test_input"});
  fc_op.SetInput("W", std::vector<std::string>{"test_input_0"});
  fc_op.SetInput("Bias", std::vector<std::string>{"test_input_1"});
  fc_op.SetOutput("Out", std::vector<std::string>{"test_output"});

  EXPECT_TRUE(test.TestIsCompat(fc_op));

  ProgramDesc prog;
  std::unique_ptr<Graph> g(new Graph(prog));
  Node* o1 = g->CreateOpNode(&fc_op);

  GraphPatternDetector detector;
  PDNode* op2 =
      detector.mutable_pattern()->NewNode([](Node* x) { return true; });
  GraphPatternDetector::subgraph_t subgraph;
  subgraph[op2] = o1;

  test.AccessSubgraph(subgraph, g.get());
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
