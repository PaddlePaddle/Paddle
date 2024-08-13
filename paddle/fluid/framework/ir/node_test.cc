/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/node.h"

#include "gtest/gtest.h"
#include "paddle/fluid/framework/var_desc.h"

namespace paddle::framework::ir {

class Node;

class RunnableOp {
 public:
  RunnableOp(Node* node, bool* alive) : node_(node), alive_(alive) {
    node_->WrappedBy(this);
  }

  virtual ~RunnableOp() { *alive_ = false; }

 private:
  Node* node_;
  bool* alive_;
};

class RunnableOp2 {
 public:
  RunnableOp2(Node* node, bool* alive) : node_(node), alive_(alive) {
    node_->WrappedBy(this);
  }

  virtual ~RunnableOp2() { *alive_ = false; }

 private:
  Node* node_;
  bool* alive_;
};

TEST(NodeTest, Basic) {
  bool alive1 = true;
  bool alive2 = true;
  std::unique_ptr<Node> n1(CreateNodeForTest("n1", Node::Type::kVariable));
  std::unique_ptr<Node> n2(CreateNodeForTest("n2", Node::Type::kVariable));

  EXPECT_FALSE(n1->IsWrappedBy<RunnableOp>());
  EXPECT_FALSE(n1->IsWrappedBy<RunnableOp2>());
  EXPECT_FALSE(n2->IsWrappedBy<RunnableOp>());
  EXPECT_FALSE(n2->IsWrappedBy<RunnableOp2>());

  new RunnableOp(n1.get(), &alive1);
  new RunnableOp2(n2.get(), &alive2);

  EXPECT_TRUE(n1->IsWrappedBy<RunnableOp>());
  EXPECT_FALSE(n1->IsWrappedBy<RunnableOp2>());
  EXPECT_FALSE(n2->IsWrappedBy<RunnableOp>());
  EXPECT_TRUE(n2->IsWrappedBy<RunnableOp2>());

  EXPECT_TRUE(alive1);
  EXPECT_TRUE(alive2);

  n1.reset(nullptr);
  n2.reset(nullptr);
  EXPECT_FALSE(alive1);
  EXPECT_FALSE(alive2);
}

TEST(NodeTest, ToString) {
  VarDesc var_desc("n2");
  OpDesc op_desc;
  op_desc.SetType("test_op");
  op_desc.SetInput("X", {"x1", "x2", "x3"});
  op_desc.SetOutput("Y", {"y1", "y2"});

  std::unique_ptr<Node> n1(CreateNodeForTest("n1", Node::Type::kVariable));
  std::unique_ptr<Node> n2(CreateNodeForTest(&var_desc));
  std::unique_ptr<Node> n3(CreateNodeForTest("n3", Node::Type::kOperation));
  std::unique_ptr<Node> n4(CreateNodeForTest(&op_desc));

  EXPECT_EQ(n1->ToString(), "n1");
  EXPECT_EQ(n2->ToString(), "n2");

  EXPECT_EQ(n3->Op(), nullptr);
  EXPECT_EQ(n3->ToString(), "{} = n3()");
  EXPECT_NE(n4->Op(), nullptr);
  EXPECT_EQ(n4->ToString(), "{Y=[y1 ,y2]} = test_op(X=[x1 ,x2 ,x3])");

  n3->inputs.push_back(n1.get());
  n3->outputs.push_back(n2.get());
  EXPECT_EQ(n3->Op(), nullptr);
  EXPECT_EQ(n3->ToString(), "{n2} = n3(n1)");
}

}  // namespace paddle::framework::ir
