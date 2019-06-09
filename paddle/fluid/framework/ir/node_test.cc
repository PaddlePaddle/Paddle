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

#include <string>
#include "gtest/gtest.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

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

}  // namespace ir
}  // namespace framework
}  // namespace paddle
