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

#include "paddle/fluid/inference/analysis/node.h"

#include <gtest/gtest.h>

namespace paddle {
namespace inference {
namespace analysis {

TEST(NodeAttr, bool) {
  AnyAttr x;
  x.Bool() = true;
  ASSERT_EQ(x.Bool(), true);
}

TEST(NodeAttr, int32) {
  AnyAttr x;
  x.Int32() = 32;
  ASSERT_EQ(x.Int32(), 32);
}

TEST(NodeAttr, string) {
  AnyAttr x;
  x.String() = "Hello";
  ASSERT_EQ(x.String(), "Hello");
}

TEST(Node, Attr) {
  // Node is an abstract class, use Value instead for they share the same Attr
  // logic.
  NodeMap nodes;
  auto* node = nodes.Create(Node::Type::kValue);
  node->attr("v0").Int32() = 2008;
  ASSERT_EQ(node->attr("v0").Int32(), 2008);

  node->attr("str").String() = "hello world";
  ASSERT_EQ(node->attr("str").String(), "hello world");
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
