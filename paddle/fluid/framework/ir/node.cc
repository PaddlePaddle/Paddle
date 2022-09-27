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

namespace paddle {
namespace framework {
namespace ir {
// msvc15 don't support constexpr in correct way.
#if !defined(_WIN32)
constexpr char Node::kControlDepVarName[];
#else
const char Node::kControlDepVarName[] = "__control_var";
#endif

std::unique_ptr<Node> CreateNodeForTest(const std::string &name,
                                        Node::Type type) {
  return std::unique_ptr<Node>(new Node(name, type));
}

std::unique_ptr<Node> CreateNodeForTest(VarDesc *var_desc) {
  return std::unique_ptr<Node>(new Node(var_desc, 0));
}

std::unique_ptr<Node> CreateNodeForTest(OpDesc *op_desc) {
  return std::unique_ptr<Node>(new Node(op_desc));
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
