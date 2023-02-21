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

#pragma once
#include <string>

namespace paddle {
namespace framework {
namespace ir {

#define GET_IR_NODE(node_) SAFE_GET_IR_NODE_FROM_SUBGRAPH(node_, node_, pattern)

// Get an ir::Node* from the matched subgraph.
// var: variable.
// arg: the argument declared by PATTERN_DECL_NODE in a pattern definition.
// pat: the pattern object.
#define SAFE_GET_IR_NODE_FROM_SUBGRAPH(var, arg, pat)                          \
  Node* var = nullptr;                                                         \
  if (pat.arg##_n()) {                                                         \
    PADDLE_ENFORCE_NE(subgraph.count(pat.arg##_n()),                           \
                      0UL,                                                     \
                      platform::errors::NotFound(                              \
                          "Node not found for PDNode %s", pat.arg##_repr()));  \
    var = subgraph.at(pat.arg##_n());                                          \
    PADDLE_ENFORCE_NOT_NULL(var,                                               \
                            platform::errors::NotFound(                        \
                                "node %s not exists in the sub-graph", #arg)); \
  }

#define SAFE_IR_NODE_LINK_TO(a, b)    \
  if (a != nullptr && b != nullptr) { \
    IR_NODE_LINK_TO(a, b)             \
  }

int ConvertActivationType(std::string act_type);

}  // namespace ir
}  // namespace framework
}  // namespace paddle
