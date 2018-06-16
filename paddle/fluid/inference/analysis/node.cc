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
#include "glog/logging.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace inference {
namespace analysis {

std::vector<Dot::Attr> Value::dot_attrs() const {
  return std::vector<Dot::Attr>({Dot::Attr("style", "filled,rounded"),
                                 Dot::Attr("shape", "box"),
                                 Dot::Attr("fillcolor", "red")});
}

std::vector<Dot::Attr> Function::dot_attrs() const {
  return std::vector<Dot::Attr>({Dot::Attr("style", "filled,rounded"),
                                 Dot::Attr("shape", "diamond"),
                                 Dot::Attr("fillcolor", "yellow")});
}

Node *NodeMap::Create(Node::Type type) {
  switch (type) {
    case Node::Type::kFunction:
      nodes_.emplace_back(new Function);
      break;
    case Node::Type::kValue:
      nodes_.emplace_back(new Value);
      break;
    case Node::Type::kFunctionBlock:
      nodes_.emplace_back(new FunctionBlock);
      break;
    default:
      PADDLE_THROW("Not supported node type.");
  }
  nodes_.back()->id_ = size() - 1;
  return nodes_.back().get();
}

Node *NodeMap::GetMutable(size_t id) {
  PADDLE_ENFORCE_GT(size(), id);
  return nodes_[id].get();
}

const Node &NodeMap::Get(size_t id) const {
  PADDLE_ENFORCE_GT(size(), id);
  return *nodes_[id].get();
}

void NodeMap::Delete(size_t id) {
  PADDLE_ENFORCE_LT(id, size());
  nodes_[id]->SetDeleted();
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
