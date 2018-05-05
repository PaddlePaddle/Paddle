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

unsigned Node::counter_ = 0;
std::string Value::repr() const { return std::string(); }

std::string Function::repr() const { return std::string(); }

Node *NodeMap::Create(Node::Type type) {
  switch (type) {
    case Node::Type::kFunction:
      nodes_.emplace_back(new Function);
      break;
    case Node::Type::kValue:
      nodes_.emplace_back(new Value);
      break;
    default:
      PADDLE_ENFORCE(false, "Not supported node type.");
      return nullptr;
  }
  CHECK_EQ(nodes_.back()->id() + 1, size());
  return nodes_.back().get();
}

Node *NodeMap::Get(size_t id) {
  PADDLE_ENFORCE_GT(size(), id);
  return nodes_[id].get();
}

void NodeMap::Delete(size_t id) {
  PADDLE_ENFORCE_LT(id, size());
  nodes_[id]->SetDeleted();
}
}
}
}  // namespace paddle
