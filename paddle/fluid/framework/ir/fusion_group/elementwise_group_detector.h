/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <vector>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/node.h"

namespace paddle {
namespace framework {
namespace ir {
class Node;
}  // namespace ir
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {
namespace fusion_group {

class GroupDetector {
 protected:
  bool CheckPrecondition(const Node* n);
};

class ElementwiseGroupDetector : GroupDetector {
 public:
  std::vector<std::vector<Node*>> operator()(Graph* graph);

 private:
  bool IsElementwiseOp(const Node* n);
};

}  // namespace fusion_group
}  // namespace ir
}  // namespace framework
}  // namespace paddle
