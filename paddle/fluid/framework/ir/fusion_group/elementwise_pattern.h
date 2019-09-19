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

#include <string>
#include <vector>
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

struct ElementwiseGroupPattern : public PatternBase {
  ElementwiseGroupPattern(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "elementwise_group") {}

  void operator()(PDNode* x, int num_operations);

  std::vector<PDNode*> ops;
  std::vector<PDNode*> outputs;
};

bool IsElementwiseOp(Node* n);
bool IsInputOfElementwiseOp(Node* n);
bool IsOutputOfElementwiseOp(Node* n);
int NumAbjacentElementwiseOps(Node* n, std::vector<Node*> expect_nodes);

}  // namespace patterns
}  // namespace ir
}  // namespace framework
}  // namespace paddle
