// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/op_compat_sensible_pass.h"

namespace paddle {
namespace framework {
namespace ir {

class Graph;

class DeleteDropoutOpPass : public FusePassBase {
 public:
  virtual ~DeleteDropoutOpPass() {}

 protected:
  void ApplyImpl(ir::Graph* graph) const override;
};

class DeleteDropoutOpXPass : public OpCompatSensiblePass {
 public:
  DeleteDropoutOpXPass();
  virtual ~DeleteDropoutOpXPass() {}

 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  bool DelDropout(Graph* graph,
                  Node* n,
                  std::unordered_set<const Node*>* del_node_set) const;
  Node* GetInputVar(Node* n, const std::string& name) const;
  Node* GetOutputVar(Node* n, const std::string& name) const;
  void ReplaceInputVar(Node* op, Node* old_var, Node* new_var) const;
  void ReplaceOutputVar(Node* op, Node* old_var, Node* new_var) const;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
