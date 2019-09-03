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
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

class SimplifyWithBasicOpsPass : public Pass {
 protected:
  void ApplyImpl(Graph* graph) const override;

 private:
  bool SimplifyDropout(Graph* graph, Node* n,
                       std::unordered_set<const Node*>* del_node_set,
                       std::vector<int>* info) const;
  bool SimplifyStack(Graph* graph, Node* n,
                     std::unordered_set<const Node*>* del_node_set,
                     std::vector<int>* info) const;

  Node* GetInputVar(Node* n, const std::string& name) const;
  Node* GetOutputVar(Node* n, const std::string& name) const;

  void ReplaceInputVar(Node* op, Node* old_var, Node* new_var) const;
  void ReplaceOutputVar(Node* op, Node* old_var, Node* new_var) const;

  enum SimplifyOption {
    DELETE_DROPOIUT_OPS = 0,
    REPLACE_DROPOUT_WITH_SCALE_OPS = 1,
    REPLACE_STACK_WITH_CONCAT_OPS = 2,
    NUM_SUPPORT_OPTIONS = 3
  };

  void AddStatis(std::vector<int>* info, SimplifyOption option) const;
  void PrintStatis(const std::vector<int>& info) const;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
