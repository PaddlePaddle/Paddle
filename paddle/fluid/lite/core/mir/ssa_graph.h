// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <list>
#include <map>
#include <stack>
#include <string>
#include <vector>
#include "paddle/fluid/lite/core/mir/node.h"
#include "paddle/fluid/lite/core/op_lite.h"

namespace paddle {
namespace lite {
namespace mir {

// A program is used to represent a code program, in Paddle, a code program
// contains:
// - main block, which is a list of OpLite
// - scope: which contains all the weights
struct Program {
  std::list<std::string> tmp_vars;
  std::list<std::string> weights;
  std::list<std::unique_ptr<OpLite>> ops;
  lite::Scope *scope;
};

// An Graph for MIR. It is built from a list of Op and a scope.
class GraphBase {};

class SSAGraph : GraphBase {
 public:
  // @param program: the op program
  // @param valid_places: the valid places user set for the system.
  void Build(const Program &program, const std::vector<Place> &valid_places) {
    // create inputs
    for (const auto &name : program.tmp_vars) {
      node_storage_.emplace_back();
      auto &new_node = node_storage_.back();
      auto &arg = new_node.AsArgument();
      arg.name = name;
      arguments_[name] = &new_node;
    }

    for (auto &op : program.ops) {
      node_storage_.emplace_back();
      // TODO(Superjomn) remove one valid_places here.
      op->SetValidPlaces(valid_places);
      auto &new_node = node_storage_.back();
      auto &new_kernel = node_storage_.back().AsInstruct(op->op_type_);
      new_kernel.valid_kernels = op->CreateKernels(valid_places);

      CHECK(new_node.inlinks.empty()) << "duplicate Build found";
      CHECK(new_node.outlinks.empty()) << "duplicate Build found";

      // collect inputs and outputs
      for (const std::string &name : op->input_names()) {
        new_node.inlinks.push_back(arguments_.at(name));
      }
      for (const std::string &name : op->output_names()) {
        if (!arguments_.count(name)) {
          node_storage_.emplace_back();
          auto &new_node = node_storage_.back();
          auto &arg = new_node.AsArgument(name);
          arg.name = name;
          arguments_.emplace(name, &new_node);
        }
        new_node.outlinks.push_back(arguments_.at(name));
      }
    }
  }

  void sort_utils(mir::Node *n, std::map<mir::Node *, bool> &visited,
                  std::stack<mir::Node *> &stack) {
    visited[n] = true;
    for (auto &out : n->outlinks) {
      if (!visited[out]) {
        sort_utils(out, visited, stack);
      }
    }
  }

  std::vector<mir::Node *> TopoloticalOrder() {
    std::map<mir::Node *, bool> visited;
    std::stack<mir::Node *> stack;
    std::vector<mir::Node *> res;

    for (auto &n : mutable_nodes()) {
      if (!visited[&n]) sort_utils(&n, visited, stack);
    }

    while (!stack.empty()) {
      res.push_back(stack.top());
      stack.pop();
    }
    return res;
  }

  const std::list<mir::Node> &nodes() const { return node_storage_; }
  std::list<mir::Node> &mutable_nodes() { return node_storage_; }

 private:
  std::list<mir::Node> node_storage_;
  std::map<std::string, mir::Node *> arguments_;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
