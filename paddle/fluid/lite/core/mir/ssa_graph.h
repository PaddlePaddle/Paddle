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
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/mir/node.h"
#include "paddle/fluid/lite/core/op_lite.h"
#include "paddle/fluid/lite/core/program.h"

namespace paddle {
namespace lite {
namespace mir {

// An Graph for MIR. It is built from a list of Op and a scope.
class GraphBase {};

class SSAGraph : GraphBase {
 public:
  // @param program: the op program
  // @param valid_places: the valid places user set for the system.
  void Build(const Program &program, const std::vector<Place> &valid_places) {
    // create temporary nodes.
    for (const auto &name : program.tmp_vars) {
      node_storage_.emplace_back();
      auto &new_node = node_storage_.back();
      auto &arg = new_node.AsArgument();
      arg.name = name;
      arguments_[name] = &new_node;
    }

    // create weight nodes.
    for (const auto &name : program.weights) {
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
      auto kernels = op->CreateKernels(valid_places);
      for (auto &kernel : kernels) {
        op->AttachKernel(kernel.get());
      }
      node_storage_.back().AsInstruct(op->op_type_, std::move(kernels), op,
                                      op->op_info());

      CHECK(new_node.inlinks.empty()) << "duplicate Build found";
      CHECK(new_node.outlinks.empty()) << "duplicate Build found";

      // collect inputs and outputs
      for (const std::string &name : op->op_info()->input_names()) {
        auto *arg = Argument(name);
        new_node.inlinks.push_back(arg);
        arg->outlinks.push_back(&new_node);
      }
      for (const std::string &name : op->op_info()->output_names()) {
        if (!arguments_.count(name)) {
          node_storage_.emplace_back();
          auto &new_node = node_storage_.back();
          auto &arg = new_node.AsArgument(name);
          arg.name = name;
          arguments_.emplace(name, &new_node);
        }
        auto *arg = arguments_.at(name);
        new_node.outlinks.push_back(arg);
        arg->inlinks.push_back(&new_node);
      }
    }

    MarkArgumentWeights(program);
  }

  mir::Node *Argument(const std::string &name) {
    auto it = arguments_.find(name);
    CHECK(it != arguments_.end()) << "no argument called " << name;
    return it->second;
  }

  std::vector<mir::Node *> InstructTopologicalOrder();

  // The inputs of the graph.
  std::vector<mir::Node *> inputs() {
    std::vector<mir::Node *> res;
    for (auto &node : node_storage_) {
      if (node.inlinks.empty()) {
        res.push_back(&node);
      }
    }
    return res;
  }

  // The outputs of the graph.
  std::vector<mir::Node *> outputs() {
    std::vector<mir::Node *> res;
    for (auto &node : node_storage_) {
      if (node.outlinks.empty()) {
        res.push_back(&node);
      }
    }
    return res;
  }

  const std::list<mir::Node> &nodes() const { return node_storage_; }
  std::list<mir::Node> &mutable_nodes() { return node_storage_; }

  mir::Node *RetrieveArgument(const std::string &arg) {
    auto it = arguments_.find(arg);
    if (it != arguments_.end()) {
      return it->second;
    }
    return nullptr;
  }

 private:
  // Check the bidirectional connection.
  bool CheckBidirectionalConnection();

  void MarkArgumentWeights(const Program &program) {
    for (const auto &name : program.weights) {
      arguments_[name]->AsArgument().is_weight = true;
    }
  }

  // Build operator inlink edge table.
  std::map<mir::Node *, std::set<mir::Node *>> BuildOperationAdjList();

  void SortHelper(const std::map<mir::Node *, std::set<mir::Node *>> &adj_list,
                  mir::Node *node, std::set<mir::Node *> *visited,
                  std::vector<mir::Node *> *ret);

 private:
  std::list<mir::Node> node_storage_;
  std::map<std::string, mir::Node *> arguments_;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
