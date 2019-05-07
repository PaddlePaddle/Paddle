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
  void Build(const Program &program, const std::vector<Place> &valid_places);

  mir::Node *Argument(const std::string &name);

  std::vector<mir::Node *> StmtTopologicalOrder();

  // The inputs of the graph.
  std::vector<mir::Node *> inputs();

  // The outputs of the graph.
  std::vector<mir::Node *> outputs();

  const std::list<mir::Node> &nodes() const { return node_storage_; }
  std::list<mir::Node> &mutable_nodes() { return node_storage_; }

  mir::Node *RetrieveArgument(const std::string &arg);

  Node *NewArgumentNode(const std::string &name);
  Node *NewInstructNode();

  void CheckValid() {
    CHECK(CheckBidirectionalConnection());
    CHECK(CheckNodesRoleSet());
    CHECK(CheckLinksRoleSet());
  }

 private:
  void GraphCreateTmpVarNodes(const Program &program);
  void GraphCreateWeightVarNodes(const Program &program);
  Node *GraphCreateInstructNode(const Program &program,
                                const std::shared_ptr<OpLite> &op,
                                const std::vector<Place> &valid_places);

  // Check the bidirectional connection.
  bool CheckBidirectionalConnection();
  bool CheckNodesRoleSet();
  // Check all the items's role in inlinks and outlinks is set.
  bool CheckLinksRoleSet();

  void MarkArgumentWeights(const Program &program) {
    for (const auto &name : program.weights) {
      arguments_[name]->AsArg().is_weight = true;
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

// Remove the link between a -> b.
static void RemoveDirectedLink(Node *a, Node *b) {
  auto it = std::find(b->inlinks.begin(), b->inlinks.end(), a);
  if (it != b->inlinks.end()) {
    b->inlinks.erase(it);
  }

  auto it1 = std::find(a->outlinks.begin(), a->outlinks.end(), b);
  if (it1 != a->outlinks.end()) {
    a->outlinks.erase((it1));
  }
}

// Link a -> b.
static void DirectedLink(Node *a, Node *b) {
  // Eagerly remove first, to avoid duplicate link.
  RemoveDirectedLink(a, b);
  a->outlinks.push_back(b);
  b->inlinks.push_back(a);
}

static void LocalInferenceType(Node *a, Node *b, const std::string &arg_name) {
  // instr -> output argument
  if (a->IsStmt() && b->IsArg()) {
    auto &inst = a->AsStmt();
    auto &output = b->AsArg();

    if (!output.type) {
      output.type = inst.picked_kernel().GetOutputDeclType(arg_name);
    }
  }

  // input argument -> instr
  if (a->IsArg() && b->IsStmt()) {
    auto &input = a->AsArg();
    auto &inst = b->AsStmt();
    if (!input.type) {
      input.type = inst.picked_kernel().GetInputDeclType(arg_name);
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
