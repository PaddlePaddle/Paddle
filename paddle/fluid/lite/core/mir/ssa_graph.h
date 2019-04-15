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
    for (auto &op : program.ops) {
      node_storage_.emplace_back();
      // TODO(Superjomn) remove one valid_places here.
      op->SetValidPlaces(valid_places);
      auto &new_node = node_storage_.back();
      auto &new_kernel = node_storage_.back().AsInstruct();
      new_kernel.valid_kernels = op->CreateKernels(valid_places);

      CHECK(new_node.inlinks.empty()) << "duplicate Build found";
      CHECK(new_node.outlinks.empty()) << "duplicate Build found";
      // collect inputs and outputs
      for (const std::string &name : op->input_names()) {
        new_node.inlinks.push_back(arguments_.at(name));
      }
      for (const std::string &name : op->output_names()) {
        new_node.outlinks.push_back(arguments_.at(name));
      }
    }
  }

  std::vector<mir::Node *> TopoloticalOrder() const;

 private:
  std::list<mir::Node> node_storage_;
  std::map<std::string, mir::Node *> arguments_;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
