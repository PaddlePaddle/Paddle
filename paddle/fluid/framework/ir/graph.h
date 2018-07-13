/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/variant.h"

namespace paddle {
namespace framework {

class Graph {
 public:
  explicit Graph(const ProgramDesc& program) : program_(program) {}

  virtual ~Graph() {
    for (auto& attr : attrs_) {
      attr_dels_[attr.first]();
    }
    attrs_.clear();
    attr_dels_.clear();
  }

  template <typename AttrType>
  AttrType& Get(const std::string& attr_name) const {
    return *boost::any_cast<AttrType*>(attrs_.at(attr_name));
  }

  template <typename AttrType>
  void Set(const std::string& attr_name, AttrType* attr) {
    attrs_[attr_name] = attr;
    attr_dels_[attr_name] = [attr, attr_name]() {
      VLOG(3) << "deleting " << attr_name;
      delete attr;
    };
  }

  template <typename AttrType>
  AttrType* Erase(const std::string& attr_name) {
    AttrType* attr = boost::any_cast<AttrType*>(attrs_[attr_name]);
    attrs_.erase(attr_name);
    attr_dels_.erase(attr_name);
    return attr;
  }

  ir::Node* CreateVarNode(VarDesc* var_desc) {
    nodes.emplace_back(new ir::Node(var_desc));
    return nodes.back().get();
  }

  ir::Node* CreateOpNode(OpDesc* op_desc) {
    nodes.emplace_back(new ir::Node(op_desc));
    return nodes.back().get();
  }

  // TODO(panyx0718): Need to handle CreateOpNode(nullptr).
  ir::Node* CreateVarNode(const std::string& var_name) {
    var_descs_.emplace_back(new VarDesc(var_name));
    nodes.emplace_back(new ir::Node(var_descs_.back().get()));
    return nodes.back().get();
  }

  std::vector<ir::Node*> inputs;
  std::vector<ir::Node*> outputs;
  std::vector<std::unique_ptr<ir::Node>> nodes;
  std::vector<std::unique_ptr<VarDesc>> var_descs_;

 private:
  // NOTE: program_ shouldn't be exposed to user.
  const ProgramDesc& program_;
  std::map<std::string, boost::any> attrs_;
  std::map<std::string, std::function<void(void)>> attr_dels_;
};

std::unique_ptr<Graph> ProgramToGraph(const ProgramDesc& program);

}  // namespace framework
}  // namespace paddle
