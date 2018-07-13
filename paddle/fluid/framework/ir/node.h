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

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/variant.h"

namespace paddle {
namespace framework {
namespace ir {

class Node {
 public:
  enum class Type { kNone = -1, kOperation, kVariable };

  Node() : type_(Type::kNone) {}

  explicit Node(Type type) : type_(type) {}

  virtual ~Node() {
    for (auto& attr : attrs_) {
      if (attr_dels_.find(attr.first) != attr_dels_.end()) {
        attr_dels_[attr.first]();
      }
    }
    attr_dels_.clear();
    attrs_.clear();
  }

  Type NodeType() const { return type_; }

  template <typename AttrType>
  void Set(const std::string& name, AttrType attr) {
    attrs_[name] = attr;
  }

  template <typename AttrType>
  void Set(const std::string& name, AttrType* attr,
           std::function<void(void)> attr_del) {
    attrs_[name] = attr;
    attr_dels_[name] = attr_del;
  }

  VarDesc* Var() { return var_desc_; }
  OpDesc* Op() { return op_desc_; }

  explicit Node(VarDesc* var_desc)
      : var_desc_(var_desc), op_desc_(nullptr), type_(Type::kVariable) {}

  explicit Node(OpDesc* op_desc)
      : var_desc_(nullptr), op_desc_(op_desc), type_(Type::kOperation) {}

  std::vector<Node*> inputs;
  std::vector<Node*> outputs;

 protected:
  std::map<std::string, boost::any> attrs_;
  std::map<std::string, std::function<void(void)>> attr_dels_;
  VarDesc* var_desc_;
  OpDesc* op_desc_;
  Type type_;

 private:
  DISABLE_COPY_AND_ASSIGN(Node);
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
