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
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/variant.h"

namespace paddle {
namespace framework {

class Graph {
 public:
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

  std::vector<Node*> inputs;
  std::vector<Node*> outputs;
  std::vector<std::unique_ptr<Node>> nodes;

 private:
  std::map<std::string, boost::any> attrs_;
  std::map<std::string, std::function<void(void)>> attr_dels_;
};

}  // namespace framework
}  // namespace paddle
