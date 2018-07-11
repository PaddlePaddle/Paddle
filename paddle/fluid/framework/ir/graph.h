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

class Graph;

template <typename AttrType>
struct AnyAttr {
 public:
  explicit AnyAttr(AttrType* attr) : attr_(attr) {}

  AttrType& Get() { return *boost::any_cast<AttrType*>(attr_); }

 private:
  friend Graph;

  AttrType* Release() {
    released_ = true;
    return boost::any_cast<AttrType*>(attr_);
  }

  void Delete() {
    if (!released_) {
      delete boost::any_cast<AttrType*>(attr_);
    }
  }

  bool released_ = false;
  boost::any attr_;
};

class Graph {
 public:
  virtual ~Graph() {
    for (auto& attr : attrs) {
      attr_dels[attr.first]();
    }
    attrs.clear();
    attr_dels.clear();
  }

  template <typename AttrType>
  AttrType& Get(const std::string& attr_name) {
    return boost::any_cast<AnyAttr<AttrType>>(attrs[attr_name]).Get();
  }

  template <typename AttrType>
  void Set(const std::string& attr_name, AttrType* attr) {
    AnyAttr<AttrType> any_attr = AnyAttr<AttrType>(attr);
    attrs[attr_name] = any_attr;
    attr_dels[attr_name] = [&any_attr]() { any_attr.Delete(); };
  }

  template <typename AttrType>
  AttrType* Erase(const std::string& attr_name) {
    AnyAttr<AttrType> attr_type =
        boost::any_cast<AnyAttr<AttrType>>(attrs[attr_name]);
    attrs.erase(attr_name);
    attr_dels.erase(attr_name);
    return attr_type.Release();
  }

  std::vector<Node*> inputs;
  std::vector<Node*> outputs;
  std::vector<std::unique_ptr<Node>> nodes;
  std::map<std::string, boost::any> attrs;
  std::map<std::string, std::function<void(void)>> attr_dels;

 private:
};

}  // namespace framework
}  // namespace paddle
