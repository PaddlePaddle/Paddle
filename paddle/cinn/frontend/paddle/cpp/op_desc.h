// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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
#include <absl/types/any.h>
#include <absl/types/variant.h>
#include <glog/logging.h>

#include <algorithm>
#include <string>
#include <vector>

#include "paddle/cinn/frontend/paddle/cpp/desc_api.h"
#include "paddle/cinn/frontend/paddle/framework.pb.h"

namespace cinn {
namespace frontend {
namespace paddle {
namespace cpp {

/*
 * The cpp::OpDesc is the internal representation for Op. All the internal
 * imprementation should use it, not the pb::OpDesc.
 */
class OpDesc : public OpDescAPI {
 public:
  using attrs_t = std::map<std::string, absl::any>;
  using attr_types_t = std::map<std::string, AttrType>;

 protected:
  std::string type_;
  std::map<std::string, std::vector<std::string>> inputs_;
  std::map<std::string, std::vector<std::string>> outputs_;
  std::map<std::string, absl::any> attrs_;
  std::map<std::string, AttrType> attr_types_;

 public:
  OpDesc() = default;

  std::string Type() const override { return type_; }
  void SetType(const std::string& x) override { type_ = x; }

  const std::map<std::string, std::vector<std::string>>& inputs() const {
    return inputs_;
  }
  const std::map<std::string, std::vector<std::string>>& outputs() const {
    return outputs_;
  }
  std::map<std::string, std::vector<std::string>>* mutable_inputs() {
    return &inputs_;
  }
  std::map<std::string, std::vector<std::string>>* mutable_outputs() {
    return &outputs_;
  }

  bool HasInput(const std::string& param) const {
    auto it = inputs_.find(param);
    return it != inputs_.end();
  }

  std::vector<std::string> Input(const std::string& param) const override;

  std::vector<std::string> InputArgumentNames() const override;
  std::vector<std::string> OutputArgumentNames() const override;

  std::vector<std::string> input_vars() const;

  std::vector<std::string> output_vars() const;

  bool HasOutput(const std::string& param) const;

  std::vector<std::string> Output(const std::string& param) const override;

  void SetInput(const std::string& param,
                const std::vector<std::string>& args) override {
    inputs_[param] = args;
  }

  void SetOutput(const std::string& param,
                 const std::vector<std::string>& args) override {
    outputs_[param] = args;
  }

  bool HasAttr(const std::string& name) const override {
    return attrs_.count(name);
  }

  AttrType GetAttrType(const std::string& name) const override {
    auto it = attr_types_.find(name);
    CHECK(it != attr_types_.end());
    return it->second;
  }

  std::vector<std::string> AttrNames() const override {
    std::vector<std::string> res;
    for (const auto& x : attrs_) {
      res.push_back(x.first);
    }
    return res;
  }

  template <typename T>
  void SetAttr(const std::string& name, const T& v);

  template <typename T>
  T GetAttr(const std::string& name) const;

  const std::map<std::string, absl::any>& attrs() const { return attrs_; }
  const std::map<std::string, AttrType>& attr_types() const {
    return attr_types_;
  }
};

}  // namespace cpp
}  // namespace paddle
}  // namespace frontend
}  // namespace cinn
