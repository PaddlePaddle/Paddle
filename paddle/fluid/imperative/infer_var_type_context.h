// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/framework/var_type_inference.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/imperative/variable_wrapper.h"

namespace paddle {
namespace imperative {

// infer var type context for imperative mode
template <typename VarType>
class RuntimeInferVarTypeContext : public framework::InferVarTypeContext {
 public:
  RuntimeInferVarTypeContext(const NameVarMap<VarType>& inputs,
                             const NameVarMap<VarType>& outputs,
                             const framework::AttributeMap& attrs_map)
      : InferVarTypeContext(nullptr, nullptr),
        inputs_(inputs),
        outputs_(outputs),
        attrs_(attrs_map),
        input_names_(),
        output_names_(),
        var_set_() {
    input_names_.reserve(inputs_.size());
    for (auto& it : inputs_) {
      for (auto& var : it.second) {
        if (var) {
          input_names_[it.first].emplace_back(var->Name());
          var_set_[var->Name()] = var.get();
        }
      }
    }

    output_names_.reserve(outputs_.size());
    for (auto& it : outputs_) {
      for (auto& var : it.second) {
        if (var) {
          output_names_[it.first].emplace_back(var->Name());
          var_set_[var->Name()] = var.get();
        }
      }
    }
  }

  virtual ~RuntimeInferVarTypeContext() {}

  framework::Attribute GetAttr(const std::string& name) const override {
    auto iter = attrs_.find(name);
    PADDLE_ENFORCE_EQ(
        iter != attrs_.end(), true,
        platform::errors::NotFound("Cannot find attribute %s", name));
    return iter->second;
  }

  bool HasVar(const std::string& name) const override {
    return var_set_.count(name) > 0;
  }

  bool HasInput(const std::string& name) const override {
    auto it = inputs_.find(name);
    return (it != inputs_.end() && it->second.size() > 0);
  }

  bool HasOutput(const std::string& name) const override {
    auto it = outputs_.find(name);
    return (it != outputs_.end() && it->second.size() > 0);
  }

  const std::vector<std::string>& Input(
      const std::string& name) const override {
    auto iter = input_names_.find(name);
    PADDLE_ENFORCE_EQ(
        iter != input_names_.end(), true,
        platform::errors::NotFound("Cannot find input var %s", name));
    return iter->second;
  }

  const std::vector<std::string>& Output(
      const std::string& name) const override {
    auto iter = output_names_.find(name);

    PADDLE_ENFORCE_EQ(
        iter != output_names_.end(), true,
        platform::errors::NotFound("Cannot find output var %s", name));
    return iter->second;
  }

  framework::proto::VarType::Type GetType(
      const std::string& name) const override {
    auto iter = var_set_.find(name);

    PADDLE_ENFORCE_EQ(
        iter != var_set_.end(), true,
        platform::errors::NotFound("Cannot find var %s in GetType", name));
    return iter->second->Type();
  }

  void SetType(const std::string& name,
               framework::proto::VarType::Type type) override {
    if (name == "kLookupTablePath") {
      VLOG(2) << "SUPER UGLY FIX, remove this when move imperative mode in C++";
    } else {
      var_set_[name]->SetType(type);
      if ((var_set_[name]->MutableVar()->IsInitialized() == true) &&
          (var_set_[name]->MutableVar()->Type() != type)) {
        var_set_[name]->MutableVar()->Clear();
      }
    }
  }

  framework::proto::VarType::Type GetDataType(
      const std::string& name) const override {
    auto iter = var_set_.find(name);

    PADDLE_ENFORCE_EQ(
        iter != var_set_.end(), true,
        platform::errors::NotFound("Cannot find var %s in GetDataType", name));
    return iter->second->DataType();
  }

  void SetDataType(const std::string& name,
                   framework::proto::VarType::Type type) override {
    var_set_[name]->SetDataType(type);
  }

  std::vector<framework::proto::VarType::Type> GetDataTypes(
      const std::string& name) const override {
    PADDLE_THROW(platform::errors::PermissionDenied(
        "GetDataTypes is not supported in runtime InferVarType"));
  }

  void SetDataTypes(const std::string& name,
                    const std::vector<framework::proto::VarType::Type>&
                        multiple_data_type) override {
    PADDLE_THROW(platform::errors::PermissionDenied(
        "SetDataTypes is not supported in runtime InferVarType"));
  }

  std::vector<int64_t> GetShape(const std::string& name) const override {
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Do not handle Shape in runtime InferVarType"));
  }

  void SetShape(const std::string& name,
                const std::vector<int64_t>& dims) override {
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Do not handle Shape in runtime InferVarType"));
  }

  int32_t GetLoDLevel(const std::string& name) const override {
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Do not handle LoDLevel in runtime InferVarType"));
  }

  void SetLoDLevel(const std::string& name, int32_t lod_level) override {
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Do not handle LoDLevel in runtime InferVarType"));
  }

 private:
  const NameVarMap<VarType>& inputs_;
  const NameVarMap<VarType>& outputs_;
  const framework::AttributeMap& attrs_;
  std::unordered_map<std::string, std::vector<std::string>> input_names_;
  std::unordered_map<std::string, std::vector<std::string>> output_names_;
  std::unordered_map<std::string, VarType*> var_set_;
};

}  // namespace imperative
}  // namespace paddle
