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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/eager/legacy/tensor_helper.h"
#include "paddle/fluid/eager/legacy/type_def.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/framework/var_type_inference.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/pten/api/all.h"

namespace egr {
namespace legacy {

// infer var type context for imperative mode
class TensorRuntimeInferVarTypeContext
    : public paddle::framework::InferVarTypeContext {
 public:
  TensorRuntimeInferVarTypeContext(
      const NameTensorMap& inputs, const NameTensorMap& outputs,
      const paddle::framework::AttributeMap& attrs_map,
      const paddle::framework::AttributeMap& default_attrs_map)
      : InferVarTypeContext(nullptr, nullptr),
        inputs_(inputs),
        outputs_(outputs),
        attrs_(attrs_map),
        default_attrs_(default_attrs_map) {}

  virtual ~TensorRuntimeInferVarTypeContext() {}

  paddle::framework::Attribute GetAttr(const std::string& name) const override {
    auto it = attrs_.find(name);

    if (it == attrs_.end()) {
      it = default_attrs_.find(name);
      if (it == default_attrs_.end()) {
        PADDLE_THROW(paddle::platform::errors::NotFound(
            "Can not find [%s] in attributes.", name));
      }
    }

    return it->second;
  }

  bool HasInput(const std::string& name) const override {
    auto it = inputs_.find(name);
    return (it != inputs_.end() && it->second.size() > 0);
  }

  bool HasOutput(const std::string& name) const override {
    auto it = outputs_.find(name);
    return (it != outputs_.end() && it->second.size() > 0);
  }

  size_t InputSize(const std::string& name) const {
    return inputs_.at(name).size();
  }

  const std::string& InputVarName(const std::string& name,
                                  const int index = 0) const {
    // TODO(jiabin): Support this usage inputs_.at(name)[index]->Name()
    auto it = inputs_.find(name);
    PADDLE_ENFORCE_NE(it, inputs_.end(),
                      paddle::platform::errors::PreconditionNotMet(
                          "Can not find [%s] in Input", name));
    return inputs_.at(name)[index]->name();
  }

  bool InputTypeAnyOf(
      const std::string& name,
      paddle::framework::proto::VarType::Type type) const override {
    auto& inputs = inputs_.at(name);
    return std::any_of(
        inputs.begin(), inputs.end(),
        [&type](const std::shared_ptr<egr::EagerTensor>& var) {
          return paddle::framework::ToVarType(var->Var().Type()) == type;
        });
  }

  bool InputTypeAllOf(
      const std::string& name,
      paddle::framework::proto::VarType::Type type) const override {
    auto& inputs = inputs_.at(name);
    return std::all_of(
        inputs.begin(), inputs.end(),
        [&type](const std::shared_ptr<egr::EagerTensor>& var) {
          return paddle::framework::ToVarType(var->Var().Type()) == type;
        });
  }

  void SyncTypeAndDataType(const std::string& input_name,
                           const std::string& output_name,
                           int index = 0) override {
    auto in_tensor = inputs_.at(input_name)[index];
    auto out_tensor = outputs_.at(output_name)[index];
    if (in_tensor != out_tensor) {
      this->SetTensorType(
          out_tensor, paddle::framework::ToVarType(in_tensor->Var().Type()));
    }
  }

  void SetOutputType(const std::string& name,
                     paddle::framework::proto::VarType::Type type,
                     int index = 0) override {
    if (index == paddle::framework::ALL_ELEMENTS) {
      for (auto& item : outputs_.at(name)) {
        this->SetTensorType(item, type);
      }
    } else {
      auto& var = outputs_.at(name)[index];
      this->SetTensorType(var, type);
    }
  }

  void SetTensorType(std::shared_ptr<egr::EagerTensor> out,
                     paddle::framework::proto::VarType::Type type) {
    switch (type) {
      case paddle::framework::proto::VarType::LOD_TENSOR: {
        out->MutableVar()->GetMutable<paddle::framework::LoDTensor>();
        break;
      }
      default: {
        PADDLE_THROW(paddle::platform::errors::NotFound(
            "Cannot found var type: %s while running runtime InferVarType",
            paddle::framework::ToTypeName(type)));
      }
    }
  }

  paddle::framework::proto::VarType::Type GetInputType(
      const std::string& name, const int& index = 0) const override {
    return paddle::framework::ToVarType(inputs_.at(name)[index]->Var().Type());
  }

  paddle::framework::proto::VarType::Type GetOutputType(
      const std::string& name, const int& index = 0) const override {
    // TODO(jiabin): Support SelectedRows when we have it.
    return paddle::framework::proto::VarType::LOD_TENSOR;
  }

  paddle::framework::proto::VarType::Type GetInputDataType(
      const std::string& name, const int& index = 0) const override {
    return inputs_.at(name)[index]
        ->Var()
        .Get<paddle::framework::LoDTensor>()
        .type();
  }

  void SetOutputDataType(const std::string& name,
                         paddle::framework::proto::VarType::Type type,
                         int index = 0) override {
    // TODO(jiabin): It seems doesn't make sense to set data_type in EagerMode.
  }

  bool IsDygraph() const override { return true; }

 protected:
  bool HasVar(const std::string& name) const override {
    PADDLE_THROW(paddle::platform::errors::PermissionDenied(
        "HasVar is not supported in runtime InferVarType"));
  }

  const std::vector<std::string>& InputVars(
      const std::string& name) const override {
    PADDLE_THROW(paddle::platform::errors::PermissionDenied(
        "InputVars is not supported in runtime InferVarType"));
  }

  const std::vector<std::string>& OutputVars(
      const std::string& name) const override {
    PADDLE_THROW(paddle::platform::errors::PermissionDenied(
        "OutputVars is not supported in runtime InferVarType"));
  }

  paddle::framework::proto::VarType::Type GetVarType(
      const std::string& name) const override {
    PADDLE_THROW(paddle::platform::errors::PermissionDenied(
        "Do not manipulate var in runtime InferVarType"));
  }

  void SetVarType(const std::string& name,
                  paddle::framework::proto::VarType::Type type) override {
    PADDLE_THROW(paddle::platform::errors::PermissionDenied(
        "Do not manipulate var in runtime InferVarType"));
  }

  paddle::framework::proto::VarType::Type GetVarDataType(
      const std::string& name) const override {
    PADDLE_THROW(paddle::platform::errors::PermissionDenied(
        "Do not manipulate var in runtime InferVarType"));
  }

  void SetVarDataType(const std::string& name,
                      paddle::framework::proto::VarType::Type type) override {
    PADDLE_THROW(paddle::platform::errors::PermissionDenied(
        "Do not manipulate var in runtime InferVarType"));
  }

  std::vector<paddle::framework::proto::VarType::Type> GetVarDataTypes(
      const std::string& name) const override {
    PADDLE_THROW(paddle::platform::errors::PermissionDenied(
        "GetVarDataTypes is not supported in runtime InferVarType"));
  }

  void SetVarDataTypes(
      const std::string& name,
      const std::vector<paddle::framework::proto::VarType::Type>&
          multiple_data_type) override {
    PADDLE_THROW(paddle::platform::errors::PermissionDenied(
        "SetVarDataTypes is not supported in runtime InferVarType"));
  }

  std::vector<int64_t> GetVarShape(const std::string& name) const override {
    PADDLE_THROW(paddle::platform::errors::PermissionDenied(
        "Do not handle Shape in runtime InferVarType"));
  }

  void SetVarShape(const std::string& name,
                   const std::vector<int64_t>& dims) override {
    PADDLE_THROW(paddle::platform::errors::PermissionDenied(
        "Do not handle Shape in runtime InferVarType"));
  }

  int32_t GetVarLoDLevel(const std::string& name) const override {
    PADDLE_THROW(paddle::platform::errors::PermissionDenied(
        "Do not handle LoDLevel in runtime InferVarType"));
  }

  void SetVarLoDLevel(const std::string& name, int32_t lod_level) override {
    PADDLE_THROW(paddle::platform::errors::PermissionDenied(
        "Do not handle LoDLevel in runtime InferVarType"));
  }

 private:
  const NameTensorMap& inputs_;
  const NameTensorMap& outputs_;
  const paddle::framework::AttributeMap& attrs_;
  const paddle::framework::AttributeMap& default_attrs_;
};

}  // namespace legacy
}  // namespace egr
