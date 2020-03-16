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
#include "paddle/fluid/framework/variable.h"

namespace paddle {
namespace imperative {

class VariableWrapper {
 public:
  explicit VariableWrapper(const std::string& name) : name_(name) {}

  const framework::Variable& Var() const { return var_; }

  framework::Variable* MutableVar() { return &var_; }

  // This is used for python api
  void SetOverridedStopGradient(bool stop_gradient) {
    overrided_stop_gradient_ = static_cast<int>(stop_gradient);
  }

  // This is used for python api
  bool OverridedStopGradient() const { return overrided_stop_gradient_ != 0; }

  // This is used inside C++
  int InnerOverridedStopGradient() const { return overrided_stop_gradient_; }

  // This is used inside C++
  void InnerSetOverridedStopGradient(bool stop_gradient) {
    if (overrided_stop_gradient_ == -1) {
      overrided_stop_gradient_ = static_cast<int>(stop_gradient);
    } else {
      VLOG(6) << "Ignore Stop gradient conversion for Var: " << Name()
              << "Set value is: " << overrided_stop_gradient_;
    }
  }

  void SetPersistable(bool persistable) { persistable_ = persistable; }

  bool Persistable() const { return persistable_; }

  const std::string& Name() const { return name_; }

  void SetName(const std::string& name) { name_ = name; }

  void SetType(framework::proto::VarType::Type type) { type_ = type; }

  framework::proto::VarType::Type Type() const { return type_; }

  void SetDataType(framework::proto::VarType::Type data_type) {
    data_type_ = data_type;
  }

  framework::proto::VarType::Type DataType() const {
    const framework::Tensor* tensor = nullptr;
    if (var_.IsInitialized()) {
      if (type_ == framework::proto::VarType::LOD_TENSOR) {
        tensor = &(var_.Get<framework::LoDTensor>());
      } else if (type_ == framework::proto::VarType::SELECTED_ROWS) {
        tensor = &(var_.Get<framework::SelectedRows>().value());
      } else {
        VLOG(6) << "Variable " << name_ << " is not initialized";
        return data_type_;
      }
    }
    if (tensor && tensor->IsInitialized()) {
      return tensor->type();
    } else {
      VLOG(6) << "The tensor of variable " << name_ << " is not initialized";
      return data_type_;
    }
  }

 private:
  framework::Variable var_;
  std::string name_;

  // add this property for users may set stop_gradient themselves and this
  // should override the frameworks setting (-1) unset, (1) true, (0) false
  int overrided_stop_gradient_{-1};
  bool persistable_{false};

  framework::proto::VarType::Type type_{framework::proto::VarType::LOD_TENSOR};
  framework::proto::VarType::Type data_type_{framework::proto::VarType::FP32};
};

}  // namespace imperative
}  // namespace paddle
