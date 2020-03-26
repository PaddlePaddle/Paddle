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
#include "paddle/fluid/framework/variable.h"

namespace paddle {
namespace imperative {

class VarBase;
class GradOpNode;

class VariableWrapper {
 public:
  friend class VarBase;

  explicit VariableWrapper(const std::string& name) : name_(name) {}

  const framework::Variable& Var() const { return var_; }

  framework::Variable* MutableVar() { return &var_; }

  // This is used for python api
  void SetOverridedStopGradient(bool stop_gradient) {
    overrided_stop_gradient_ = static_cast<int>(stop_gradient);

    if (auto grad_var = grad_var_.lock()) {
      grad_var->SetOverridedStopGradient(stop_gradient);
    }
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

    if (auto grad_var = grad_var_.lock()) {
      grad_var->InnerSetOverridedStopGradient(stop_gradient);
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

  std::shared_ptr<VariableWrapper> GetGradVar() const {
    return grad_var_.lock();
  }

  const std::weak_ptr<VariableWrapper>& GetWeakGradVar() const {
    return grad_var_;
  }

  std::shared_ptr<GradOpNode> GetGradNode() const { return grad_node_.lock(); }

  bool HasGradNode() const { return !grad_node_.expired(); }

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
  void SetGradVar(const std::shared_ptr<VariableWrapper>& var) {
    auto shared_var = grad_var_.lock();
    if (shared_var != var) {
      PADDLE_ENFORCE_EQ(shared_var, nullptr,
                        platform::errors::PermissionDenied(
                            "Cannot set gradient var wrapper twice"));
      grad_var_ = var;
    }
  }

  void SetGradNode(const std::shared_ptr<GradOpNode>& grad_node) {
    if (!grad_node) {
      grad_node_.reset();
      return;
    }

    auto shared_node = grad_node_.lock();
    if (shared_node != grad_node) {
      PADDLE_ENFORCE_EQ(
          shared_node, nullptr,
          platform::errors::PermissionDenied("Cannot set gradient op twice"));
      grad_node_ = grad_node;
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

  std::weak_ptr<VariableWrapper> grad_var_;
  std::weak_ptr<GradOpNode> grad_node_;
};

}  // namespace imperative
}  // namespace paddle
