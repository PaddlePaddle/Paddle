// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
// framework deps
#include "paddle/fluid/framework/phi_utils.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
// Phi deps
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/api/lib/utils/tensor_utils.h"
#include "paddle/phi/core/compat/convert_utils.h"
/**
 * This class is used by Eager mode for now. It's painful to do this in Eager
 * Mode, the better
 * choice is to use paddle::experimental::Tensor directly. However, we have a
 * punch of nested kernel code, and
 * they use paddle::framework::Variable in inner logic code. So, we have to
 * provide variable in
 * paddle::framework::ExecutionContext to support it. We should remove this as
 * soon as we finish our latest
 * Phi Lib, and use paddle::experimental::Tensor instead.
 *
 * Note: Keep this class as clean as possible.
 * This class should only support method declared in
 * paddle::experimental::Tensor with access method of
 * paddle::framework::Variable no more members are acceptable.
 * **/

namespace egr {
class EagerVariable final {
 public:
  /* Default constructor and name constructor should only be used for contruct
   * output and in fluid*/
  EagerVariable() = default;

  explicit EagerVariable(const std::string& name) : name_(name) {}

  explicit EagerVariable(const paddle::experimental::Tensor& tensor)
      : name_(tensor.name()) {
    if (tensor.defined()) {
      if (tensor.is_dense_tensor()) {
        ConstructVariableFromTensor<phi::DenseTensor>(tensor);
      } else if (tensor.is_selected_rows()) {
        ConstructVariableFromTensor<phi::SelectedRows>(tensor);
      } else {
        PADDLE_THROW(paddle::platform::errors::Fatal(
            "Unrecognized egr::EagerVariable type, only "
            "DenseTensor and SelectedRows are supported for now."));
      }
    } else {
      VLOG(6) << "Build Empty EagerVariable with name " << name_;
    }
  }

  /** Part 11: Construct paddle::framework::Variable with phi::Tensor **/
  std::shared_ptr<phi::TensorBase> GetTensorBase() {
    // Construct allocation only once.
    if (var_.IsInitialized()) {
      if (var_.IsType<paddle::framework::LoDTensor>() ||
          var_.IsType<paddle::framework::Tensor>()) {
        return SetImplWithLegacyTensor<phi::DenseTensor>();
      } else if (var_.IsType<phi::SelectedRows>()) {
        return SetImplWithLegacyTensor<phi::SelectedRows>();
      } else {
        PADDLE_THROW(paddle::platform::errors::Fatal(
            "Unable to fetch underlying tensor "
            "from EagerVariable, only LoDTensor and "
            "Tensor are supported for now"));
      }
    } else {
      PADDLE_THROW(paddle::platform::errors::Fatal(
          "Can not Sync EagerVariable %s whose paddle::framework::Variable is "
          "not initialized!",
          name()));
    }
  }
  const paddle::framework::Variable& Var() const { return var_; }

  paddle::framework::Variable* MutableVar() { return &var_; }

  void ResetVar(const paddle::framework::Variable& src) { var_ = src; }

  const std::string& name() const { return name_; }

  void set_name(const std::string& name) { name_ = name; }

 private:
  template <typename VarType>
  std::shared_ptr<phi::TensorBase> SetImplWithLegacyTensor() {
    const auto& framework_tensor = var_.Get<VarType>();
    VLOG(8) << "Sync Var to tensor for: " << name();
    return std::make_shared<VarType>(framework_tensor);
  }

  template <typename VarType>
  void ConstructVariableFromTensor(const paddle::experimental::Tensor& tensor) {
    auto* framework_tensor = var_.GetMutable<VarType>();
    // Contruct framework::Tensor from egr::EagerVariable
    auto tensor_dense = std::dynamic_pointer_cast<VarType>(tensor.impl());
    PADDLE_ENFORCE_EQ(
        (tensor_dense.get() && tensor_dense), true,
        paddle::platform::errors::Fatal(
            "Tensor %s does not hold phi::SelectedRows or phi::DenseTensor. "
            "Or it holds empty impl, this should not happend since we should "
            "treat all kinds of tensor as what they are.",
            tensor.name()));
    *framework_tensor = *tensor_dense;
  }

 private:
  std::string name_{""};
  paddle::framework::Variable var_;
};
}  // namespace egr
