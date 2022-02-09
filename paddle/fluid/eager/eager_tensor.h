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
#include "paddle/fluid/framework/pten_utils.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
// pten deps
#include "paddle/pten/api/include/tensor.h"
#include "paddle/pten/api/lib/api_declare.h"
#include "paddle/pten/api/lib/utils/tensor_utils.h"
#include "paddle/pten/core/compat/convert_utils.h"
/**
 * This class is used by Eager mode for now. It's painful to do this in Eager
 * Mode, the better
 * choice is to use paddle::experimental::Tensor directly. However, we have a
 * punch of nested kernel code, and
 * they use paddle::framework::Variable in inner logic code. So, we have to
 * provide variable in
 * paddle::framework::ExecutionContext to support it. We should remove this as
 * soon as we finish our latest
 * Pten Lib, and use paddle::experimental::Tensor instead.
 *
 * Note: Keep this class as clean as possible.
 * This class should only support method declared in
 * paddle::experimental::Tensor with access method of
 * paddle::framework::Variable no more members are acceptable.
 * **/

namespace egr {
class EagerTensor final {
 public:
  /* Default constructor and name constructor should only be used for contruct
   * output and in fluid*/
  EagerTensor() = default;

  explicit EagerTensor(const std::string& name) : name_(name) {}

  explicit EagerTensor(const paddle::experimental::Tensor& tensor)
      : name_(tensor.name()) {
    if (tensor.defined()) {
      if (tensor.is_dense_tensor()) {
        ConstructVariableFromTensor(tensor);
      } else if (tensor.is_selected_rows()) {
        ConstructVariableFromSelectedRows(tensor);
      } else {
        PADDLE_THROW(paddle::platform::errors::Fatal(
            "Unrecognized egr::EagerTensor type, only "
            "DenseTensor and SelectedRows are supported for now."));
      }
    } else {
      VLOG(6) << "Build Empty EagerTensor with name " << name_;
    }
  }

  /** Part 11: Construct paddle::framework::Variable with pten::Tensor **/
  std::shared_ptr<pten::TensorBase> GetTensorBase() {
    // Construct allocation only once.
    if (var_.IsInitialized()) {
      if (var_.IsType<paddle::framework::LoDTensor>() ||
          var_.IsType<paddle::framework::Tensor>()) {
        return SetImplWithLegacyTensor();
      } else if (var_.IsType<pten::SelectedRows>()) {
        return SetImplWithLegacySelectedRows();
      } else {
        PADDLE_THROW(paddle::platform::errors::Fatal(
            "Unable to fetch underlying tensor "
            "from EagerTensor, only LoDTensor and "
            "Tensor are supported for now"));
      }
    } else {
      PADDLE_THROW(paddle::platform::errors::Fatal(
          "Can not Sync EagerTensor %s whose paddle::framework::Variable is "
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
  std::shared_ptr<pten::TensorBase> SetImplWithLegacyTensor() {
    const auto& framework_tensor = var_.Get<pten::DenseTensor>();
    VLOG(8) << "Sync Var to tensor for: " << name();
    return std::make_shared<pten::DenseTensor>(framework_tensor);
  }

  std::shared_ptr<pten::TensorBase> SetImplWithLegacySelectedRows() {
    auto* framework_tensor = var_.GetMutable<pten::SelectedRows>();
    VLOG(8) << "Sync SelectedRows to tensor for: " << name();
    auto res =
        std::make_shared<pten::SelectedRows>(std::move(*framework_tensor));
    var_.Clear();
    return res;
  }

  void ConstructVariableFromTensor(const paddle::experimental::Tensor& tensor) {
    auto* framework_tensor = var_.GetMutable<pten::DenseTensor>();
    // Contruct framework::Tensor from egr::EagerTensor
    auto tensor_dense =
        std::dynamic_pointer_cast<pten::DenseTensor>(tensor.impl());
    PADDLE_ENFORCE_EQ(
        (tensor_dense.get() && tensor_dense), true,
        paddle::platform::errors::Fatal(
            "Tensor %s does not hold pten::SelectedRows or pten::DenseTensor. "
            "Or it holds empty impl, this should not happend since we should "
            "treat all kinds of tensor as what they are.",
            tensor.name()));
    *framework_tensor = *tensor_dense;
  }

  void ConstructVariableFromSelectedRows(
      const paddle::experimental::Tensor& tensor) {
    auto* framework_tensor = var_.GetMutable<pten::SelectedRows>();
    // Contruct framework::Tensor from egr::EagerTensor
    auto tensor_dense =
        std::dynamic_pointer_cast<pten::SelectedRows>(tensor.impl());
    PADDLE_ENFORCE_EQ(
        (tensor_dense.get() && tensor_dense), true,
        paddle::platform::errors::Fatal(
            "Tensor %s does not hold pten::SelectedRows or pten::DenseTensor. "
            "Or it holds empty impl, this should not happend since we should "
            "treat all kinds of tensor as what they are.",
            tensor.name()));
    *framework_tensor = std::move(*tensor_dense);
  }

 private:
  std::string name_{""};
  paddle::framework::Variable var_;
};
}  // namespace egr
