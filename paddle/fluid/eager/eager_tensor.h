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
        auto* framework_tensor =
            var_.GetMutable<paddle::framework::LoDTensor>();
        // Contruct framework::Tensor from egr::EagerTensor
        auto tensor_dense =
            std::dynamic_pointer_cast<pten::DenseTensor>(tensor.impl());
        PADDLE_ENFORCE_EQ((tensor_dense.get() && tensor_dense), true,
                          paddle::platform::errors::Fatal(
                              "Failed to Trans Tensor to EagerVariable since "
                              "we got Tensor with type DenseTensor, and we got "
                              "EagerVariable with another type."));
        *framework_tensor = *tensor_dense;
      } else {
        PADDLE_THROW(paddle::platform::errors::Fatal(
            "Unrecognized egr::EagerVariable type, only "
            "DenseTensor and SelectedRows is supported for now."));
      }
    } else {
      VLOG(6) << "Build Empty EagerTensor with name " << name_;
    }
  }

  /** Part 11: Construct paddle::framework::Variable with pten::Tensor **/
  std::shared_ptr<pten::TensorBase> GetTensorBase() {
    // Construct allocation only once.
    if (var_.IsInitialized()) {
      if (var_.IsType<paddle::framework::LoDTensor>()) {
        return SetImplWithLegacyTensor<pten::DenseTensor>();
      } else if (var_.IsType<paddle::framework::Tensor>()) {
        return SetImplWithLegacyTensor<pten::DenseTensor>();
      } else if (var_.IsType<pten::SelectedRows>()) {
        return SetImplWithSelectedRows();
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
  template <typename LEGACY_TYPE>
  std::shared_ptr<pten::TensorBase> SetImplWithLegacyTensor() {
    const auto& framework_tensor = var_.Get<LEGACY_TYPE>();
    VLOG(8) << "Sync Var to tensor for: " << name();
    return std::make_shared<LEGACY_TYPE>(std::move(framework_tensor));
  }

  std::shared_ptr<pten::TensorBase> SetImplWithSelectedRows() {
    auto* selected_rows = var_.GetMutable<pten::SelectedRows>();
    auto res = std::make_shared<pten::SelectedRows>(selected_rows->rows_,
                                                    selected_rows->height_);
    res->value_.reset(selected_rows->value_.release());
    res->id_to_index_ = std::move(selected_rows->id_to_index_);
    res->rwlock_.reset(selected_rows->rwlock_.release());
    return res;
  }

 private:
  std::string name_{""};
  paddle::framework::Variable var_;
};
}  // namespace egr
