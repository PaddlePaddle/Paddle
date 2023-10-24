// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/compat/get_kerneltype_forvar_utils.h"

#include "paddle/phi/core/enforce.h"
namespace phi {

const std::string& GetKernelTypeForVarContext::GetVarName() const {
  PADDLE_ENFORCE_NE(
      var_name_,
      nullptr,
      errors::InvalidArgument(
          "Variable name is null. The context hasn't been initialized. "));
  return *var_name_;
}

const DenseTensor& GetKernelTypeForVarContext::GetTensor() const {
  PADDLE_ENFORCE_NE(
      tensor_,
      nullptr,
      errors::InvalidArgument(
          "Tensor is null. The context hasn't been initialized. "));
  return *tensor_;
}

const KernelKey& GetKernelTypeForVarContext::GetKernelKey() const {
  PADDLE_ENFORCE_NE(
      kernel_key_,
      nullptr,
      errors::InvalidArgument(
          "Kernel key is null. The context hasn't been initialized. "));
  return *kernel_key_;
}

const AttributeMap& GetKernelTypeForVarContext::GetAttrs() const {
  return *attrs_;
}

void GetKernelTypeForVarContext::SetVarName(std::string* var_name) {
  this->var_name_ = var_name;
}

void GetKernelTypeForVarContext::SetDenseTensor(DenseTensor* tensor) {
  this->tensor_ = tensor;
}

}  // namespace phi
