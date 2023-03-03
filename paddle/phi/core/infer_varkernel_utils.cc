//   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/infer_varkernel_utils.h"

namespace phi {

const std::string& InferVarKernelContext::GetVarName(void) const {
  return *var_name_;
}

const DenseTensor& InferVarKernelContext::GetTensor(void) const {
  return *tensor_;
}

const KernelKey& InferVarKernelContext::GetKernelKey(void) const {
  return *kernel_key_;
}

const AttributeMap& InferVarKernelContext::GetAttrs(void) const {
  return *attrs_;
}

void InferVarKernelContext::SetVarName(std::string* var_name) {
  this->var_name_ = var_name;
}

void InferVarKernelContext::SetDenseTensor(DenseTensor* tensor) {
  this->tensor_ = tensor;
}

}  // namespace phi
