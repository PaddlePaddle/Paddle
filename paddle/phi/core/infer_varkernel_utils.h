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

#pragma once

#include <iterator>
#include <utility>

#include "paddle/phi/core/attribute.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/tensor_base.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/type_defs.h"
#include "paddle/utils/optional.h"
#include "paddle/utils/small_vector.h"

namespace phi {

class KernelKey;

/**
 * Note: InferVarKernelContext is only designed to MKLDNN kernel when the
 * related memeber function 'GetKernelTypeFor' is special.
 */
class InferVarKernelContext {
 public:
  InferVarKernelContext() = default;
  InferVarKernelContext(const InferVarKernelContext&) = default;
  explicit InferVarKernelContext(const phi::KernelKey* kernel_key,
                                 const AttributeMap* attrs)
      : kernel_key_(kernel_key), attrs_(attrs) {}

  const std::string& GetVarName(void) const { return *var_name_; }

  const DenseTensor& GetTensor(void) const { return *tensor_; }

  const KernelKey& GetKernelKey(void) const { return *kernel_key_; }

  const AttributeMap& GetAttrs(void) const { return *attrs_; }

  void SetVarName(std::string* var_name) { this->var_name_ = var_name; }

  void SetDenseTensor(DenseTensor* tensor) { this->tensor_ = tensor; }

 private:
  const KernelKey* kernel_key_;
  // Use AttributeMap in namespace 'phi' to avoid depending 'fuild'
  const AttributeMap* attrs_;
  std::string* var_name_;
  DenseTensor* tensor_;
};

typedef KernelKey (*InferVarKernelFn)(const InferVarKernelContext*);

}  // namespace phi
