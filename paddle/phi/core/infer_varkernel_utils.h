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

#include "paddle/phi/core/attribute.h"
#include "paddle/phi/core/enforce.h"

namespace phi {

class KernelKey;
class DenseTensor;
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

  const std::string& GetVarName(void) const;

  const DenseTensor& GetTensor(void) const;

  const KernelKey& GetKernelKey(void) const;

  const AttributeMap& GetAttrs(void) const;

  void SetVarName(std::string* var_name);

  void SetDenseTensor(DenseTensor* tensor);

 private:
  const KernelKey* kernel_key_;
  // Use AttributeMap in namespace 'phi' to avoid depending 'fuild'
  const AttributeMap* attrs_;
  std::string* var_name_;
  DenseTensor* tensor_;
};

typedef KernelKey (*InferVarKernelFn)(const InferVarKernelContext*);

}  // namespace phi
