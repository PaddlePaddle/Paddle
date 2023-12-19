// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/phi/core/attribute.h"

namespace phi {

class KernelKey;
class DenseTensor;
/**
 * Note: GetKernelTypeForVarContext is currently designed for oneDNN kernel when
 * the related member function 'GetKernelTypeForVar' is special. It is
 * possible to leverage to other vendor libraries in the future.
 */
class GetKernelTypeForVarContext {
 public:
  GetKernelTypeForVarContext() = default;
  GetKernelTypeForVarContext(const GetKernelTypeForVarContext&) = default;
  explicit GetKernelTypeForVarContext(const phi::KernelKey* kernel_key,
                                      const AttributeMap* attrs)
      : kernel_key_(kernel_key), attrs_(attrs) {}

  const std::string& GetVarName(void) const;

  const DenseTensor& GetTensor(void) const;

  const KernelKey& GetKernelKey(void) const;

  const AttributeMap& GetAttrs(void) const;

  void SetVarName(std::string* var_name);

  void SetDenseTensor(DenseTensor* tensor);

 private:
  const KernelKey* kernel_key_;  // not owned
  // Use AttributeMap in namespace 'phi' to avoid depending 'fluid'
  const AttributeMap* attrs_;  // not owned
  std::string* var_name_;      // not owned
  DenseTensor* tensor_;        // not owned
};

typedef KernelKey (*GetKernelTypeForVarFn)(const GetKernelTypeForVarContext*);

}  // namespace phi
