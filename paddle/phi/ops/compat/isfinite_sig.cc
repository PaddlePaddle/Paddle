// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/compat/op_utils.h"

namespace phi {

KernelSignature IsInfArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("isinf", {"X"}, {}, {"Out"});
}

KernelSignature IsNanArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("isnan", {"X"}, {}, {"Out"});
}

KernelSignature IsFiniteArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("isfinite", {"X"}, {}, {"Out"});
}

}  // namespace phi

PD_REGISTER_BASE_KERNEL_NAME(isinf_v2, isinf);
PD_REGISTER_BASE_KERNEL_NAME(isnan_v2, isnan);
PD_REGISTER_BASE_KERNEL_NAME(isfinite_v2, isfinite);

PD_REGISTER_ARG_MAPPING_FN(isinf, phi::IsInfArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(isnan, phi::IsNanArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(isfinite, phi::IsFiniteArgumentMapping);