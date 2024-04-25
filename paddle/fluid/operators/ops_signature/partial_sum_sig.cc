// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

KernelSignature PartialSumOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature(
      "partial_sum", {"X"}, {"start_index", "length"}, {"Out"});
}

KernelSignature PartialSumGradOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature("partial_sum_grad",
                         {"X", "Out@GRAD"},
                         {"start_index", "length"},
                         {"X@GRAD"});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(partial_sum, phi::PartialSumOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(partial_sum_grad,
                           phi::PartialSumGradOpArgumentMapping);
