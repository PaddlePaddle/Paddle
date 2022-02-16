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

#include "paddle/pten/core/compat/op_utils.h"

namespace pten {

KernelSignature TransposeOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("transpose", {"X"}, {"axis"}, {"Out"});
}

KernelSignature TransposeGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "transpose_grad", {GradVarName("Out")}, {"axis"}, {GradVarName("X")});
}

}  // namespace pten

PT_REGISTER_ARG_MAPPING_FN(transpose, pten::TransposeOpArgumentMapping);
PT_REGISTER_ARG_MAPPING_FN(transpose_grad,
                           pten::TransposeGradOpArgumentMapping);
PT_REGISTER_ARG_MAPPING_FN(transpose2, pten::TransposeOpArgumentMapping);
PT_REGISTER_ARG_MAPPING_FN(transpose2_grad,
                           pten::TransposeGradOpArgumentMapping);

PT_REGISTER_BASE_KERNEL_NAME(transpose2, transpose);
PT_REGISTER_BASE_KERNEL_NAME(transpose2_grad, transpose_grad);
