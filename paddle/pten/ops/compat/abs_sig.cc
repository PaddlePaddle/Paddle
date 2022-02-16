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

KernelSignature AbsOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("abs", {"X"}, {}, {"Out"});
}

KernelSignature AbsGradOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "abs_grad", {"X", GradVarName("Out")}, {}, {GradVarName("X")});
}

KernelSignature AbsDoubleGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("abs_double_grad", {"X", "DDX"}, {}, {"DDOut"});
}

}  // namespace pten

PT_REGISTER_ARG_MAPPING_FN(abs, pten::AbsOpArgumentMapping);
PT_REGISTER_ARG_MAPPING_FN(abs_grad, pten::AbsGradOpArgumentMapping);
PT_REGISTER_ARG_MAPPING_FN(abs_double_grad,
                           pten::AbsDoubleGradOpArgumentMapping);
