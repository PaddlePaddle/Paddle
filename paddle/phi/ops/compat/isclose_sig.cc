
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

KernelSignature IscloseOpArgumentMapping(const ArgumentMappingContext& ctx) {
  if (ctx.HasInput("Rtol")) {
    if (ctx.HasInput("Atol")) {
      return KernelSignature("isclose",
                             {"Input", "Other"},
                             {"Rtol", "Atol", "equal_nan"},
                             {"Out"});

    } else {
      return KernelSignature("isclose",
                             {"Input", "Other"},
                             {"Rtol", "atol", "equal_nan"},
                             {"Out"});
    }
  } else {
    if (ctx.HasInput("Atol")) {
      return KernelSignature("isclose",
                             {"Input", "Other"},
                             {"rtol", "Atol", "equal_nan"},
                             {"Out"});
    } else {
      return KernelSignature("isclose",
                             {"Input", "Other"},
                             {"rtol", "atol", "equal_nan"},
                             {"Out"});
    }
  }
}

}  // namespace phi
PD_REGISTER_ARG_MAPPING_FN(isclose, phi::IscloseOpArgumentMapping);
