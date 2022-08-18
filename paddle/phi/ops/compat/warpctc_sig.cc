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

KernelSignature WarpctcOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("warpctc",
                         {"Logits", "Label", "LogitsLength", "LabelLength"},
                         {"blank", "norm_by_times"},
                         {"Loss", "WarpCTCGrad"});
}

KernelSignature WarpctcGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("warpctc_grad",
                         {"Logits", "LogitsLength", "WarpCTCGrad", "Loss@GRAD"},
                         {"blank", "norm_by_times"},
                         {"Logits@GRAD"});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(warpctc, phi::WarpctcOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(warpctc_grad, phi::WarpctcGradOpArgumentMapping);
