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

KernelSignature NllLossOpArgumentMapping(const ArgumentMappingContext& ctx) {
  // TODO(xiongkun): can't remove the forward mapping, because the Weight is
  // optional
  return KernelSignature("nll_loss",
                         {"X", "Label", "Weight"},
                         {"ignore_index", "reduction"},
                         {"Out", "Total_weight"});
}

KernelSignature NllLossGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "nll_loss_grad",
      {"X", "Label", "Total_weight", "Weight", GradVarName("Out")},
      {"ignore_index", "reduction"},
      {GradVarName("X")});
}

}  // namespace phi
PD_REGISTER_ARG_MAPPING_FN(nll_loss_grad, phi::NllLossGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(nll_loss, phi::NllLossOpArgumentMapping);
