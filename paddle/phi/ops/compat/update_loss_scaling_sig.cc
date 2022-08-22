/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/core/compat/op_utils.h"

namespace phi {

KernelSignature UpdateLossScalingOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  if (ctx.HasInput("StopUpdate")) {
    return KernelSignature(
        "update_loss_scaling",
        {"X", "FoundInfinite", "PrevLossScaling", "InGoodSteps", "InBadSteps"},
        {"incr_every_n_steps",
         "decr_every_n_nan_or_inf",
         "incr_ratio",
         "decr_ratio",
         "StopUpdate"},
        {"Out", "LossScaling", "OutGoodSteps", "OutBadSteps"});
  } else {
    return KernelSignature(
        "update_loss_scaling",
        {"X", "FoundInfinite", "PrevLossScaling", "InGoodSteps", "InBadSteps"},
        {"incr_every_n_steps",
         "decr_every_n_nan_or_inf",
         "incr_ratio",
         "decr_ratio",
         "stop_update"},
        {"Out", "LossScaling", "OutGoodSteps", "OutBadSteps"});
  }
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(update_loss_scaling,
                           phi::UpdateLossScalingOpArgumentMapping);
