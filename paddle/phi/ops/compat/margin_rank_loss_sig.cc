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

KernelSignature MarginRankLossMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("margin_rank_loss", 
                         {"Label", "X1", "X2"}, 
                         {"margin"}, 
                         {"Out", "Activated"});
}

KernelSignature MarginRankLossGradMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("margin_rank_loss_grad", 
                         {"Label", "Activated", "Out@GRAD"}, 
                         {"margin"}, 
                         {"X1@GRAD", "X2@GRAD"});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(margin_rank_loss, phi::MarginRankLossMapping);
PD_REGISTER_ARG_MAPPING_FN(margin_rank_loss_grad, 
                           phi::MarginRankLossGradMapping);
