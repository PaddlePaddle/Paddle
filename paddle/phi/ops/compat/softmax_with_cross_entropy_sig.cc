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

KernelSignature SoftmaxWithCrossEntropyOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("cross_entropy_with_softmax",
                         {"Logits", "Label"},
                         {"soft_label",
                          "use_softmax",
                          "numeric_stable_mode",
                          "ignore_index",
                          "axis"},
                         {"Softmax", "Loss"});
}

KernelSignature SoftmaxWithCrossEntropyGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("cross_entropy_with_softmax_grad",
                         {"Label", "Softmax", "Loss@GRAD"},
                         {"soft_label",
                          "use_softmax",
                          "numeric_stable_mode",
                          "ignore_index",
                          "axis"},
                         {"Logits@GRAD"});
}

}  // namespace phi

PD_REGISTER_BASE_KERNEL_NAME(softmax_with_cross_entropy,
                             cross_entropy_with_softmax);
PD_REGISTER_BASE_KERNEL_NAME(softmax_with_cross_entropy_grad,
                             cross_entropy_with_softmax_grad);

PD_REGISTER_ARG_MAPPING_FN(softmax_with_cross_entropy,
                           phi::SoftmaxWithCrossEntropyOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(softmax_with_cross_entropy_grad,
                           phi::SoftmaxWithCrossEntropyGradOpArgumentMapping);
