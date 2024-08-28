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
KernelSignature CSoftmaxWithCrossEntropyOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("c_softmax_with_cross_entropy",
                         {"Logits", "Label"},
                         {"ignore_index", "ring_id", "rank", "nranks"},
                         {"Softmax", "Loss"});
}

KernelSignature CSoftmaxWithCrossEntropyGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("c_softmax_with_cross_grad",
                         {"SoftMax", "Label", "Loss@GRAD"},
                         {"ignore_index", "ring_id", "rank", "nranks"},
                         {"Logits@GRAD"});
}
}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(c_softmax_with_cross,
                           phi::CSoftmaxWithCrossEntropyOpArgumentMapping);

PD_REGISTER_ARG_MAPPING_FN(c_softmax_with_cross_grad,
                           phi::CSoftmaxWithCrossEntropyGradOpArgumentMapping);
