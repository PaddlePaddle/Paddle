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

KernelSignature WarprnntOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("warprnnt",
                         {"Logits", "Label", "LogitsLength", "LabelLength"},
                         {"blank", "fastemit_lambda", "num_threads"},
                         {"Loss", "WarpRNNTGrad"});
}

KernelSignature WarprnntGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "warprnnt_grad",
      {"Logits", "LogitsLength", "WarpRNNTGrad", "Loss@GRAD"},
      {"blank", "fastemit_lambda", "num_threads"},
      {"Logits@GRAD"});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(warprnnt, phi::WarprnntOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(warprnnt_grad, phi::WarprnntGradOpArgumentMapping);
