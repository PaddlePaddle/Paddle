/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

KernelSignature FeedForwardFuseOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature("fused_feedforward",
                         {"X",
                          "Dropout1Seed",
                          "Dropout2Seed",
                          "Linear1Weight",
                          "Linear1Bias",
                          "Linear2Weight",
                          "Linear2Bias",
                          "Ln1Scale",
                          "Ln1Bias",
                          "Ln2Scale",
                          "Ln2Bias"},
                         {"pre_layer_norm",
                          "ln1_epsilon",
                          "ln2_epsilon",
                          "act_method",
                          "dropout1_rate",
                          "dropout2_rate",
                          "dropout1_implementation",
                          "dropout2_implementation",
                          "is_test",
                          "dropout1_fix_seed",
                          "dropout2_fix_seed",
                          "dropout1_seed",
                          "dropout2_seed",
                          "add_residual",
                          "ring_id"},
                         {"Out",
                          "Dropout1Mask",
                          "Dropout2Mask",
                          "Ln1Mean",
                          "Ln1Variance",
                          "Ln2Mean",
                          "Ln2Variance",
                          "Linear1Out",
                          "Ln1Out",
                          "Dropout1Out",
                          "Dropout2Out"});
}

KernelSignature FeedForwardGradFuseOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature("fused_feedforward_grad",
                         {"Out@GRAD",      "X",
                          "Linear1Weight", "Linear1Bias",
                          "Linear2Weight", "Dropout1Mask",
                          "Dropout2Mask",  "Linear1Out",
                          "Dropout1Out",   "Dropout2Out",
                          "Ln1Scale",      "Ln1Bias",
                          "Ln1Out",        "Ln1Mean",
                          "Ln1Variance",   "Ln2Scale",
                          "Ln2Bias",       "Ln2Mean",
                          "Ln2Variance",   "Linear2Bias"},
                         {"pre_layer_norm",
                          "ln1_epsilon",
                          "ln2_epsilon",
                          "act_method",
                          "dropout1_rate",
                          "dropout2_rate",
                          "dropout1_implementation",
                          "dropout2_implementation",
                          "is_test",
                          "dropout1_fix_seed",
                          "dropout2_fix_seed",
                          "dropout1_seed",
                          "dropout2_seed",
                          "add_residual",
                          "ring_id"},
                         {"X@GRAD",
                          "Linear1Weight@GRAD",
                          "Linear1Bias@GRAD",
                          "Linear2Weight@GRAD",
                          "Linear2Bias@GRAD",
                          "Ln1Scale@GRAD",
                          "Ln1Bias@GRAD",
                          "Ln2Scale@GRAD",
                          "Ln2Bias@GRAD"});
}
}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(fused_feedforward,
                           phi::FeedForwardFuseOpArgumentMapping);

PD_REGISTER_ARG_MAPPING_FN(fused_feedforward_grad,
                           phi::FeedForwardGradFuseOpArgumentMapping);
