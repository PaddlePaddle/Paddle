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

KernelSignature DataNormOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature(
      "data_norm",
      {"scale_w", "bias", "X", "BatchSize", "BatchSum", "BatchSquareSum"},
      {"epsilon",
       "slot_dim",
       "summary_decay_rate",
       "enable_scale_and_shift",
       "data_layout",
       "sync_stats"},
      {"Y", "Means", "Scales"});
}

KernelSignature DataNormGradOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature("data_norm_grad",
                         {"scale_w", "bias", "X", "Means", "Scales", "Y@GRAD"},
                         {"epsilon",
                          "slot_dim",
                          "summary_decay_rate",
                          "enable_scale_and_shift",
                          "data_layout",
                          "sync_stats"},
                         {"BatchSize",
                          "BatchSum",
                          "BatchSquareSum",
                          "scale_w@GRAD",
                          "bias@GRAD",
                          "X@GRAD",
                          "BatchSize@GRAD",
                          "BatchSum@GRAD",
                          "BatchSquareSum@GRAD"});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(data_norm, phi::DataNormOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(data_norm_grad, phi::DataNormGradOpArgumentMapping);
