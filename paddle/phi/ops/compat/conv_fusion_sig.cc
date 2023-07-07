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

KernelSignature ConvFusionOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature("conv2d_fusion",
                         {"Input", "Filter", "Bias", "ResidualData"},
                         {
                             "strides",
                             "paddings",
                             "padding_algorithm",
                             "dilations",
                             "groups",
                             "data_format",
                             "activation",
                             "exhaustive_search",
                             "split_channels",
                             "workspace_size_MB",
                         },
                         {"Output", "Outputs"});
}
}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(conv2d_fusion, phi::ConvFusionOpArgumentMapping);
