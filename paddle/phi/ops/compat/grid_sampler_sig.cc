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

KernelSignature GridSamplerOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("grid_sample",
                         {"X", "Grid"},
                         {"mode", "padding_mode", "align_corners"},
                         {"Output"});
}

KernelSignature GridSamplerGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("grid_sample_grad",
                         {"X", "Grid", GradVarName("Output")},
                         {"mode", "padding_mode", "align_corners"},
                         {GradVarName("X"), GradVarName("Grid")});
}

}  // namespace phi

// use Python API name as kernel name
PD_REGISTER_BASE_KERNEL_NAME(grid_sampler, grid_sample);
PD_REGISTER_BASE_KERNEL_NAME(grid_sampler_grad, grid_sample_grad);

PD_REGISTER_ARG_MAPPING_FN(grid_sampler, phi::GridSamplerOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(grid_sampler_grad,
                           phi::GridSamplerGradOpArgumentMapping);
