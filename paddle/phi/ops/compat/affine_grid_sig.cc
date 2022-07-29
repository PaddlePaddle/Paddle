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

KernelSignature AffineGridOpArgumentMapping(const ArgumentMappingContext& ctx) {
  if (ctx.HasInput("OutputShape")) {
    return KernelSignature(
        "affine_grid", {"Theta"}, {"OutputShape", "align_corners"}, {"Output"});
  } else {
    return KernelSignature("affine_grid",
                           {"Theta"},
                           {"output_shape", "align_corners"},
                           {"Output"});
  }
}

KernelSignature AffineGridGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  if (ctx.HasInput("OutputShape")) {
    return KernelSignature("affine_grid_grad",
                           {"Output@GRAD"},
                           {"OutputShape", "align_corners"},
                           {"Theta@GRAD"});
  } else {
    return KernelSignature("affine_grid_grad",
                           {"Output@GRAD"},
                           {"output_shape", "align_corners"},
                           {"Theta@GRAD"});
  }
}
}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(affine_grid, phi::AffineGridOpArgumentMapping);

PD_REGISTER_ARG_MAPPING_FN(affine_grid_grad,
                           phi::AffineGridGradOpArgumentMapping);
