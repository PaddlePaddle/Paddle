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

KernelSignature ExpandOpArgumentMapping(const ArgumentMappingContext& ctx) {
  if (ctx.HasInput("Shape")) {
    return KernelSignature("expand", {"X"}, {"Shape"}, {"Out"});
  } else if (ctx.InputSize("expand_shapes_tensor") > 0) {
    return KernelSignature("expand", {"X"}, {"expand_shapes_tensor"}, {"Out"});
  } else {
    return KernelSignature("expand", {"X"}, {"shape"}, {"Out"});
  }
}

KernelSignature ExpandGradOpArgumentMapping(const ArgumentMappingContext& ctx) {
  if (ctx.HasInput("Shape")) {
    return KernelSignature("expand_grad",
                           {"X", GradVarName("Out")},
                           {"Shape"},
                           {GradVarName("X")});
  } else if (ctx.InputSize("expand_shapes_tensor") > 0) {
    return KernelSignature("expand_grad",
                           {"X", GradVarName("Out")},
                           {"expand_shapes_tensor"},
                           {GradVarName("X")});
  } else {
    return KernelSignature("expand_grad",
                           {"X", GradVarName("Out")},
                           {"shape"},
                           {GradVarName("X")});
  }
}

}  // namespace phi

PD_REGISTER_BASE_KERNEL_NAME(expand_v2, expand);
PD_REGISTER_BASE_KERNEL_NAME(expand_v2_grad, expand_grad);

PD_REGISTER_ARG_MAPPING_FN(expand_v2, phi::ExpandOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(expand_v2_grad, phi::ExpandGradOpArgumentMapping);
