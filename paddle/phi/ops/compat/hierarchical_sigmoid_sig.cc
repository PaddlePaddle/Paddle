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

KernelSignature HierarchicalSigmoidOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("hierarchical_sigmoid",
                         {"X", "W", "Label", "PathTable", "PathCode", "Bias"},
                         {"num_classes",
                          "remote_prefetch",
                          "trainer_id",
                          "height_sections",
                          "epmap",
                          "table_names",
                          "is_sparse"},
                         {"Out", "PreOut", "W_Out"});
}

KernelSignature HierarchicalSigmoidGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  if (ctx.IsDenseTensorOutput(GradVarName("W"))) {
    return KernelSignature(
        "hierarchical_sigmoid_grad",
        {"X",
         "W",
         "Label",
         "PreOut",
         GradVarName("Out"),
         "PathTable",
         "PathCode",
         "Bias"},
        {"num_classes",
         "remote_prefetch",
         "trainer_id",
         "height_sections",
         "epmap",
         "table_names",
         "is_sparse"},
        {GradVarName("X"), GradVarName("W"), GradVarName("Bias")});
  } else if (ctx.IsSelectedRowsOutput(GradVarName("W"))) {
    return KernelSignature(
        "hierarchical_sigmoid_grad_sr",
        {"X",
         "W",
         "Label",
         "PreOut",
         GradVarName("Out"),
         "PathTable",
         "PathCode",
         "Bias"},
        {"num_classes",
         "remote_prefetch",
         "trainer_id",
         "height_sections",
         "epmap",
         "table_names",
         "is_sparse"},
        {GradVarName("X"), GradVarName("W"), GradVarName("Bias")});
  } else {
    return KernelSignature("unregistered", {}, {}, {});
  }
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(hierarchical_sigmoid,
                           phi::HierarchicalSigmoidOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(hierarchical_sigmoid_grad,
                           phi::HierarchicalSigmoidGradOpArgumentMapping);
