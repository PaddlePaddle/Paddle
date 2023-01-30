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

KernelSignature Conv3dOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("conv3d",
                         {"Input", "Filter"},
<<<<<<< HEAD
                         {
                             "strides",
                             "paddings",
                             "padding_algorithm",
                             "groups",
                             "dilations",
                             "data_format",
                         },
=======
                         {"strides",
                          "paddings",
                          "padding_algorithm",
                          "groups",
                          "dilations",
                          "data_format",
                          "use_addto",
                          "workspace_size_MB",
                          "exhaustive_search"},
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                         {"Output"});
}

KernelSignature Conv3dGradOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("conv3d_grad",
                         {"Input", "Filter", "Output@GRAD"},
                         {"strides",
                          "paddings",
                          "padding_algorithm",
                          "groups",
                          "dilations",
<<<<<<< HEAD
                          "data_format"},
=======
                          "data_format",
                          "use_addto",
                          "workspace_size_MB",
                          "exhaustive_search"},
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                         {"Input@GRAD", "Filter@GRAD"});
}

KernelSignature Conv3dDoubleGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
<<<<<<< HEAD
  return KernelSignature("conv3d_double_grad",
=======
  return KernelSignature("conv3d_grad_grad",
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                         {"Input", "Filter", "DOutput", "DDInput", "DDFilter"},
                         {"strides",
                          "paddings",
                          "padding_algorithm",
                          "groups",
                          "dilations",
<<<<<<< HEAD
                          "data_format"},
=======
                          "data_format",
                          "use_addto",
                          "workspace_size_MB",
                          "exhaustive_search"},
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                         {"DInput", "DFilter", "DDOutput"});
}

}  // namespace phi

<<<<<<< HEAD
PD_REGISTER_BASE_KERNEL_NAME(conv3d_grad_grad, conv3d_double_grad);

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
PD_REGISTER_ARG_MAPPING_FN(conv3d, phi::Conv3dOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(conv3d_grad, phi::Conv3dGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(conv3d_grad_grad,
                           phi::Conv3dDoubleGradOpArgumentMapping);
