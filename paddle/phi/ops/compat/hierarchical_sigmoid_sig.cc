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
<<<<<<< HEAD
  return KernelSignature("hsigmoid_loss",
                         {"X", "Label", "W", "Bias", "PathTable", "PathCode"},
                         {"num_classes", "remote_prefetch", "is_sparse"},
=======
  return KernelSignature("hierarchical_sigmoid",
                         {"X", "W", "Label", "PathTable", "PathCode", "Bias"},
                         {"num_classes",
                          "remote_prefetch",
                          "trainer_id",
                          "height_sections",
                          "epmap",
                          "table_names",
                          "is_sparse"},
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                         {"Out", "PreOut", "W_Out"});
}

KernelSignature HierarchicalSigmoidGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  if (ctx.IsDenseTensorOutput("W@GRAD")) {
<<<<<<< HEAD
    return KernelSignature("hsigmoid_loss_grad",
=======
    return KernelSignature("hierarchical_sigmoid_grad",
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                           {"X",
                            "W",
                            "Label",
                            "PathTable",
                            "PathCode",
                            "Bias",
                            "PreOut",
                            "Out@GRAD"},
<<<<<<< HEAD
                           {"num_classes", "remote_prefetch", "is_sparse"},
                           {"X@GRAD", "W@GRAD", "Bias@GRAD"});
  } else if (ctx.IsSelectedRowsOutput("W@GRAD")) {
    return KernelSignature("hsigmoid_loss_grad_sr",
=======
                           {"num_classes",
                            "remote_prefetch",
                            "trainer_id",
                            "height_sections",
                            "epmap",
                            "table_names",
                            "is_sparse"},
                           {"X@GRAD", "W@GRAD", "Bias@GRAD"});
  } else if (ctx.IsSelectedRowsOutput("W@GRAD")) {
    return KernelSignature("hierarchical_sigmoid_grad_sr",
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                           {"X",
                            "W",
                            "Label",
                            "PathTable",
                            "PathCode",
                            "Bias",
                            "PreOut",
                            "Out@GRAD"},
<<<<<<< HEAD
                           {"num_classes", "remote_prefetch", "is_sparse"},
=======
                           {"num_classes",
                            "remote_prefetch",
                            "trainer_id",
                            "height_sections",
                            "epmap",
                            "table_names",
                            "is_sparse"},
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                           {"X@GRAD", "W@GRAD", "Bias@GRAD"});
  } else {
    return KernelSignature("unregistered", {}, {}, {});
  }
}

}  // namespace phi

<<<<<<< HEAD
PD_REGISTER_BASE_KERNEL_NAME(hierarchical_sigmoid, hsigmoid_loss);
PD_REGISTER_BASE_KERNEL_NAME(hierarchical_sigmoid_grad, hsigmoid_loss_grad);

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
PD_REGISTER_ARG_MAPPING_FN(hierarchical_sigmoid,
                           phi::HierarchicalSigmoidOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(hierarchical_sigmoid_grad,
                           phi::HierarchicalSigmoidGradOpArgumentMapping);
