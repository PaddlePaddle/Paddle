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
KernelSignature UniformRandomInplaceOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature(
<<<<<<< HEAD
      "uniform_inplace",
=======
      "uniform_random_inplace",
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
      {"X"},
      {"min", "max", "seed", "diag_num", "diag_step", "diag_val"},
      {"Out"});
}

KernelSignature UniformRandomInplaceGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature(
<<<<<<< HEAD
      "uniform_inplace_grad",
=======
      "uniform_random_inplace_grad",
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
      {"Out@GRAD"},
      {"min", "max", "seed", "diag_num", "diag_step", "diag_val"},
      {"X@GRAD"});
}

}  // namespace phi

<<<<<<< HEAD
PD_REGISTER_BASE_KERNEL_NAME(uniform_random_inplace, uniform_inplace);

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
PD_REGISTER_ARG_MAPPING_FN(uniform_random_inplace,
                           phi::UniformRandomInplaceOpArgumentMapping);

PD_REGISTER_ARG_MAPPING_FN(uniform_random_inplace_grad,
                           phi::UniformRandomInplaceGradOpArgumentMapping);
