/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <functional>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/framework/selected_rows_utils.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/transform.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace framework {

class OpKernelType;
class Variable;

void TransformData(const OpKernelType &expected_kernel_type,
                   const OpKernelType &kernel_type_for_var,
<<<<<<< HEAD
                   const Tensor &input_tensor,
                   Tensor *out);
=======
                   const phi::DenseTensor &input_tensor,
                   phi::DenseTensor *out);
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

/**
 * Set OutVar from InVar, except the tensor is shared with `tensor`
 */
void SetTensorToVariable(const Variable &in_var,
<<<<<<< HEAD
                         const Tensor &tensor,
=======
                         const phi::DenseTensor &tensor,
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
                         Variable *out_var);
}  // namespace framework
}  // namespace paddle
