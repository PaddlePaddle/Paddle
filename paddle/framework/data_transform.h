/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include "paddle/framework/op_kernel_type.h"
#include "paddle/framework/selected_rows.h"
#include "paddle/framework/tensor.h"
#include "paddle/framework/variable.h"
#include "paddle/operators/math/math_function.h"
#include "paddle/platform/device_context.h"
#include "paddle/platform/macros.h"
#include "paddle/platform/transform.h"

namespace paddle {
namespace framework {

void DataTransform(const OpKernelType& expected_kernel_type,
                   const OpKernelType& kernel_type_for_var,
                   const Tensor& input_tensor, Tensor* out);

void CopyVariableWithTensor(const Variable& in_var, const Tensor& tensor,
                            Variable& out_var);

}  // namespace framework
}  // namespace paddle
