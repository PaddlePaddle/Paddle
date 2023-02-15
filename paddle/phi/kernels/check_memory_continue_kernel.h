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

#pragma once

#include <vector>

#include "paddle/phi/core/dense_tensor.h"

namespace phi {

// WHY add this op?
// This op is used for convert fused_all_reduce_op_handle in Graph to Program.
// i.e, fused_all_reduce_op_handle = check_memory_continue + c_allreduce_sum
// There are two reasons that check_memory_continue is added:
// 1. c_allreduce_sum takes single tensor as input, while
// fused_all_reduce_op_handle takse tensor array as input, so we need a op to
// convert tensor array into a single tensor
// 2. fused_all_reduce_op_handle has a premise that all tensor's addresses are
// continue, so we need a op to do the check.

// see details in fused_all_reduce_op_handle.cc
template <typename T, typename Context>
void CheckMemoryContinueKernel(const Context &dev_ctx,
                               const std::vector<const DenseTensor *> &input,
                               DenseTensor *output,
                               std::vector<DenseTensor *> xout);

}  // namespace phi
