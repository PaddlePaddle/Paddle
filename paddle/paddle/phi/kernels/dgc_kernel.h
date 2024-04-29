// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/dense_tensor.h"

namespace phi {

template <typename T, typename Context>
void DGCKernel(const Context& dev_ctx,
               const DenseTensor& u,
               const DenseTensor& v,
               const DenseTensor& grad,
               const DenseTensor& param,
               const DenseTensor& current_step_tensor,
               const DenseTensor& nranks_tensor,
               float m,
               bool use_nesterov,
               const std::vector<float>& sparsity,
               float rampup_begin_step,
               float rampup_step,
               float regular_coeff,
               int regular_type,
               DenseTensor* u_out,
               DenseTensor* v_out,
               DenseTensor* encode_grad_out,
               DenseTensor* grad_out,
               DenseTensor* k_out,
               DenseTensor* gather_buff);

}  // namespace phi
