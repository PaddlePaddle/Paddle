// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
void RankAttentionKernel(const Context &dev_ctx,
                         const DenseTensor &x UNUSED,
                         const DenseTensor &rank_offset UNUSED,
                         const DenseTensor &rank_param UNUSED,
                         int max_rank UNUSED,
                         int max_size UNUSED,
                         DenseTensor *input_help UNUSED,
                         DenseTensor *out UNUSED,
                         DenseTensor *ins_rank UNUSED) {
  PADDLE_ENFORCE_EQ(
      dev_ctx.GetPlace().GetType() == phi::AllocationType::GPU,
      true,
      phi::errors::Unimplemented("Rank Attention only supports GPU now."));
}
}  // namespace phi
