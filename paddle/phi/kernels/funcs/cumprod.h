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
#include "paddle/common/ddim.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {
static void GetCumprodDimInfo(const DDim& dim,
                              int cumprod_dim,
                              size_t* outer_dim,
                              size_t* mid_dim,
                              size_t* inner_dim) {
  PADDLE_ENFORCE_GE(
      cumprod_dim,
      -dim.size(),
      phi::errors::InvalidArgument(
          "The input dim of CumprodOp should be larger than the opposite "
          "rank of input x which is %d.But received dim=%d",
          -dim.size(),
          cumprod_dim));
  if (dim.size() == 0) {
    PADDLE_ENFORCE_LE(
        cumprod_dim,
        dim.size(),
        phi::errors::InvalidArgument(
            "The input dim of CumprodOp should be smaller than the "
            "rank of input x which is %d.But received dim=%d",
            dim.size(),
            cumprod_dim));
    return;
  }

  PADDLE_ENFORCE_LT(cumprod_dim,
                    dim.size(),
                    phi::errors::InvalidArgument(
                        "The input dim of CumprodOp should be smaller than the "
                        "rank of input x which is %d.But received dim=%d",
                        dim.size(),
                        cumprod_dim));
  if (cumprod_dim < 0) cumprod_dim += dim.size();

  *outer_dim = 1;
  for (int i = 0; i < cumprod_dim; ++i) {
    *outer_dim *= dim[i];
  }
  *mid_dim = dim[cumprod_dim];
  *inner_dim = 1;
  for (int i = cumprod_dim + 1; i < dim.size(); ++i) {
    *inner_dim *= dim[i];
  }
}
}  // namespace phi
