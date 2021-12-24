// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/pten/core/dense_tensor.h"

namespace pten {

template <typename T, typename DevCtx>
void MatmulGrad(const DevCtx& dev_ctx,
                const DenseTensor& x,
                const DenseTensor& y,
                const DenseTensor& dout,
                bool transpose_x,
                bool transpose_y,
                DenseTensor* dx,
                DenseTensor* dy);

template <typename T, typename DevCtx>
void MatmulDoubleGrad(const DevCtx& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      const DenseTensor& dout,
                      const DenseTensor& ddx,
                      const DenseTensor& ddy,
                      bool transpose_x,
                      bool transpose_y,
                      DenseTensor* dx,
                      DenseTensor* dy,
                      DenseTensor* ddout);

template <typename T, typename DevCtx>
void MatmulTripleGrad(const DevCtx& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      const DenseTensor& dout,
                      const DenseTensor& ddx,
                      const DenseTensor& ddy,
                      const DenseTensor& d_dx,
                      const DenseTensor& d_dy,
                      const DenseTensor& d_ddout,
                      bool transpose_x,
                      bool transpose_y,
                      DenseTensor* out_d_x,
                      DenseTensor* out_d_y,
                      DenseTensor* out_d_dout,
                      DenseTensor* out_d_ddx,
                      DenseTensor* out_d_ddy);

}  // namespace pten
