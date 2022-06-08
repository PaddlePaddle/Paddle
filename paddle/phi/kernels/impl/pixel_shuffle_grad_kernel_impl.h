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
#include <algorithm>
#include <string>
#include <vector>

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename Context>
void PixelShuffleGradKernel(const Context& ctx,
                            const DenseTensor& out_grad,
                            int upscale_factor,
                            const std::string& data_format,
                            DenseTensor* x_grad) {
  auto* dout = &out_grad;
  auto* dx = x_grad;
  ctx.template Alloc<T>(dx);
  int factor = upscale_factor;
  bool channel_last = (data_format == "NHWC");
  const auto& do_dims = dout->dims();
  const auto& dx_dims = dx->dims();

  DenseTensor t(*dout);
  if (!channel_last) {
    t.Resize({do_dims[0], do_dims[1], dx_dims[2], factor, dx_dims[3], factor});
  } else {
    t.Resize({do_dims[0], dx_dims[1], factor, dx_dims[2], factor, do_dims[3]});
  }
  std::vector<int> axis = {0, 1, 3, 5, 2, 4};

  DenseTensor o(*dx);
  if (!channel_last) {
    o.Resize({do_dims[0], do_dims[1], factor, factor, dx_dims[2], dx_dims[3]});
  } else {
    o.Resize({do_dims[0], dx_dims[1], dx_dims[2], do_dims[3], factor, factor});
  }
  phi::funcs::Transpose<Context, T, 6> trans;
  trans(ctx, t, &o, axis);
  dx->Resize(dx_dims);
}

}  // namespace phi
