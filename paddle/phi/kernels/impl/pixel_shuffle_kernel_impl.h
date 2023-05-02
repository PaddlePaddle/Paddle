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
void PixelShuffleKernel(const Context& ctx,
                        const DenseTensor& x,
                        int upscale_factor,
                        const std::string& data_format,
                        DenseTensor* out) {
  auto* in = &x;
  ctx.template Alloc<T>(out);
  int factor = upscale_factor;
  bool channel_last = (data_format == "NHWC");
  const auto& in_dims = in->dims();
  const auto& o_dims = out->dims();

  DenseTensor t(*in);
  if (!channel_last) {
    t.Resize({in_dims[0], o_dims[1], factor, factor, in_dims[2], in_dims[3]});
  } else {
    t.Resize({in_dims[0], in_dims[1], in_dims[2], o_dims[3], factor, factor});
  }
  std::vector<int> axis = {0, 1, 4, 2, 5, 3};

  DenseTensor o(*out);
  if (!channel_last) {
    o.Resize({in_dims[0], o_dims[1], in_dims[2], factor, in_dims[3], factor});
  } else {
    o.Resize({in_dims[0], in_dims[1], factor, in_dims[2], factor, o_dims[3]});
  }
  phi::funcs::Transpose<Context, T, 6> trans;
  trans(ctx, t, &o, axis);
  out->Resize(o_dims);
}

}  // namespace phi
