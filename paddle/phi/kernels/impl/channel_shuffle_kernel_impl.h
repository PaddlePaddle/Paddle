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

#include <string>
#include <vector>

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename Context>
void ChannelShuffleKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          int groups,
                          const std::string& data_format,
                          DenseTensor* out) {
  auto* in = &x;
  dev_ctx.template Alloc<T>(out);
  bool channel_last = (data_format == "NHWC");
  const auto& in_dims = in->dims();
  const auto& o_dims = out->dims();

  DenseTensor t(*in);
  if (!channel_last) {
    t.Resize({in_dims[0], groups, in_dims[1] / groups, in_dims[2], in_dims[3]});
  } else {
    t.Resize({in_dims[0], in_dims[1], in_dims[2], groups, in_dims[3] / groups});
  }
  auto axis = !channel_last ? std::vector<int>{0, 2, 1, 3, 4}
                            : std::vector<int>{0, 1, 2, 4, 3};

  DenseTensor o(*out);
  if (!channel_last) {
    o.Resize({in_dims[0], in_dims[1] / groups, groups, in_dims[2], in_dims[3]});
  } else {
    o.Resize({in_dims[0], in_dims[1], in_dims[2], in_dims[3] / groups, groups});
  }
  phi::funcs::Transpose<Context, T, 5> trans;
  trans(dev_ctx, t, &o, axis);
  out->Resize(o_dims);
}

}  // namespace phi
