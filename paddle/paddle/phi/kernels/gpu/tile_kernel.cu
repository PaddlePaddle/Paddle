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

#include "paddle/phi/kernels/tile_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"

namespace phi {

template <typename T, typename Context>
void TileKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const IntArray& repeat_times,
                DenseTensor* out) {
  auto x_dims = x.dims();
  auto rank = x_dims.size();
  auto repeat_times_data = repeat_times.GetData();
  int repeat_times_size = repeat_times_data.size();
  rank = std::max(rank, repeat_times_size);

  if (rank == 0) {
    phi::Copy<DeviceContext>(dev_ctx, x, dev_ctx.GetPlace(), false, out);
    return;
  }

  for (size_t i = 0; i < repeat_times_data.size(); ++i) {
    PADDLE_ENFORCE_GT(
        repeat_times_data[i],
        0,
        errors::InvalidArgument(
            "All elements of the input 'repeat_times' for tile op must "
            "be positive integers, but the value received is %d.",
            repeat_times_data[i]));
  }

  auto vec_x_dims = common::vectorize<int>(x_dims);
  if (repeat_times_data.size() < vec_x_dims.size()) {
    int diff = vec_x_dims.size() - repeat_times_data.size();
    repeat_times_data.insert(repeat_times_data.begin(), diff, 1);
  } else {
    int diff = repeat_times_data.size() - vec_x_dims.size();
    vec_x_dims.insert(vec_x_dims.begin(), diff, 1);
  }

  PADDLE_ENFORCE_EQ(
      repeat_times_data.size(),
      vec_x_dims.size(),
      errors::InvalidArgument(
          "The rank (%d) of the input 'x' and the rank (%d) of the input "
          "'repeat_times' for tile op must match after promotion.",
          vec_x_dims.size(),
          repeat_times_data.size()));

  DDim new_x_dims = common::make_ddim(vec_x_dims);
  DDim out_dims(new_x_dims);
  DenseTensor new_x = x;
  vec_x_dims.insert(vec_x_dims.begin(), 1, 1);
  for (size_t i = 0; i < repeat_times_data.size(); ++i) {
    out_dims[i] *= repeat_times_data[i];
    new_x.Resize(common::make_ddim(vec_x_dims));
    std::vector<const DenseTensor*> ins = {&new_x};
    vec_x_dims[i] *= repeat_times_data[i];
    if (i != repeat_times_data.size() - 1) {
      if (repeat_times_data[i] != 1) {
        DenseTensor tmp_out;
        tmp_out.Resize(common::make_ddim(vec_x_dims));
        dev_ctx.template Alloc<T>(&tmp_out);
        std::vector<DenseTensor*> outs = {&tmp_out};
        phi::funcs::BroadcastKernel<T>(
            dev_ctx, ins, &outs, kps::IdentityFunctor<T>(), i);
        tmp_out.Resize(out_dims);
        new_x = tmp_out;
      }
      vec_x_dims[i] *= vec_x_dims[i + 1];
      vec_x_dims[i + 1] = 1;
    } else {
      out->Resize(common::make_ddim(vec_x_dims));
      dev_ctx.template Alloc<T>(out);
      std::vector<DenseTensor*> outs = {out};
      phi::funcs::BroadcastKernel<T>(
          dev_ctx, ins, &outs, kps::IdentityFunctor<T>(), i);
      out->Resize(out_dims);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(tile,
                   GPU,
                   ALL_LAYOUT,
                   phi::TileKernel,
                   bool,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
