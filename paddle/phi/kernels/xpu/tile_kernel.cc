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

#include <type_traits>
#include <vector>

#include "paddle/phi/kernels/tile_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"

namespace phi {

template <typename T, typename Context>
void TileKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const IntArray& repeat_times_arr,
                DenseTensor* out) {
  auto rank = x.dims().size();
  PADDLE_ENFORCE_GE(
      rank,
      1,
      errors::InvalidArgument(
          "The rank of the input 'x' for tile op must be a positive "
          "integer, but the value received is %d.",
          rank));
  PADDLE_ENFORCE_LE(
      rank,
      MAX_RANK_SUPPORTED,
      errors::InvalidArgument(
          "The rank of the input 'x' for tile op "
          "must be less than or equal to %d, but the value received is %d.",
          MAX_RANK_SUPPORTED,
          rank));
  std::vector<int64_t> repeat_times = repeat_times_arr.GetData();
  int repeat_times_size = repeat_times.size();
  PADDLE_ENFORCE_GE(
      repeat_times_size,
      1,
      errors::InvalidArgument(
          "The number of elements of the input 'repeat_times' for tile "
          "op must be positive, but the value received is %d.",
          repeat_times_size));
  PADDLE_ENFORCE_LE(
      repeat_times_size,
      MAX_RANK_SUPPORTED,
      errors::InvalidArgument(
          "The number of elements of the input 'repeat_times' for tile op "
          "must be less than or equal to %d, but the value received is %d.",
          MAX_RANK_SUPPORTED,
          repeat_times_size));

  auto in_dims = x.dims();
  for (size_t i = 0; i < repeat_times.size(); ++i) {
    PADDLE_ENFORCE_GT(
        repeat_times[i],
        0,
        errors::InvalidArgument(
            "All elements of the input 'repeat_times' for tile op must "
            "be positive integers, but the value received is %d.",
            repeat_times[i]));
  }
  auto vec_in_dims = phi::vectorize<int>(in_dims);
  if (repeat_times.size() < vec_in_dims.size()) {
    int diff = vec_in_dims.size() - repeat_times.size();
    repeat_times.insert(repeat_times.begin(), diff, 1);
  } else {
    int diff = repeat_times.size() - vec_in_dims.size();
    vec_in_dims.insert(vec_in_dims.begin(), diff, 1);
  }
  PADDLE_ENFORCE_EQ(
      repeat_times.size(),
      vec_in_dims.size(),
      errors::InvalidArgument(
          "The rank (%d) of the input 'x' and the rank (%d) of the input "
          "'repeat_times' for tile op must match after promotion.",
          vec_in_dims.size(),
          repeat_times.size()));

  DDim new_in_dims = phi::make_ddim(vec_in_dims);
  DDim out_dims(new_in_dims);

  for (size_t i = 0; i < repeat_times.size(); ++i) {
    out_dims[i] *= repeat_times[i];
  }
  auto vec_out_dims = phi::vectorize<int>(out_dims);
  out->Resize(out_dims);
  dev_ctx.template Alloc<T>(out);

  std::vector<int64_t> temp(repeat_times.size(), 1);
  if (repeat_times == temp) {
    phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
    return;
  }

  int ret = XPU_SUCCESS;
  if (std::is_same<T, bool>::value) {
    ret = xpu::broadcast<int8_t>(dev_ctx.x_context(),
                                 reinterpret_cast<const int8_t*>(x.data<T>()),
                                 reinterpret_cast<int8_t*>(out->data<T>()),
                                 vec_in_dims,
                                 vec_out_dims);

  } else {
    ret = xpu::broadcast<T>(dev_ctx.x_context(),
                            x.data<T>(),
                            out->data<T>(),
                            vec_in_dims,
                            vec_out_dims);
  }
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "broadcast");
}

}  // namespace phi

PD_REGISTER_KERNEL(
    tile, XPU, ALL_LAYOUT, phi::TileKernel, bool, float, int, int64_t) {}
