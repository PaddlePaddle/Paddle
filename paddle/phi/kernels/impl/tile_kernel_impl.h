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
#include <type_traits>
#include <vector>

#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/tile_kernel.h"

namespace phi {

template <typename Context, typename T, int Rank>
void Tile(const Context& dev_ctx,
          const DenseTensor& x,
          std::vector<int64_t> repeat_times,
          DenseTensor* out) {
  auto x_dims = x.dims();
  for (size_t i = 0; i < repeat_times.size(); ++i) {
    PADDLE_ENFORCE_GT(
        repeat_times[i],
        0,
        errors::InvalidArgument(
            "All elements of the input 'repeat_times' for tile op must "
            "be positive integers, but the value received is %d.",
            repeat_times[i]));
  }
  auto vec_x_dims = phi::vectorize<int>(x_dims);
  if (repeat_times.size() < vec_x_dims.size()) {
    int diff = vec_x_dims.size() - repeat_times.size();
    repeat_times.insert(repeat_times.begin(), diff, 1);
  } else {
    int diff = repeat_times.size() - vec_x_dims.size();
    vec_x_dims.insert(vec_x_dims.begin(), diff, 1);
  }
  PADDLE_ENFORCE_EQ(
      repeat_times.size(),
      vec_x_dims.size(),
      errors::InvalidArgument(
          "The rank (%d) of the input 'x' and the rank (%d) of the input "
          "'repeat_times' for tile op must match after promotion.",
          vec_x_dims.size(),
          repeat_times.size()));

  Eigen::DSizes<Eigen::DenseIndex, Rank> bcast_dims;
  for (size_t i = 0; i < repeat_times.size(); ++i) {
    bcast_dims[i] = repeat_times[i];
  }

  DDim new_x_dims = make_ddim(vec_x_dims);
  DDim out_dims(new_x_dims);
  for (size_t i = 0; i < repeat_times.size(); ++i) {
    out_dims[i] *= repeat_times[i];
  }

  out->Resize(out_dims);
  auto eigen_x = EigenTensor<T, Rank>::From(x, new_x_dims);
  dev_ctx.template Alloc<T>(out);

  auto eigen_out = EigenTensor<T, Rank>::From(*out, out_dims);
  auto& place = *dev_ctx.eigen_device();
  // use 32-bit index to speed up
  bool use_32bit_index = eigen_out.size() < Eigen::NumTraits<int>::highest();
  if (use_32bit_index) {
    funcs::EigenBroadcast<std::decay_t<decltype(place)>, T, Rank>::Eval(
        place, To32BitIndex(eigen_out), To32BitIndex(eigen_x), bcast_dims);
  } else {
    funcs::EigenBroadcast<std::decay_t<decltype(place)>, T, Rank>::Eval(
        place, eigen_out, eigen_x, bcast_dims);
  }
}

template <typename T, typename Context>
void TileKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const ScalarArray& repeat_times,
                DenseTensor* out) {
  auto rank = x.dims().size();
  auto& repeat_times_data = repeat_times.GetData();
  int repeat_times_size = repeat_times_data.size();
  rank = std::max(rank, repeat_times_size);

  switch (rank) {
    case 1:
      Tile<Context, T, 1>(dev_ctx, x, repeat_times_data, out);
      break;
    case 2:
      Tile<Context, T, 2>(dev_ctx, x, repeat_times_data, out);
      break;
    case 3:
      Tile<Context, T, 3>(dev_ctx, x, repeat_times_data, out);
      break;
    case 4:
      Tile<Context, T, 4>(dev_ctx, x, repeat_times_data, out);
      break;
    case 5:
      Tile<Context, T, 5>(dev_ctx, x, repeat_times_data, out);
      break;
    case 6:
      Tile<Context, T, 6>(dev_ctx, x, repeat_times_data, out);
      break;
  }
}

}  // namespace phi
