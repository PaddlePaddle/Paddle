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

#include "paddle/phi/kernels/funcs/frame_functor.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {
template <typename T, typename Context>
void FrameGradKernel(const Context& dev_ctx,
                     const DenseTensor& x UNUSED,
                     const DenseTensor& dout,
                     int frame_length,
                     int hop_length,
                     int axis,
                     DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  const size_t dout_rank = dout.dims().size();
  const size_t dx_rank = dx->dims().size();
  const int n_frames =
      (axis == 0) ? dout.dims()[0] : dout.dims()[dout_rank - 1];
  const int seq_length = (axis == 0) ? dx->dims()[0] : dx->dims()[dx_rank - 1];
  DenseTensor dout_tmp = dout;

  DDim preserved_dims;
  if (dx_rank > 2) {
    // Save dims used to flatten both input and output tensors and restore
    // output tensor.
    DDim dx_resized_dims;
    DDim dout_resized_dims;
    if (axis == 0) {
      preserved_dims = common::slice_ddim(dx->dims(), 1, dx_rank);
      dx_resized_dims = {seq_length, common::product(preserved_dims)};
      dout_resized_dims = {
          n_frames, frame_length, common::product(preserved_dims)};
    } else {
      preserved_dims = common::slice_ddim(dx->dims(), 0, dx_rank - 1);
      dx_resized_dims = {common::product(preserved_dims), seq_length};
      dout_resized_dims = {
          common::product(preserved_dims), frame_length, n_frames};
    }
    dx->Resize(dx_resized_dims);
    dout_tmp.Resize(dout_resized_dims);
  }

  DenseTensor trans_dx;
  DenseTensor trans_dout;

  // Transpose input and output in case that axis is 0.
  if (axis == 0) {
    if (dx_rank == 1U) {
      trans_dx = *dx;

      std::vector<int> perm_dout{1, 0};
      auto dout_dims_vec = common::vectorize(dout_tmp.dims());
      for (int i = 0; i < dout_tmp.dims().size(); ++i) {
        dout_dims_vec[i] = dout_tmp.dims()[perm_dout[i]];
      }
      trans_dout.Resize(common::make_ddim(dout_dims_vec));
      dev_ctx.template Alloc<T>(&trans_dout);
      phi::funcs::TransCompute<Context, T>(
          perm_dout.size(), dev_ctx, dout_tmp, &trans_dout, perm_dout);
    } else {
      std::vector<int> perm_dx{1, 0};
      auto dx_dims_vec = common::vectorize(dx->dims());
      for (int i = 0; i < dx->dims().size(); ++i) {
        dx_dims_vec[i] = dx->dims()[perm_dx[i]];
      }
      trans_dx.Resize(common::make_ddim(dx_dims_vec));
      dev_ctx.template Alloc<T>(&trans_dx);
      phi::funcs::TransCompute<Context, T>(
          perm_dx.size(), dev_ctx, *dx, &trans_dx, perm_dx);

      std::vector<int> perm_dout{2, 1, 0};
      auto dout_dims_vec = common::vectorize(dout_tmp.dims());
      for (int i = 0; i < dout_tmp.dims().size(); ++i) {
        dout_dims_vec[i] = dout_tmp.dims()[perm_dout[i]];
      }
      trans_dout.Resize(common::make_ddim(dout_dims_vec));
      dev_ctx.template Alloc<T>(&trans_dout);
      phi::funcs::TransCompute<Context, T>(
          perm_dout.size(), dev_ctx, dout_tmp, &trans_dout, perm_dout);
    }
  } else {
    trans_dx = *dx;
    trans_dout = dout_tmp;
  }

  phi::funcs::FrameFunctor<Context, T>()(dev_ctx,
                                         &trans_dout,
                                         &trans_dx,
                                         seq_length,
                                         frame_length,
                                         n_frames,
                                         hop_length,
                                         /*is_grad*/ true);

  // Transpose output in case axis is 0.
  if (axis == 0 && dx_rank > 1U) {
    std::vector<int> perm_dx{1, 0};
    phi::funcs::TransCompute<Context, T>(
        perm_dx.size(), dev_ctx, trans_dx, dx, perm_dx);
  }

  // Restore output dims when the number of dims is larger than 2.
  if (dx_rank > 2) {
    std::vector<int64_t> restored_dx_shape;
    for (int i = 0; i < preserved_dims.size(); i++) {
      restored_dx_shape.push_back(preserved_dims[i]);
    }

    if (axis == 0) {
      // (seq_length, ...)
      restored_dx_shape.insert(restored_dx_shape.begin(), seq_length);
    } else {
      // (..., seq_length)
      restored_dx_shape.push_back(seq_length);
    }

    dx->Resize(common::make_ddim(restored_dx_shape));
  }
}
}  // namespace phi
