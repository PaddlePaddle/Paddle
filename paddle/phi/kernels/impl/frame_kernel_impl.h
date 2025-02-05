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
void FrameKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 int frame_length,
                 int hop_length,
                 int axis,
                 DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  const size_t x_rank = x.dims().size();
  const size_t out_rank = out->dims().size();
  const int n_frames = (axis == 0) ? out->dims()[0] : out->dims()[out_rank - 1];
  const int seq_length = (axis == 0) ? x.dims()[0] : x.dims()[x_rank - 1];
  // When the number of input dims is larger than 2, it needs to copy
  // from x to resize input into 2d and output into 3d. Moreover, output
  // dims will be restored at the last step.
  DenseTensor x_tmp = x;

  DDim preserved_dims;
  if (x_rank > 2) {
    // Save dims used to flatten both input and output tensors and restore
    // output tensor.
    DDim x_resized_dims;
    DDim out_resized_dims;
    if (axis == 0) {
      preserved_dims = common::slice_ddim(x_tmp.dims(), 1, x_rank);
      x_resized_dims = {seq_length, common::product(preserved_dims)};
      out_resized_dims = {
          n_frames, frame_length, common::product(preserved_dims)};
    } else {
      preserved_dims = common::slice_ddim(x_tmp.dims(), 0, x_rank - 1);
      x_resized_dims = {common::product(preserved_dims), seq_length};
      out_resized_dims = {
          common::product(preserved_dims), frame_length, n_frames};
    }
    x_tmp.Resize(x_resized_dims);
    out->Resize(out_resized_dims);
  }

  DenseTensor trans_x;
  DenseTensor trans_out;

  // Transpose input and output in case that axis is 0.
  if (axis == 0) {
    if (x_rank == 1U) {
      trans_x = x_tmp;

      std::vector<int> perm_out{1, 0};
      auto out_dims_vec = common::vectorize(out->dims());
      for (int i = 0; i < out->dims().size(); ++i) {
        out_dims_vec[i] = out->dims()[perm_out[i]];
      }
      trans_out.Resize(common::make_ddim(out_dims_vec));

      dev_ctx.template Alloc<T>(&trans_out);
      phi::funcs::TransCompute<Context, T>(
          perm_out.size(), dev_ctx, *out, &trans_out, perm_out);
    } else {
      std::vector<int> perm_x{1, 0};
      auto x_dims_vec = common::vectorize(x_tmp.dims());
      for (int i = 0; i < x_tmp.dims().size(); ++i) {
        x_dims_vec[i] = x_tmp.dims()[perm_x[i]];
      }
      trans_x.Resize(common::make_ddim(x_dims_vec));
      dev_ctx.template Alloc<T>(&trans_x);
      phi::funcs::TransCompute<Context, T>(
          perm_x.size(), dev_ctx, x_tmp, &trans_x, perm_x);

      std::vector<int> perm_out{2, 1, 0};
      auto out_dims_vec = common::vectorize(out->dims());
      for (int i = 0; i < out->dims().size(); ++i) {
        out_dims_vec[i] = out->dims()[perm_out[i]];
      }
      trans_out.Resize(common::make_ddim(out_dims_vec));
      dev_ctx.template Alloc<T>(&trans_out);
      phi::funcs::TransCompute<Context, T>(
          perm_out.size(), dev_ctx, *out, &trans_out, perm_out);
    }
  } else {
    trans_x = x_tmp;
    trans_out = *out;
  }

  phi::funcs::FrameFunctor<Context, T>()(dev_ctx,
                                         &trans_x,
                                         &trans_out,
                                         seq_length,
                                         frame_length,
                                         n_frames,
                                         hop_length,
                                         /*is_grad*/ false);

  // Transpose output in case axis is 0.
  if (axis == 0) {
    if (x_rank == 1U) {
      std::vector<int> perm_out{1, 0};
      funcs::TransCompute<Context, T>(
          perm_out.size(), dev_ctx, trans_out, out, perm_out);
    } else {
      std::vector<int> perm_out{2, 1, 0};
      funcs::TransCompute<Context, T>(
          perm_out.size(), dev_ctx, trans_out, out, perm_out);
    }
  }

  // Restore output dims when the number of dims is larger than 2.
  if (x_rank > 2) {
    std::vector<int64_t> restored_out_shape;
    for (int i = 0; i < preserved_dims.size(); i++) {
      restored_out_shape.push_back(preserved_dims[i]);
    }

    if (axis == 0) {
      // (n_frames, frame_length, ...)
      restored_out_shape.insert(restored_out_shape.begin(), frame_length);
      restored_out_shape.insert(restored_out_shape.begin(), n_frames);
    } else {
      // (..., frame_length, n_frames)
      restored_out_shape.push_back(frame_length);
      restored_out_shape.push_back(n_frames);
    }

    out->Resize(common::make_ddim(restored_out_shape));
  }
}

}  // namespace phi
