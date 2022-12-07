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

#include "paddle/phi/kernels/overlap_add_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/overlap_add_functor.h"

namespace phi {

template <typename T, typename Context>
void OverlapAddKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      int hop_length,
                      int axis,
                      DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  const size_t x_rank = x.dims().size();
  const size_t out_rank = out->dims().size();

  const int n_frames = (axis == 0) ? x.dims()[0] : x.dims()[x_rank - 1];
  const int frame_length = (axis == 0) ? x.dims()[1] : x.dims()[x_rank - 2];
  const int seq_length =
      (axis == 0) ? out->dims()[0] : out->dims()[out_rank - 1];

  // auto& dev_ctx = ctx.device_context<Context>();

  DenseTensor x_(x.type());
  x_ = x;

  phi::DDim preserved_dims;
  if (out_rank > 2) {
    // Save dims used to flatten both input and output tensors and restore
    // output tensor.
    phi::DDim x_resized_dims;
    phi::DDim out_resized_dims;
    if (axis == 0) {
      preserved_dims = phi::slice_ddim(out->dims(), 1, out_rank);
      x_resized_dims = {n_frames, frame_length, phi::product(preserved_dims)};
      out_resized_dims = {seq_length, phi::product(preserved_dims)};
    } else {
      preserved_dims = phi::slice_ddim(out->dims(), 0, out_rank - 1);
      x_resized_dims = {phi::product(preserved_dims), frame_length, n_frames};
      out_resized_dims = {phi::product(preserved_dims), seq_length};
    }
    x_.Resize(x_resized_dims);
    out->Resize(out_resized_dims);
  }

  DenseTensor trans_x(x_.type());
  DenseTensor trans_out(out->type());

  // Transpose input and output in case that axis is 0.
  if (axis == 0) {
    if (out_rank == 1U) {
      trans_out = *out;

      std::vector<int> perm_x{1, 0};
      auto x_dims_vec = phi::vectorize(x_.dims());
      for (int i = 0; i < x_.dims().size(); ++i) {
        x_dims_vec[i] = x_.dims()[perm_x[i]];
      }
      trans_x.Resize(phi::make_ddim(x_dims_vec));
      dev_ctx.template Alloc<T>(&trans_x);
      phi::funcs::TransCompute<Context, T>(
          perm_x.size(), dev_ctx, x_, &trans_x, perm_x);
    } else {
      std::vector<int> perm_out{1, 0};
      auto out_dims_vec = phi::vectorize(out->dims());
      for (int i = 0; i < out->dims().size(); ++i) {
        out_dims_vec[i] = out->dims()[perm_out[i]];
      }
      trans_out.Resize(phi::make_ddim(out_dims_vec));
      dev_ctx.template Alloc<T>(&trans_out);
      phi::funcs::TransCompute<Context, T>(
          perm_out.size(), dev_ctx, *out, &trans_out, perm_out);

      std::vector<int> perm_x{2, 1, 0};
      auto x_dims_vec = phi::vectorize(x_.dims());
      for (int i = 0; i < x_.dims().size(); ++i) {
        x_dims_vec[i] = x_.dims()[perm_x[i]];
      }
      trans_x.Resize(phi::make_ddim(x_dims_vec));
      dev_ctx.template Alloc<T>(&trans_x);
      phi::funcs::TransCompute<Context, T>(
          perm_x.size(), dev_ctx, x_, &trans_x, perm_x);
    }
  } else {
    trans_x = x_;
    trans_out = *out;
  }

  OverlapAddFunctor<Context, T>()(dev_ctx,
                                  &trans_x,
                                  &trans_out,
                                  seq_length,
                                  frame_length,
                                  n_frames,
                                  hop_length,
                                  /*is_grad*/ false);

  // Transpose output in case axis is 0.
  if (axis == 0 && out_rank > 1U) {
    std::vector<int> perm_out{1, 0};
    phi::funcs::TransCompute<Context, T>(
        perm_out.size(), dev_ctx, trans_out, out, perm_out);
  }

  // Restore output dims when the number of dims is larger than 2.
  if (out_rank > 2) {
    std::vector<int64_t> restored_out_shape;
    for (int i = 0; i < preserved_dims.size(); i++) {
      restored_out_shape.push_back(preserved_dims[i]);
    }

    if (axis == 0) {
      // (seq_length, ...)
      restored_out_shape.insert(restored_out_shape.begin(), seq_length);
    } else {
      // (..., seq_length)
      restored_out_shape.push_back(seq_length);
    }

    out->Resize(phi::make_ddim(restored_out_shape));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(overlap_add,
                   GPU,
                   ALL_LAYOUT,
                   phi::OverlapAddKernel,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   paddle::platform::complex<float>,
                   paddle::platform::complex<double>) {}
