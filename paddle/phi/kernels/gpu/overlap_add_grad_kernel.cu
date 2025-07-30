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

#include "paddle/phi/kernels/overlap_add_grad_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/overlap_add_functor.h"

namespace phi {

template <typename T, typename Context>
void OverlapAddGradKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& out_grad,
                          int hop_length,
                          int axis,
                          DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);
  const size_t out_grad_rank = out_grad.dims().size();
  const size_t x_grad_rank = x_grad->dims().size();

  const int n_frames =
      (axis == 0) ? x_grad->dims()[0] : x_grad->dims()[x_grad_rank - 1];
  const int frame_length =
      (axis == 0) ? x_grad->dims()[1] : x_grad->dims()[x_grad_rank - 2];
  const int seq_length =
      (axis == 0) ? out_grad.dims()[0] : out_grad.dims()[out_grad_rank - 1];

  // When the number of input dims is larger than 2, it needs to copy
  // from x to resize input into 2d and output into 3d. Moreover, output
  // dims will be restored at the last step.
  DenseTensor out_grad_(out_grad.type());
  out_grad_ = out_grad;

  phi::DDim preserved_dims;
  if (out_grad_rank > 2) {
    // Save dims used to flatten both input and output tensors and restore
    // output tensor.
    phi::DDim x_grad_resized_dims;
    phi::DDim out_grad_resized_dims;
    if (axis == 0) {
      preserved_dims = common::slice_ddim(out_grad_.dims(), 1, out_grad_rank);
      x_grad_resized_dims = {
          n_frames, frame_length, common::product(preserved_dims)};
      out_grad_resized_dims = {seq_length, common::product(preserved_dims)};
    } else {
      preserved_dims =
          common::slice_ddim(out_grad_.dims(), 0, out_grad_rank - 1);
      x_grad_resized_dims = {
          common::product(preserved_dims), frame_length, n_frames};
      out_grad_resized_dims = {common::product(preserved_dims), seq_length};
    }
    x_grad->Resize(x_grad_resized_dims);
    out_grad_.Resize(out_grad_resized_dims);
  }

  DenseTensor trans_x_grad(x_grad->type());
  DenseTensor trans_out_grad(out_grad_.type());

  // Transpose input and output in case that axis is 0.
  if (axis == 0) {
    if (out_grad_rank == 1U) {
      trans_out_grad = out_grad_;

      std::vector<int> perm_x_grad{1, 0};
      auto x_grad_dims_vec = common::vectorize(x_grad->dims());
      for (int i = 0; i < x_grad->dims().size(); ++i) {
        x_grad_dims_vec[i] = x_grad->dims()[perm_x_grad[i]];
      }
      trans_x_grad.Resize(common::make_ddim(x_grad_dims_vec));
      dev_ctx.template Alloc<T>(&trans_x_grad);
      phi::funcs::TransCompute<Context, T>(
          perm_x_grad.size(), dev_ctx, *x_grad, &trans_x_grad, perm_x_grad);
    } else {
      std::vector<int> perm_d_out{1, 0};
      auto out_grad_dims_vec = common::vectorize(out_grad_.dims());
      for (int i = 0; i < out_grad_.dims().size(); ++i) {
        out_grad_dims_vec[i] = out_grad_.dims()[perm_d_out[i]];
      }
      trans_out_grad.Resize(common::make_ddim(out_grad_dims_vec));
      dev_ctx.template Alloc<T>(&trans_out_grad);
      phi::funcs::TransCompute<Context, T>(
          perm_d_out.size(), dev_ctx, out_grad_, &trans_out_grad, perm_d_out);

      std::vector<int> perm_x_grad{2, 1, 0};
      auto x_grad_dims_vec = common::vectorize(x_grad->dims());
      for (int i = 0; i < x_grad->dims().size(); ++i) {
        x_grad_dims_vec[i] = x_grad->dims()[perm_x_grad[i]];
      }
      trans_x_grad.Resize(common::make_ddim(x_grad_dims_vec));
      dev_ctx.template Alloc<T>(&trans_x_grad);
      phi::funcs::TransCompute<Context, T>(
          perm_x_grad.size(), dev_ctx, *x_grad, &trans_x_grad, perm_x_grad);
    }
  } else {
    trans_x_grad = *x_grad;
    trans_out_grad = out_grad_;
  }

  OverlapAddFunctor<Context, T>()(dev_ctx,
                                  &trans_out_grad,
                                  &trans_x_grad,
                                  seq_length,
                                  frame_length,
                                  n_frames,
                                  hop_length,
                                  /*is_grad*/ true);

  // Transpose output in case axis is 0.
  if (axis == 0) {
    if (out_grad_rank == 1U) {
      std::vector<int> perm_x_grad{1, 0};
      phi::funcs::TransCompute<Context, T>(
          perm_x_grad.size(), dev_ctx, trans_x_grad, x_grad, perm_x_grad);
    } else {
      std::vector<int> perm_x_grad{2, 1, 0};
      phi::funcs::TransCompute<Context, T>(
          perm_x_grad.size(), dev_ctx, trans_x_grad, x_grad, perm_x_grad);
    }
  }

  // Restore output dims when the number of dims is larger than 2.
  if (out_grad_rank > 2) {
    std::vector<int64_t> restored_x_grad_shape;
    for (int i = 0; i < preserved_dims.size(); i++) {
      restored_x_grad_shape.push_back(preserved_dims[i]);
    }

    if (axis == 0) {
      // (n_frames, frame_length, ...)
      restored_x_grad_shape.insert(restored_x_grad_shape.begin(), frame_length);
      restored_x_grad_shape.insert(restored_x_grad_shape.begin(), n_frames);
    } else {
      // (..., frame_length, n_frames)
      restored_x_grad_shape.push_back(frame_length);
      restored_x_grad_shape.push_back(n_frames);
    }

    x_grad->Resize(common::make_ddim(restored_x_grad_shape));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(overlap_add_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::OverlapAddGradKernel,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
