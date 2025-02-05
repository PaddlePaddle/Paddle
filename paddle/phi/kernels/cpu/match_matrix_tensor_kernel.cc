// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/kernels/funcs/search_compute.h"

namespace phi {

template <typename T, typename Context>
void CPUMatchMatrixTensorOPKernel(const Context& dev_ctx,
                                  const DenseTensor& x_in,
                                  const DenseTensor& y_in,
                                  const DenseTensor& w_in,
                                  int dim_t,
                                  DenseTensor* out,
                                  DenseTensor* tmp) {
  auto* x = &x_in;
  auto* y = &y_in;
  auto* w = &w_in;

  const auto& x_lod = x->lod();
  PADDLE_ENFORCE_EQ(x_lod.empty(),
                    false,
                    common::errors::InvalidArgument(
                        "The Input(X) should hold LoD information, but "
                        "received Input(X).lod() is empty."));
  const auto& x_lod_0 = x_lod[0];
  PADDLE_ENFORCE_GE(x_lod_0.size(),
                    2,
                    common::errors::InvalidArgument(
                        "The dimensions of Input(X)'s LoD data should be "
                        "equal to 2, but received %d.",
                        x_lod_0.size()));
  auto x_dims = x->dims();
  PADDLE_ENFORCE_EQ(x_dims[0],
                    static_cast<int64_t>(x_lod_0.back()),
                    common::errors::InvalidArgument(
                        "The last element of Input(X)'s LoD data should be "
                        "equal to the first dimension of Input(X). "
                        "But received the last element of Input(X)'s LoD "
                        "data is %d, the first dimension of Input(X) is %d.",
                        x_lod_0.back(),
                        x_dims[0]));
  const auto& y_lod = y->lod();
  PADDLE_ENFORCE_EQ(y_lod.empty(),
                    false,
                    common::errors::InvalidArgument(
                        "The Input(Y) should hold LoD information, but "
                        "received Input(Y).lod() is empty."));
  const auto& y_lod_0 = y_lod[0];
  PADDLE_ENFORCE_GE(y_lod_0.size(),
                    2,
                    common::errors::InvalidArgument(
                        "The dimensions of Input(Y)'s LoD data should be "
                        "equal to 2, but received %d.",
                        y_lod_0.size()));
  auto y_dims = y->dims();
  PADDLE_ENFORCE_EQ(y_dims[0],
                    static_cast<int64_t>(y_lod_0.back()),
                    common::errors::InvalidArgument(
                        "The last element of Input(Y)'s LoD data should be "
                        "equal to the first dimension of Input(Y). "
                        "But received the last element of Input(Y)'s LoD "
                        "data is %d, the first dimension of Input(Y) is %d.",
                        y_lod_0.back(),
                        y_dims[0]));

  PADDLE_ENFORCE_EQ(x_lod_0.size(),
                    y_lod_0.size(),
                    common::errors::InvalidArgument(
                        "The dimensions of Input(X)'s and Input(Y)'s LoD "
                        "data should be equal. "
                        "But received the dimensions of Input(X)'s LoD is "
                        "%d, the dimensions of Input(Y)'s LoD is %d.",
                        x_lod_0.size(),
                        y_lod_0.size()));

  int64_t out_dim_0 = 0;
  int64_t tmp_dim_0 = -1;
  for (size_t i = 1; i < x_lod_0.size(); i++) {
    int64_t x_len = x_lod_0[i] - x_lod_0[i - 1];
    int64_t y_len = y_lod_0[i] - y_lod_0[i - 1];
    out_dim_0 += (x_len * y_len);
  }
  out_dim_0 *= dim_t;

  tmp_dim_0 = x_dims[0] * dim_t * x_dims[1];
  std::vector<int64_t> out_dims_vec{out_dim_0};
  out_dims_vec.push_back(1);
  std::vector<int64_t> tmp_dims_vec{tmp_dim_0};
  tmp_dims_vec.push_back(1);

  auto& out_meta = out->meta();
  phi::DenseTensorMeta new_out_meta(out_meta.dtype,
                                    common::make_ddim(out_dims_vec),
                                    out_meta.layout,
                                    out_meta.legacy_lod);
  out->set_meta(new_out_meta);

  auto& tmp_meta = tmp->meta();
  phi::DenseTensorMeta new_tmp_meta(tmp_meta.dtype,
                                    common::make_ddim(tmp_dims_vec),
                                    tmp_meta.layout,
                                    tmp_meta.legacy_lod);
  tmp->set_meta(new_tmp_meta);

  int64_t dim_in = x->dims()[1];

  const auto& offset_l = x->lod()[0];
  const auto& offset_r = y->lod()[0];

  std::vector<size_t> top_offset;
  size_t top_size = 0;
  top_offset.push_back(top_size);
  for (size_t b = 0; b < x->lod()[0].size() - 1; b++) {
    size_t len_l = offset_l[b + 1] - offset_l[b];
    size_t len_r = offset_r[b + 1] - offset_r[b];
    top_size += dim_t * len_l * len_r;
    top_offset.push_back(top_size);
  }
  auto* out_data = dev_ctx.template Alloc<T>(out);
  memset(out_data, 0.0, out->dims()[0] * out->dims()[1] * sizeof(T));

  auto* bottom_l_data = x->data<T>();
  auto* bottom_r_data = y->data<T>();
  auto* t_data = w->data<T>();
  auto* bottom_l_trans_data = dev_ctx.template Alloc<T>(tmp);
  memset(bottom_l_trans_data, 0.0, tmp->dims()[0] * tmp->dims()[1] * sizeof(T));

  auto blas = phi::funcs::GetBlas<phi::CPUContext, T>(dev_ctx);

  phi::funcs::call_gemm(blas,
                        CblasNoTrans,
                        CblasNoTrans,
                        x->dims()[0],
                        dim_t * dim_in,
                        dim_in,
                        1.0f,
                        bottom_l_data,
                        t_data,
                        0.0f,
                        bottom_l_trans_data);

  for (size_t b = 0; b < x->lod()[0].size() - 1; b++) {
    for (int t = 0; t < dim_t; t++) {
      size_t len_l = offset_l[b + 1] - offset_l[b];
      size_t len_r = offset_r[b + 1] - offset_r[b];
      auto* top_data = out_data + top_offset[b] + t * len_l * len_r;
      const auto* l_t_data =
          bottom_l_trans_data + offset_l[b] * dim_t * dim_in + t * dim_in;
      const auto* r_data = bottom_r_data + offset_r[b] * dim_in;
      auto blas_2 = phi::funcs::GetBlas<phi::CPUContext, T>(dev_ctx);
      phi::funcs::call_gemm_with_lda(blas_2,
                                     CblasNoTrans,
                                     CblasTrans,
                                     len_l,
                                     len_r,
                                     dim_in,
                                     1.0f,
                                     l_t_data,
                                     r_data,
                                     0.0f,
                                     top_data,
                                     dim_t * dim_in);
    }
  }

  phi::LegacyLoD out_lod;
  out_lod.push_back(top_offset);

  out->set_lod(out_lod);
}

template <typename T, typename Context>
void CPUMatchMatrixTensorOPGradKernel(const Context& dev_ctx,
                                      const DenseTensor& x_in,
                                      const DenseTensor& y_in,
                                      const DenseTensor& w_in,
                                      const DenseTensor& tmp_in,
                                      const DenseTensor& out_grad,
                                      int dim_t,
                                      DenseTensor* x_grad,
                                      DenseTensor* y_grad,
                                      DenseTensor* w_grad) {
  auto* x = &x_in;
  auto* y = &y_in;
  auto* w = &w_in;
  auto* tmp = &tmp_in;

  int64_t dim_in = x->dims()[1];

  const auto& offset_l = x->lod()[0];
  const auto& offset_r = y->lod()[0];
  std::vector<size_t> top_offset;
  size_t top_size = 0;
  top_offset.push_back(top_size);
  for (size_t b = 0; b < x->lod()[0].size() - 1; b++) {
    size_t len_l = offset_l[b + 1] - offset_l[b];
    size_t len_r = offset_r[b + 1] - offset_r[b];
    top_size += dim_t * len_l * len_r;
    top_offset.push_back(top_size);
  }

  auto* bottom_l_data = x->data<T>();
  auto* bottom_r_data = y->data<T>();
  auto* bottom_l_trans_data = tmp->data<T>();

  auto* d_out = &out_grad;
  auto* d_x = x_grad;
  auto* d_y = y_grad;

  phi::DenseTensor tmp_grad;
  tmp_grad.Resize(tmp->dims());
  auto* d_tmp_data = dev_ctx.template Alloc<T>(&tmp_grad);
  auto* top_diff = d_out->data<T>();
  auto* bottom_l_diff = dev_ctx.template Alloc<T>(d_x);
  auto* bottom_r_diff = dev_ctx.template Alloc<T>(d_y);
  auto* bottom_l_trans_diff = const_cast<T*>(d_tmp_data);
  memset(bottom_l_diff, 0.0, x->dims()[0] * x->dims()[1] * sizeof(T));
  memset(bottom_r_diff, 0.0, y->dims()[0] * y->dims()[1] * sizeof(T));
  memset(bottom_l_trans_diff, 0.0, tmp->dims()[0] * tmp->dims()[1] * sizeof(T));

  for (size_t b = 0; b < x->lod()[0].size() - 1; b++) {
    for (int t = 0; t < dim_t; t++) {
      size_t len_l = offset_l[b + 1] - offset_l[b];
      size_t len_r = offset_r[b + 1] - offset_r[b];

      for (size_t i = 0; i < len_l; i++) {
        for (size_t j = 0; j < len_r; j++) {
          auto diff =
              top_diff[top_offset[b] + t * len_l * len_r + i * len_r + j];
          auto* l_trans_data = bottom_l_trans_data +
                               (offset_l[b] + i) * dim_in * dim_t + t * dim_in;
          auto* l_trans_diff = bottom_l_trans_diff +
                               (offset_l[b] + i) * dim_in * dim_t + t * dim_in;
          auto* r_data = bottom_r_data + (offset_r[b] + j) * dim_in;
          auto* r_diff = bottom_r_diff + (offset_r[b] + j) * dim_in;
          if (diff != 0.0) {
            phi::funcs::axpy(r_data, l_trans_diff, dim_in, diff);
            phi::funcs::axpy(l_trans_data, r_diff, dim_in, diff);
          }
        }
      }
    }
  }

  auto blas = phi::funcs::GetBlas<phi::CPUContext, T>(dev_ctx);

  auto* t_data = w->data<T>();
  auto* d_w = w_grad;
  auto* t_diff = dev_ctx.template Alloc<T>(d_w);
  memset(t_diff, 0.0, w->dims()[0] * w->dims()[1] * w->dims()[2] * sizeof(T));
  // bottom_diff
  phi::funcs::call_gemm(blas,
                        CblasNoTrans,
                        CblasTrans,
                        x->dims()[0],
                        dim_in,
                        dim_t * dim_in,
                        1.0f,
                        bottom_l_trans_diff,
                        t_data,
                        1.0f,
                        bottom_l_diff);

  // t_diff
  phi::funcs::call_gemm(blas,
                        CblasTrans,
                        CblasNoTrans,
                        dim_in,
                        dim_t * dim_in,
                        x->dims()[0],
                        1.0f,
                        bottom_l_data,
                        bottom_l_trans_diff,
                        1.0f,
                        t_diff);
}
}  // namespace phi

PD_REGISTER_KERNEL(match_matrix_tensor,
                   CPU,
                   ALL_LAYOUT,
                   phi::CPUMatchMatrixTensorOPKernel,
                   float) {}
PD_REGISTER_KERNEL(match_matrix_tensor_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::CPUMatchMatrixTensorOPGradKernel,
                   float) {}
