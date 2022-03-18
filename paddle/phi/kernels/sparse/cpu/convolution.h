/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <set>

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/sparse/convolution_kernel.h"

namespace phi {
namespace sparse {

using Dims4D = phi::funcs::sparse::Dims4D;

// such as: kernel(3, 3, 3), kernel_size = 27
// counter_per_weight: (kernel_size)
// TODO(zhangkaihuo): optimize performance with multithreading
template <typename T, typename Context>
void ProductRuleBook(const Context& dev_ctx,
                     const SparseCooTensor& x,
                     const std::vector<int>& kernel_sizes,
                     const std::vector<int>& paddings,
                     const std::vector<int>& dilations,
                     const std::vector<int>& strides,
                     const DDim& out_dims,
                     const bool subm,
                     DenseTensor* rulebook,
                     DenseTensor* counter_per_kernel) {
  const int64_t non_zero_num = x.nnz();
  const auto& non_zero_indices = x.non_zero_indices();
  const int* indices_ptr = non_zero_indices.data<int>();
  int* counter_ptr = counter_per_kernel->data<int>();
  int kernel_size = kernel_sizes[0] * kernel_sizes[1] * kernel_sizes[2];
  memset(counter_ptr, 0, kernel_size * sizeof(int));

  int rulebook_len = 0;
  // calc the rulebook_len
  const auto& x_dims = x.dims();
  const Dims4D c_x_dims(x_dims[0], x_dims[3], x_dims[2], x_dims[1]);
  const Dims4D c_kernel_dims(
      1, kernel_sizes[2], kernel_sizes[1], kernel_sizes[0]);
  const Dims4D c_out_dims(out_dims[0], out_dims[3], out_dims[2], out_dims[1]);
  const Dims4D c_paddings(1, paddings[2], paddings[1], paddings[0]);
  const Dims4D c_strides(1, strides[2], strides[1], strides[0]);
  const Dims4D c_dilations(1, dilations[2], dilations[1], dilations[0]);

  std::set<int> hash_in;
  if (subm) {
    for (int i = 0; i < non_zero_num; i++) {
      int batch = indices_ptr[i];
      int in_z = indices_ptr[i + non_zero_num];
      int in_y = indices_ptr[i + 2 * non_zero_num];
      int in_x = indices_ptr[i + 3 * non_zero_num];
      int index = phi::funcs::sparse::PointToIndex<DDim>(
          batch, in_x, in_y, in_z, x_dims);
      hash_in.insert(index);
    }
  }

  auto f_calc_rulebook = [&](int* rulebook_ptr) {
    int kernel_index = 0, rulebook_index = 0;
    for (int kz = 0; kz < kernel_sizes[0]; kz++) {
      for (int ky = 0; ky < kernel_sizes[1]; ky++) {
        for (int kx = 0; kx < kernel_sizes[2]; kx++) {
          ++kernel_index;
          for (int64_t i = 0; i < non_zero_num; i++) {
            int batch = indices_ptr[i];
            int in_z = indices_ptr[i + non_zero_num];
            int in_y = indices_ptr[i + 2 * non_zero_num];
            int in_x = indices_ptr[i + 3 * non_zero_num];
            int out_z = (in_z + paddings[0] - kz * dilations[0]) / strides[0];
            int out_y = (in_y + paddings[1] - ky * dilations[1]) / strides[1];
            int out_x = (in_x + paddings[2] - kx * dilations[2]) / strides[2];
            if (phi::funcs::sparse::Check(c_x_dims,
                                          c_kernel_dims,
                                          c_paddings,
                                          c_dilations,
                                          c_strides,
                                          in_x,
                                          in_y,
                                          in_z,
                                          kx,
                                          ky,
                                          kz)) {
              if (subm) {
                int out_index = phi::funcs::sparse::PointToIndex<DDim>(
                    batch, out_x, out_y, out_z, out_dims);
                if (hash_in.find(out_index) == hash_in.end()) {
                  continue;
                }
              }

              if (rulebook_ptr == nullptr) {
                counter_ptr[kernel_index - 1] += 1;
                ++rulebook_len;
              } else {
                rulebook_ptr[rulebook_index] = kernel_index - 1;
                rulebook_ptr[rulebook_index + rulebook_len] = i;  // in_i
                rulebook_ptr[rulebook_index + rulebook_len * 2] =
                    phi::funcs::sparse::PointToIndex<DDim>(
                        batch, out_x, out_y, out_z, out_dims);  // out_index
                ++rulebook_index;
              }
            }
          }
        }
      }
    }
  };

  f_calc_rulebook(nullptr);
  // alloc the rulebook
  DenseTensorMeta rulebook_meta(
      DataType::INT32, {3, rulebook_len}, DataLayout::NCHW);
  rulebook->set_meta(rulebook_meta);
  dev_ctx.Alloc(rulebook, rulebook->dtype(), rulebook->numel() * sizeof(int));
  int* rulebook_ptr = rulebook->data<int>();
  f_calc_rulebook(rulebook_ptr);
}

template <typename T, typename Context>
void UpdateRulebookAndOutIndex(const Context& dev_ctx,
                               const SparseCooTensor& x,
                               const int kernel_size,
                               const int out_channels,
                               const DDim& out_dims,
                               DenseTensor* rulebook,
                               SparseCooTensor* out) {
  std::set<int> out_indexs;
  int n = rulebook->dims()[1];
  int* rulebook_ptr = rulebook->data<int>();
  for (int i = 0; i < n; i++) {
    out_indexs.insert(rulebook_ptr[i + n * 2]);
  }

  int out_non_zero_num = out_indexs.size();
  const int64_t sparse_dim = 4;
  DenseTensorMeta indices_meta(
      DataType::INT32, {sparse_dim, out_non_zero_num}, DataLayout::NCHW);
  DenseTensorMeta values_meta(
      x.dtype(), {out_non_zero_num, out_channels}, x.layout());
  phi::DenseTensor out_indices = phi::Empty(dev_ctx, std::move(indices_meta));
  phi::DenseTensor out_values = phi::Empty(dev_ctx, std::move(values_meta));
  int* out_indices_ptr = out_indices.data<int>();
  int i = 0;
  for (auto it = out_indexs.begin(); it != out_indexs.end(); it++, i++) {
    const int index = *it;
    int batch, x, y, z;
    phi::funcs::sparse::IndexToPoint<DDim>(index, out_dims, &batch, &x, &y, &z);
    out_indices_ptr[i] = batch;
    out_indices_ptr[i + out_non_zero_num] = z;
    out_indices_ptr[i + out_non_zero_num * 2] = y;
    out_indices_ptr[i + out_non_zero_num * 3] = x;
  }
  for (i = 0; i < n; i++) {
    int out_index = rulebook_ptr[i + n * 2];
    rulebook_ptr[i + n * 2] =
        std::distance(out_indexs.begin(), out_indexs.find(out_index));
  }

  out->SetMember(out_indices, out_values, out_dims, true);
}

template <typename T>
void Gather(
    const T* x, const int* indexs, const int n, const int channels, T* out) {
  for (int i = 0; i < n; i++) {
    int real_i = indexs[i];
    memcpy(out + i * channels, x + real_i * channels, channels * sizeof(T));
  }
}

template <typename T>
void Scatter(
    const T* x, const int* indexs, const int n, const int channels, T* out) {
  for (int i = 0; i < n; i++) {
    int real_i = indexs[i];
    for (int j = 0; j < channels; j++) {
      out[real_i * channels + j] += x[i * channels + j];
    }
  }
}

}  // namespace sparse
}  // namespace phi
