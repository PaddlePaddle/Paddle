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

#include "paddle/phi/kernels/nanmedian_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/nanmedian_kernel_impl.h"
#include "paddle/phi/kernels/top_k_kernel.h"

namespace phi {

template <typename T, typename Context>
void CalcMedianFunc(const Context& dev_ctx,
                    const DenseTensor& x,
                    const std::vector<int64_t>& nan_counts,
                    bool ignore_nan,
                    int64_t sort_k,
                    int64_t stride,
                    int64_t pre_dim,
                    T* o_ptr,
                    int64_t* m_ptr) {
  bool should_ignore_nan = ignore_nan;
  DenseTensor sort_out;
  DenseTensor sort_indices;
  auto sort_dim = x.dims();
  int64_t rank = sort_dim.size();
  sort_dim[rank - 1] = sort_k;
  sort_out.Resize(sort_dim);
  sort_indices.Resize(sort_dim);

  dev_ctx.template Alloc<T>(&sort_out);
  T* sort_out_ptr = sort_out.data<T>();
  dev_ctx.template Alloc<int64_t>(&sort_indices);
  int64_t* sort_indices_ptr = sort_indices.data<int64_t>();

  TopkKernel<T, Context>(
      dev_ctx, x, Scalar(sort_k), -1, false, true, &sort_out, &sort_indices);

  T div_factor = static_cast<T>(2.0);
  int64_t offset = 0;
  int64_t i = 0;
  bool is_ori_odd = stride & 1;
  if (should_ignore_nan) {
    for (i = 0; i < pre_dim; i++) {
      offset = i * sort_k;
      if (nan_counts[i] == stride) {
        m_ptr[i * 2] = -1;
        m_ptr[i * 2 + 1] = -1;
        o_ptr[i] = sort_out_ptr[offset];
      } else {
        int64_t nan_k = nan_counts[i] > 0
                            ? static_cast<int64_t>(stride - nan_counts[i])
                            : sort_k;
        int64_t row_pos = static_cast<int64_t>(nan_k >> 1);
        int64_t pos = offset + row_pos;
        if (nan_k & 1) {
          m_ptr[2 * i] = sort_indices_ptr[pos];
          m_ptr[2 * i + 1] = sort_indices_ptr[pos];
          o_ptr[i] = sort_out_ptr[pos];
        } else {
          m_ptr[2 * i] =
              row_pos > 0 ? sort_indices_ptr[pos - 1] : sort_indices_ptr[pos];
          m_ptr[2 * i + 1] = sort_indices_ptr[pos];
          T m_val_left =
              row_pos > 0 ? sort_out_ptr[pos - 1] : sort_out_ptr[pos];
          T m_val_right = sort_out_ptr[pos];
          o_ptr[i] = (m_val_left + m_val_right) / div_factor;
        }
      }
    }
  } else {
    if (is_ori_odd) {
      for (i = 0; i < pre_dim; i++) {
        offset = i * sort_k;
        int64_t pos = offset + sort_k - 1;
        o_ptr[i] = sort_out_ptr[pos];
        m_ptr[2 * i] = sort_indices_ptr[pos];
        m_ptr[2 * i + 1] = sort_indices_ptr[pos];
      }
    } else {
      for (i = 0; i < pre_dim; i++) {
        offset = i * sort_k;
        int64_t pos = offset + sort_k - 1;
        m_ptr[2 * i] =
            sort_k > 1 ? sort_indices_ptr[pos - 1] : sort_indices_ptr[pos];
        m_ptr[2 * i + 1] = sort_indices_ptr[pos];
        T m_val_left = sort_k > 1 ? sort_out_ptr[pos - 1] : sort_out_ptr[pos];
        T m_val_right = sort_out_ptr[pos];
        o_ptr[i] = (m_val_left + m_val_right) / div_factor;
      }
    }
  }
}

template <typename T, typename Context>
void ProcessMedianKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         T* o_ptr,
                         int64_t* m_ptr,
                         bool ignore_nan) {
  bool should_ignore_nan = ignore_nan;
  const T* x_ptr = x.data<T>();

  int64_t numel = x.numel();
  auto x_dim = x.dims();
  int64_t x_rank = x_dim.size();
  int64_t stride = x_dim[x_rank - 1];
  int64_t pre_dim = numel / stride;
  int64_t i = 0;

  int64_t max_valid_num = 0;
  std::vector<int64_t> nan_counts;
  if (should_ignore_nan) {
    int64_t total_nan_num = 0;
    std::vector<T> col_vec;
    col_vec.reserve(stride);
    col_vec.resize(stride);
    nan_counts.clear();
    nan_counts.reserve(pre_dim);
    nan_counts.resize(pre_dim);
    for (int64_t i = 0; i < pre_dim; i++) {
      col_vec.clear();
      col_vec.insert(
          col_vec.begin(), x_ptr + i * stride, x_ptr + (i + 1) * stride);
      nan_counts[i] =
          std::count_if(col_vec.begin(), col_vec.end(), [&](const T& val) {
            return std::isnan(static_cast<float>(val));
          });
      total_nan_num += nan_counts[i];
      if (stride - nan_counts[i] > max_valid_num)
        max_valid_num = stride - nan_counts[i];
    }
    // all elems are nan
    if (total_nan_num == numel) {
      for (i = 0; i < pre_dim; i++) {
        o_ptr[i] = x_ptr[0];
        m_ptr[2 * i] = -1;
        m_ptr[2 * i + 1] = -1;
      }
      return;
    }
    should_ignore_nan = total_nan_num > 0;
  }

  int64_t sort_k = should_ignore_nan ? max_valid_num : ((stride >> 1) + 1);
  CalcMedianFunc<T, Context>(dev_ctx,
                             x,
                             nan_counts,
                             should_ignore_nan,
                             sort_k,
                             stride,
                             pre_dim,
                             o_ptr,
                             m_ptr);
}

template <typename T, typename Context>
void BaseMedianKernel(const Context& dev_ctx,
                      const DenseTensor& input,
                      const IntArray& axes,
                      DenseTensor* out,
                      DenseTensor* median_index,
                      bool ignore_nan) {
  DenseTensor x;
  auto rank = input.dims().size();
  if ((axes.size() == 0) || rank <= 1) {
    x = input;
    x.Resize({input.numel()});
  } else {
    PreprocessMedianKernel<T, Context>(dev_ctx, input, axes, &x);
  }

  T* o_ptr = dev_ctx.template Alloc<T>(out);
  int64_t* m_ptr = dev_ctx.template Alloc<int64_t>(median_index);
  ProcessMedianKernel<T, Context>(dev_ctx, x, o_ptr, m_ptr, ignore_nan);
  out->Resize(out->dims());
}

template <typename T, typename Context>
void NanmedianKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const IntArray& axes,
                     bool keepdim,
                     DenseTensor* out,
                     DenseTensor* median_index) {
  BaseMedianKernel<T, Context>(dev_ctx, x, axes, out, median_index, true);
}

}  // namespace phi

PD_REGISTER_KERNEL(nanmedian,
                   CPU,
                   ALL_LAYOUT,
                   phi::NanmedianKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
