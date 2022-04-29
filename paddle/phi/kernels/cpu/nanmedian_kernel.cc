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

namespace phi {

template <typename T, typename Context>
void NanmedianKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     bool ignore_nan,
                     DenseTensor* out,
                     DenseTensor* medians) {
  const T* x_ptr = x.data<T>();
  T* o_ptr = dev_ctx.template Alloc<T>(out);
  T* m_ptr = dev_ctx.template Alloc<T>(medians);

  int64_t numel = x.numel();
  auto x_dim = x.dims();
  int64_t x_rank = x_dim.size();
  int64_t stride = x_dim[x_rank - 1];
  int64_t pre_dim = numel / stride;
  int64_t i = 0;

  bool all_nan = true;
  for (i = 0; i < numel; i++) {
    if (!std::isnan(*(x_ptr + i))) {
      all_nan = false;
      break;
    }
  }

  if (all_nan) {
    for (i = 0; i < pre_dim; i++) {
      o_ptr[i] = x_ptr[0];
      m_ptr[2 * i] = x_ptr[0];
      m_ptr[2 * i + 1] = x_ptr[0];
    }
    return;
  }

  std::vector<T> col_vec;
  col_vec.reserve(stride);
  col_vec.resize(stride);
  for (i = 0; i < pre_dim; i++) {
    col_vec.clear();
    col_vec.insert(
        col_vec.begin(), x_ptr + i * stride, x_ptr + (i + 1) * stride);

    int64_t num_nan =
        std::count_if(col_vec.begin(), col_vec.end(), [&](const T& val) {
          return std::isnan(static_cast<double>(val));
        });

    int64_t pos = (stride - num_nan - 1) / 2;
    std::nth_element(col_vec.begin(),
                     col_vec.begin() + pos,
                     col_vec.end(),
                     [](const T& l, const T& r) {
                       return (!std::isnan(static_cast<double>(l)) &&
                               std::isnan(static_cast<double>(r))) ||
                              (l < r);
                     });

    m_ptr[2 * i] = col_vec[pos];
    m_ptr[2 * i + 1] = col_vec[pos];
    if ((stride - num_nan) % 2 == 0) {
      std::nth_element(col_vec.begin(),
                       col_vec.begin() + pos + 1,
                       col_vec.end(),
                       [](const T& l, const T& r) {
                         return (!std::isnan(static_cast<double>(l)) &&
                                 std::isnan(static_cast<double>(r))) ||
                                (l < r);
                       });
      m_ptr[2 * i + 1] = col_vec[pos + 1];
    }
    o_ptr[i] = static_cast<T>((m_ptr[2 * i] + m_ptr[2 * i + 1]) / 2.0);
  }
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
