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

#include "paddle/phi/kernels/group_norm_kernel.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <string>

#include "paddle/common/layout.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/extensions.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename Context>
void GroupNormKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const paddle::optional<DenseTensor>& scale,
                     const paddle::optional<DenseTensor>& bias,
                     float epsilon,
                     int groups,
                     const std::string& data_layout_str,
                     DenseTensor* y,
                     DenseTensor* mean,
                     DenseTensor* var) {
  const DataLayout data_layout = common::StringToDataLayout(data_layout_str);
  const auto scale_ptr = scale.get_ptr();
  const auto bias_ptr = bias.get_ptr();

  const auto x_dims = x.dims();
  const int C = static_cast<int>(
      data_layout == DataLayout::kNCHW ? x_dims[1] : x_dims[x_dims.size() - 1]);
  const int group_size = C / groups;

  dev_ctx.template Alloc<T>(y);
  dev_ctx.template Alloc<T>(mean);
  dev_ctx.template Alloc<T>(var);

  auto* x_data = x.data<T>();
  auto* y_data = y->data<T>();
  auto* mean_data = mean->data<T>();
  auto* var_data = var->data<T>();

  const T* scale_data = nullptr;
  if (scale_ptr) scale_data = scale_ptr->data<T>();
  const T* bias_data = nullptr;
  if (bias_ptr) bias_data = bias_ptr->data<T>();

  int imsize = 1;
  if (data_layout == DataLayout::kNCHW) {
    for (int i = 2; i < x_dims.size(); ++i) {
      imsize *= static_cast<int>(x_dims[i]);
    }
  } else {
    for (int i = 1; i < x_dims.size() - 1; ++i) {
      imsize *= static_cast<int>(x_dims[i]);
    }
  }
  auto* iter_x_data = x_data;
  auto* iter_y_data = y_data;
  for (int bid = 0; bid < x_dims[0]; bid++) {
    for (int gid = 0; gid < groups; gid++) {
      const int64_t M = 8;
      std::array<T, M> x_mean_arr;
      std::array<T, M> x_var_arr;
      std::fill(x_mean_arr.begin(), x_mean_arr.end(), T(0));
      std::fill(x_var_arr.begin(), x_var_arr.end(), T(0));
      T x_mean = 0, x_var = 0;
      int number = std::min(group_size, static_cast<int>(C - gid * group_size));
      auto* tmp_x = iter_x_data;
      auto* x_src_data = iter_x_data;
      auto* tmp_y = iter_y_data;
      auto* y_src_data = iter_y_data;

      if (data_layout == DataLayout::kNCHW) {
        for (int cid = 0; cid < number; cid++) {
          int imid = 0;
          for (imid = 0; imid < imsize - (imsize % M);
               imid += M, iter_x_data += M) {
            // TODO(gaoxiang): Because AVX/AVX2/AVX512 can not directly used
            // in template class/function, before we complete high
            // performance cpu vector extension, temporarily unrolling
            // loop to get high precision and performance
            x_mean_arr[0] += iter_x_data[0];
            x_var_arr[0] += iter_x_data[0] * iter_x_data[0];
            x_mean_arr[1] += iter_x_data[1];
            x_var_arr[1] += iter_x_data[1] * iter_x_data[1];
            x_mean_arr[2] += iter_x_data[2];
            x_var_arr[2] += iter_x_data[2] * iter_x_data[2];
            x_mean_arr[3] += iter_x_data[3];
            x_var_arr[3] += iter_x_data[3] * iter_x_data[3];
            x_mean_arr[4] += iter_x_data[4];
            x_var_arr[4] += iter_x_data[4] * iter_x_data[4];
            x_mean_arr[5] += iter_x_data[5];
            x_var_arr[5] += iter_x_data[5] * iter_x_data[5];
            x_mean_arr[6] += iter_x_data[6];
            x_var_arr[6] += iter_x_data[6] * iter_x_data[6];
            x_mean_arr[7] += iter_x_data[7];
            x_var_arr[7] += iter_x_data[7] * iter_x_data[7];
          }
          x_mean =
              std::accumulate(x_mean_arr.cbegin(), x_mean_arr.cend(), x_mean);
          x_var = std::accumulate(x_var_arr.cbegin(), x_var_arr.cend(), x_var);
          std::fill(x_mean_arr.begin(), x_mean_arr.end(), T(0));
          std::fill(x_var_arr.begin(), x_var_arr.end(), T(0));
          for (; imid < imsize; imid++, iter_x_data++) {
            x_mean += iter_x_data[0];
            x_var += iter_x_data[0] * iter_x_data[0];
          }
        }
      } else {
        for (int cid = 0; cid < number; cid++) {
          iter_x_data = tmp_x + cid;
          int imid = 0;
          for (imid = 0; imid < imsize - (imsize % M);
               imid += M, iter_x_data += M * C) {
            // TODO(gaoxiang): Because AVX/AVX2/AVX512 can not directly used
            // in template class/function, before we complete high
            // performance cpu vector extension, temporarily unrolling
            // loop to get high precision and performance
            x_mean_arr[0] += iter_x_data[0 * C];
            x_var_arr[0] += iter_x_data[0 * C] * iter_x_data[0 * C];
            x_mean_arr[1] += iter_x_data[1 * C];
            x_var_arr[1] += iter_x_data[1 * C] * iter_x_data[1 * C];
            x_mean_arr[2] += iter_x_data[2 * C];
            x_var_arr[2] += iter_x_data[2 * C] * iter_x_data[2 * C];
            x_mean_arr[3] += iter_x_data[3 * C];
            x_var_arr[3] += iter_x_data[3 * C] * iter_x_data[3 * C];
            x_mean_arr[4] += iter_x_data[4 * C];
            x_var_arr[4] += iter_x_data[4 * C] * iter_x_data[4 * C];
            x_mean_arr[5] += iter_x_data[5 * C];
            x_var_arr[5] += iter_x_data[5 * C] * iter_x_data[5 * C];
            x_mean_arr[6] += iter_x_data[6 * C];
            x_var_arr[6] += iter_x_data[6 * C] * iter_x_data[6 * C];
            x_mean_arr[7] += iter_x_data[7 * C];
            x_var_arr[7] += iter_x_data[7 * C] * iter_x_data[7 * C];
          }
          x_mean =
              std::accumulate(x_mean_arr.cbegin(), x_mean_arr.cend(), x_mean);
          x_var = std::accumulate(x_var_arr.cbegin(), x_var_arr.cend(), x_var);
          std::fill(x_mean_arr.begin(), x_mean_arr.end(), T(0));
          std::fill(x_var_arr.begin(), x_var_arr.end(), T(0));
          for (; imid < imsize; imid++, iter_x_data += C) {
            x_mean += iter_x_data[0];
            x_var += iter_x_data[0] * iter_x_data[0];
          }
        }
        iter_x_data = tmp_x + group_size;
      }

      x_mean /= number * imsize;
      x_var /= number * imsize;
      x_var = std::max(x_var - x_mean * x_mean, T(0));
      T var_inv = T(1) / std::sqrt(x_var + epsilon);
      mean_data[bid * groups + gid] = x_mean;
      var_data[bid * groups + gid] = x_var;

      if (data_layout == DataLayout::kNCHW) {
        for (int cid = 0; cid < number; cid++) {
          for (int imid = 0; imid < imsize; imid++, tmp_x++, iter_y_data++) {
            T val = (tmp_x[0] - x_mean) * var_inv;
            if (scale_data) val *= scale_data[gid * group_size + cid];
            if (bias_data) val += bias_data[gid * group_size + cid];
            iter_y_data[0] = val;
          }
        }
      } else {
        for (int cid = 0; cid < number; cid++) {
          tmp_x = x_src_data + cid;
          iter_y_data = y_src_data + cid;
          for (int imid = 0; imid < imsize;
               imid++, tmp_x += C, iter_y_data += C) {
            T val = (tmp_x[0] - x_mean) * var_inv;
            if (scale_data) val *= scale_data[gid * group_size + cid];
            if (bias_data) val += bias_data[gid * group_size + cid];
            iter_y_data[0] = val;
          }
        }
        iter_y_data = tmp_y + group_size;
      }
    }
    if (data_layout == DataLayout::kNHWC) {
      iter_x_data = x_data + (bid + 1) * C * imsize;
      iter_y_data = y_data + (bid + 1) * C * imsize;
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    group_norm, CPU, ALL_LAYOUT, phi::GroupNormKernel, float, double) {}
