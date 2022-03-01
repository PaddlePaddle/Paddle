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

#include "paddle/phi/kernels/top_k_v2_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/transpose.h"

namespace phi {

template <typename T, typename Type>
static void FullTopK(Type input_height,
                     Type input_width,
                     int input_dim,
                     const framework::Tensor* input,
                     T* t_out,
                     Type* t_indices,
                     const int& k,
                     const bool& largest,
                     const bool& sorted) {
  // when the k is small, will the partial sort
  bool partial_sort_flag = (k * 64) < input_width;

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (Type i = 0; i < input_height; ++i) {
    std::vector<std::pair<T, Type>> col_vec;
    col_vec.reserve(input_width);
    if (input_dim == 1) {
      auto e_input = framework::EigenVector<T>::Flatten(*input);
      for (Type j = 0; j < input_width; ++j) {
        col_vec.emplace_back(std::pair<T, Type>(e_input(j), j));
      }
    } else {
      auto e_input = framework::EigenMatrix<T>::Reshape(*input, input_dim - 1);
      for (Type j = 0; j < input_width; ++j) {
        col_vec.emplace_back(std::pair<T, Type>(e_input(i, j), j));
      }
    }
    if (partial_sort_flag) {
      std::partial_sort(
          col_vec.begin(),
          col_vec.begin() + k,
          col_vec.end(),
          [&largest](const std::pair<T, Type>& l, const std::pair<T, Type>& r) {
            if (largest) {
              return (std::isnan(static_cast<double>(l.first)) &&
                      !std::isnan(static_cast<double>(r.first))) ||
                     (l.first > r.first);
            } else {
              return (!std::isnan(static_cast<double>(l.first)) &&
                      std::isnan(static_cast<double>(r.first))) ||
                     (l.first < r.first);
            }
          });
    } else {
      // use the nth-element to get the K-larger or K-small element
      if (largest) {
        std::nth_element(
            col_vec.begin(),
            col_vec.begin() + k - 1,
            col_vec.end(),
            [](const std::pair<T, Type>& l, const std::pair<T, Type>& r) {
              return (std::isnan(static_cast<double>(l.first)) &&
                      !std::isnan(static_cast<double>(r.first))) ||
                     (l.first > r.first);
            });
        // the nth-element will get the unorder elements, sort the element
        if (sorted) {
          std::sort(col_vec.begin(),
                    col_vec.begin() + k - 1,
                    [&largest](const std::pair<T, Type>& l,
                               const std::pair<T, Type>& r) {
                      return (std::isnan(static_cast<double>(l.first)) &&
                              !std::isnan(static_cast<double>(r.first))) ||
                             (l.first > r.first);
                    });
        }
      } else {
        std::nth_element(
            col_vec.begin(),
            col_vec.begin() + k - 1,
            col_vec.end(),
            [](const std::pair<T, Type>& l, const std::pair<T, Type>& r) {
              return (!std::isnan(static_cast<double>(l.first)) &&
                      std::isnan(static_cast<double>(r.first))) ||
                     (l.first < r.first);
            });
        // the nth-element will get the unorder elements, sort the element
        if (sorted) {
          std::sort(
              col_vec.begin(),
              col_vec.begin() + k - 1,
              [](const std::pair<T, Type>& l, const std::pair<T, Type>& r) {
                return (!std::isnan(static_cast<double>(l.first)) &&
                        std::isnan(static_cast<double>(r.first))) ||
                       (l.first < r.first);
              });
        }
      }
    }
    for (Type j = 0; j < k; ++j) {
      t_out[i * k + j] = col_vec[j].first;
      t_indices[i * k + j] = col_vec[j].second;
    }
  }
}

template <typename T, typename Context>
void TopkV2Kernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& k_t,
                  int k,
                  int axis,
                  bool largest,
                  bool sorted,
                  DenseTensor* out,
                  DenseTensor* indices) {
  // Get the top k elements of each row of input tensor
  const auto& x_dims = x.dims();

  // axis < 0, cacluate the real axis
  if (axis < 0) {
    axis += x_dims.size();
  }

  // if K tensor is not null, will the use K tesnor as k
  if (k_t) {
    k = k_t.data<int>()[0];
    auto out_dims = out->dims();
    // accroding to axis to set K value in the dim
    out_dims[axis] = k;
    out->Resize(out_dims);
    indices->Resize(out_dims);
  }

  T* out_data = dev_ctx.template Alloc<T>(out);
  int64_t* indices_data = dev_ctx.template Alloc<int64_t>(indices);
  const auto& out_dims = out->dims();
  if (axis + 1 == x_dims.size()) {
    const int64_t& input_height =
        phi::product(phi::slice_ddim(x_dims, 0, x_dims.size() - 1));
    const int64_t& input_width = x_dims[x_dims.size() - 1];
    FullTopK<T, int64_t>(input_height,
                         input_width,
                         x_dims.size(),
                         input,
                         out_data,
                         indices_data,
                         k,
                         largest,
                         sorted);
  } else {
    // if the topk dims is not last dim, will tranpose and do topk
    std::vector<int> trans;
    for (int i = 0; i < axis; i++) {
      trans.emplace_back(i);
    }
    trans.push_back(x_dims.size() - 1);
    for (int i = axis + 1; i < x_dims.size() - 1; i++) {
      trans.emplace_back(i);
    }
    trans.emplace_back(axis);

    // get the trans input_dims, out_dims
    phi::DDim trans_dims(x_dims);
    phi::DDim trans_out_dims(out->dims());
    for (size_t i = 0; i < trans.size(); i++) {
      trans_dims[i] = x_dims[trans[i]];
    }
    for (size_t i = 0; i < trans.size(); i++) {
      trans_out_dims[i] = out_dims[trans[i]];
    }

    DenseTensor trans_inp;
    trans_inp->Resize(trans_dims);
    dev_ctx.template Alloc<T>(trans_inp);
    int ndims = trans.size();

    // transpose the input value
    TransCompute<phi::CPUContext, T>(ndims, dev_ctx, *input, &trans_inp, trans);

    const int64_t input_height =
        phi::product(phi::slice_ddim(trans_dims, 0, trans_dims.size() - 1));
    const int64_t input_width = trans_dims[trans_dims.size() - 1];

    // Allocate the temp tensor to the save the topk indices, values
    DenseTensor tmp_out;
    DenseTensor tmp_indices;
    tmp_out->Resize(trans_out_dims);
    tmp_indices->Resize(trans_out_dims);
    T* t_out = dev_ctx.template Alloc<T>(tmp_out);
    auto* t_ind = dev_ctx.template Alloc<T>(tmp_indices);

    // get the TopK value
    FullTopK<T, int64_t>(input_height,
                         input_width,
                         x_dims.size(),
                         &trans_inp,
                         t_out,
                         t_ind,
                         k,
                         largest,
                         sorted);
    // transpose back
    TransCompute<phi::CPUContext, int64_t>(
        ndims, dev_ctx, tmp_indices, indices, trans);
    TransCompute<phi::CPUContext, T>(ndims, dev_ctx, tmp_out, out, trans);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(top_k_v2,
                   CPU,
                   ALL_LAYOUT,
                   phi::TopkV2Kernel,
                   float,
                   double,
                   int32_t,
                   int64_t) {}
