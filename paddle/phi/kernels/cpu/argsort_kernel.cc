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

#include "paddle/phi/kernels/argsort_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {

template <typename T, typename Type>
static void FullSort(Type input_height,
                     Type input_width,
                     int input_dim,
                     const DenseTensor* input,
                     T* t_out,
                     Type* t_indices,
                     bool descending) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (Type i = 0; i < input_height; ++i) {
    std::vector<std::pair<T, Type>> col_vec;
    col_vec.reserve(input_width);
    if (input_dim == 1) {
      auto e_input = EigenVector<T>::Flatten(*input);
      for (Type j = 0; j < input_width; ++j) {
        col_vec.push_back(std::pair<T, Type>(e_input(j), j));
      }
    } else {
      auto e_input = EigenMatrix<T>::Reshape(*input, input_dim - 1);
      for (Type j = 0; j < input_width; ++j) {
        col_vec.push_back(std::pair<T, Type>(e_input(i, j), j));
      }
    }
    std::sort(col_vec.begin(),
              col_vec.end(),
              [&](const std::pair<T, Type>& l, const std::pair<T, Type>& r) {
                if (descending)
                  return l.first > r.first;
                else
                  return l.first < r.first;
              });

    for (Type j = 0; j < input_width; ++j) {
      t_out[i * input_width + j] = col_vec[j].first;
      t_indices[i * input_width + j] = col_vec[j].second;
    }
  }
}

template <typename T, typename Context>
void ArgsortKernel(const Context& dev_ctx,
                   const DenseTensor& input,
                   int axis,
                   bool descending,
                   DenseTensor* output,
                   DenseTensor* indices) {
  auto in_dims = input.dims();
  axis = (axis < 0) ? (in_dims.size() + axis) : axis;
  T* out_data = dev_ctx.template Alloc<T>(output);

  // Do full sort
  if (axis == -1 || axis + 1 == in_dims.size()) {
    const int64_t input_height =
        phi::product(phi::slice_ddim(in_dims, 0, in_dims.size() - 1));
    const int64_t input_width = in_dims[in_dims.size() - 1];
    int64_t* ids_data = dev_ctx.template Alloc<int64_t>(indices);
    FullSort<T, int64_t>(input_height,
                         input_width,
                         in_dims.size(),
                         &input,
                         out_data,
                         ids_data,
                         descending);
  } else {
    // If not full sort do transpose
    std::vector<int> trans;
    for (int i = 0; i < axis; i++) {
      trans.push_back(i);
    }
    trans.push_back(in_dims.size() - 1);
    for (int i = axis + 1; i < in_dims.size() - 1; i++) {
      trans.push_back(i);
    }
    trans.push_back(axis);
    phi::DDim trans_dims(in_dims);
    for (size_t i = 0; i < trans.size(); i++) {
      trans_dims[i] = in_dims[trans[i]];
    }

    DenseTensor trans_inp;
    trans_inp.Resize(trans_dims);
    dev_ctx.template Alloc<T>(&trans_inp);
    // Do transpose
    TransposeKernel<T, Context>(dev_ctx, input, trans, &trans_inp);

    const int64_t input_height =
        phi::product(phi::slice_ddim(trans_dims, 0, trans_dims.size() - 1));
    const int64_t input_width = trans_dims[trans_dims.size() - 1];

    DenseTensor tmp_out;
    tmp_out.Resize(trans_dims);
    T* t_out = dev_ctx.template Alloc<T>(&tmp_out);

    DenseTensor tmp_indices;
    tmp_indices.Resize(trans_dims);
    auto* t_ind = dev_ctx.template Alloc<int64_t>(&tmp_indices);

    FullSort<T, int64_t>(input_height,
                         input_width,
                         in_dims.size(),
                         &trans_inp,
                         t_out,
                         t_ind,
                         descending);

    dev_ctx.template Alloc<int64_t>(indices);
    TransposeKernel<int64_t, Context>(dev_ctx, tmp_indices, trans, indices);
    // transpose back
    TransposeKernel<T, Context>(dev_ctx, tmp_out, trans, output);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    argsort, CPU, ALL_LAYOUT, phi::ArgsortKernel, float, double, int, int64_t) {
}
