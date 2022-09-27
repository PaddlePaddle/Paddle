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

#include "paddle/phi/kernels/kthvalue_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {
template <typename T, typename Type>
static void getKthvalue(Type input_height,
                        Type input_width,
                        int input_dim,
                        const DenseTensor* input,
                        T* t_out,
                        Type* t_indices,
                        const int& k) {
  bool partial_sort_flag = (k * 64) < input_width;
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (Type i = 0; i < input_height; ++i) {
    std::vector<std::pair<T, Type>> col_vec;
    col_vec.reserve(input_width);
    if (input_dim == 1) {
      auto e_input = EigenVector<T>::Flatten(*input);
      for (Type j = 0; j < input_width; ++j) {
        col_vec.emplace_back(std::pair<T, Type>(e_input(j), j));
      }
    } else {
      auto e_input = EigenMatrix<T>::Reshape(*input, input_dim - 1);
      for (Type j = 0; j < input_width; ++j) {
        col_vec.emplace_back(std::pair<T, Type>(e_input(i, j), j));
      }
    }
    if (partial_sort_flag) {
      std::partial_sort(
          col_vec.begin(),
          col_vec.begin() + k,
          col_vec.end(),
          [](const std::pair<T, Type>& l, const std::pair<T, Type>& r) {
            return (!std::isnan(static_cast<double>(l.first)) &&
                    std::isnan(static_cast<double>(r.first))) ||
                   (l.first < r.first);
          });
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
    }
    t_out[i] = col_vec[k - 1].first;
    t_indices[i] = col_vec[k - 1].second;
  }
}

template <typename T, typename Context>
void KthvalueKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    int k,
                    int axis,
                    bool keepdim,
                    DenseTensor* output,
                    DenseTensor* indices) {
  const auto& in_dims = x.dims();
  if (axis < 0) axis += in_dims.size();
  T* output_data = dev_ctx.template Alloc<T>(output);
  int64_t* indices_data = dev_ctx.template Alloc<int64_t>(indices);
  auto out_dims = output->dims();
  if (axis == in_dims.size() - 1) {
    const int64_t& input_height =
        phi::product(phi::slice_ddim(in_dims, 0, in_dims.size() - 1));
    const int64_t& input_width = in_dims[in_dims.size() - 1];
    getKthvalue<T, int64_t>(input_height,
                            input_width,
                            in_dims.size(),
                            &x,
                            output_data,
                            indices_data,
                            k);
  } else {
    std::vector<int> trans;
    for (int i = 0; i < axis; i++) {
      trans.emplace_back(i);
    }
    trans.emplace_back(in_dims.size() - 1);
    for (int i = axis + 1; i < in_dims.size() - 1; i++) {
      trans.emplace_back(i);
    }
    trans.emplace_back(axis);
    if (!keepdim) {
      std::vector<int> tmp_out_shape;
      for (int i = 0; i < axis; i++) {
        tmp_out_shape.emplace_back(in_dims[i]);
      }
      tmp_out_shape.emplace_back(1);
      for (int i = axis + 1; i < in_dims.size(); i++) {
        tmp_out_shape.emplace_back(in_dims[i]);
      }
      DDim tmp_out_dims = phi::make_ddim(tmp_out_shape);
      output->Resize(tmp_out_dims);
      indices->Resize(tmp_out_dims);
    }
    DDim trans_dims(in_dims);
    DDim trans_out_dims(in_dims);

    for (size_t i = 0; i < trans.size(); i++) {
      trans_dims[i] = in_dims[trans[i]];
      trans_out_dims[i] = in_dims[trans[i]];
    }
    trans_out_dims[in_dims.size() - 1] = 1;
    DenseTensor trans_inp;
    trans_inp.Resize(trans_dims);
    dev_ctx.template Alloc<T>(&trans_inp);
    int ndims = trans.size();
    funcs::TransCompute<phi::CPUContext, T>(
        ndims, dev_ctx, x, &trans_inp, trans);

    const int64_t input_height =
        phi::product(phi::slice_ddim(trans_dims, 0, trans_dims.size() - 1));
    const int64_t input_width = trans_dims[trans_dims.size() - 1];
    DenseTensor tmp_out, tmp_indices;
    tmp_out.Resize(trans_out_dims);
    T* t_out = dev_ctx.template Alloc<T>(&tmp_out);
    tmp_indices.Resize(trans_out_dims);
    int64_t* t_ind = dev_ctx.template Alloc<int64_t>(&tmp_indices);
    getKthvalue<T, int64_t>(
        input_height, input_width, in_dims.size(), &trans_inp, t_out, t_ind, k);
    funcs::TransCompute<phi::CPUContext, int64_t>(
        ndims, dev_ctx, tmp_indices, indices, trans);
    funcs::TransCompute<phi::CPUContext, T>(
        ndims, dev_ctx, tmp_out, output, trans);
    if (!keepdim) {
      output->Resize(out_dims);
      indices->Resize(out_dims);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(kthvalue,
                   CPU,
                   ALL_LAYOUT,
                   phi::KthvalueKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
