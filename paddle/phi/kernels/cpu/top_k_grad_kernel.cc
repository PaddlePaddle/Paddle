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

#include "paddle/phi/kernels/top_k_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename Type>
static void FullTopKAssign(const Type& input_height,
                           const Type& input_width,
                           const int& input_dim,
                           const DenseTensor* input,
                           const DenseTensor* indices,
                           T* output_data,
                           const int& k) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (Type i = 0; i < input_height; ++i) {
    if (input_dim == 1) {
      auto e_input = EigenVector<T>::Flatten(*input);
      auto e_indices = EigenVector<Type>::Flatten(*indices);
      for (Type j = 0; j < k; ++j) {
        output_data[i * input_width + e_indices(j)] = e_input(j);
      }
    } else {
      auto e_input = EigenMatrix<T>::Reshape(*input, input_dim - 1);
      auto e_indices = EigenMatrix<Type>::Reshape(*indices, input_dim - 1);
      for (Type j = 0; j < k; ++j) {
        output_data[i * input_width + e_indices(i, j)] = e_input(i, j);
      }
    }
  }
}

template <typename T, typename Context>
void TopkGradKernel(const Context& dev_ctx,
                    const DenseTensor& out_grad,
                    const DenseTensor& x,
                    const DenseTensor& indices,
                    int k,
                    int axis,
                    bool largest,
                    bool sorted,
                    DenseTensor* x_grad) {
  const auto& in_dims = x.dims();
  const auto& out_dims = indices.dims();

  // axis < 0, get the real axis
  axis = (axis < 0) ? (in_dims.size() + axis) : axis;

  T* x_grad_data = dev_ctx.template Alloc<T>(x_grad);
  if (axis + 1 == in_dims.size()) {
    // allocate the memory for the input_grad

    // assign the out_grad to input_grad directly
    const int64_t input_height =
        phi::product(phi::slice_ddim(in_dims, 0, in_dims.size() - 1));
    const int64_t input_width = in_dims[in_dims.size() - 1];

    // init the output grad with 0, because some input elements has no grad
    memset(x_grad_data, 0, x_grad->numel() * sizeof(T));
    // Assign the output_grad to input_grad
    FullTopKAssign(input_height,
                   input_width,
                   in_dims.size(),
                   &out_grad,
                   &indices,
                   x_grad_data,
                   k);
  } else {
    // can not assign grad to input_grad, must do the transpose
    std::vector<int> trans;
    for (int i = 0; i < axis; i++) {
      trans.emplace_back(i);
    }
    trans.emplace_back(out_dims.size() - 1);
    for (int i = axis + 1; i < out_dims.size() - 1; i++) {
      trans.emplace_back(i);
    }
    trans.emplace_back(axis);
    phi::DDim trans_dims(out_dims);
    phi::DDim trans_in_dims(in_dims);
    for (size_t i = 0; i < trans.size(); i++) {
      trans_dims[i] = out_dims[trans[i]];
      trans_in_dims[i] = in_dims[trans[i]];
    }
    // transpose the out_grad, indices
    DenseTensor trans_dO;
    DenseTensor trans_ind;
    trans_dO.Resize(trans_dims);
    trans_ind.Resize(trans_dims);
    dev_ctx.template Alloc<T>(&trans_dO);
    dev_ctx.template Alloc<int64_t>(&trans_ind);
    int ndims = trans.size();

    // Do transpose
    funcs::TransCompute<phi::CPUContext, T>(
        ndims, dev_ctx, out_grad, &trans_dO, trans);
    funcs::TransCompute<phi::CPUContext, int64_t>(
        ndims, dev_ctx, indices, &trans_ind, trans);
    const int64_t input_height = phi::product(
        phi::slice_ddim(trans_in_dims, 0, trans_in_dims.size() - 1));
    const int64_t input_width = trans_in_dims[trans_in_dims.size() - 1];

    // Assign the out_grad to tranpose input_grad
    DenseTensor tmp_out;
    tmp_out.Resize(trans_in_dims);
    T* t_out = dev_ctx.template Alloc<T>(&tmp_out);
    memset(t_out, 0, x_grad->numel() * sizeof(T));

    FullTopKAssign<T, int64_t>(input_height,
                               input_width,
                               in_dims.size(),
                               &trans_dO,
                               &trans_ind,
                               t_out,
                               k);

    // Transpose back
    funcs::TransCompute<phi::CPUContext, T>(
        ndims, dev_ctx, tmp_out, x_grad, trans);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(top_k_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::TopkGradKernel,
                   float,
                   double,
                   int32_t,
                   int64_t) {}
