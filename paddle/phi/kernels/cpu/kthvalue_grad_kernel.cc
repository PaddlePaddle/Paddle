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

#include "paddle/phi/kernels/kthvalue_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {
template <typename T, typename Type>
static void kthvalueAssign(const Type& input_height,
                           const Type& input_width,
                           const int& input_dim,
                           const DenseTensor* input,
                           const DenseTensor* indices,
                           T* output_data) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (Type i = 0; i < input_height; ++i) {
    if (input_dim == 1) {
      auto e_input = EigenVector<T>::Flatten(*input);
      auto e_indices = EigenVector<Type>::Flatten(*indices);
      output_data[i * input_width + e_indices(0)] = e_input(0);
    } else {
      auto e_input = EigenMatrix<T>::Reshape(*input, input_dim - 1);
      auto e_indices = EigenMatrix<Type>::Reshape(*indices, input_dim - 1);
      output_data[i * input_width + e_indices(i, 0)] = e_input(i, 0);
    }
  }
}

template <typename T, typename Context>
void KthvalueGradKernel(const Context& dev_ctx,
                        const DenseTensor& d_out,
                        const DenseTensor& x,
                        const DenseTensor& indices,
                        int k,
                        int axis,
                        bool keepdim,
                        DenseTensor* d_x) {
  auto in_dims = x.dims();
  auto out_dims = indices.dims();
  axis = (axis < 0) ? (in_dims.size() + axis) : axis;
  if (!keepdim) {
    std::vector<int> tmp_out_shape;
    for (int i = 0; i < axis; i++) {
      tmp_out_shape.emplace_back(out_dims[i]);
    }
    tmp_out_shape.emplace_back(1);
    for (int i = axis + 1; i < in_dims.size(); i++) {
      tmp_out_shape.emplace_back(out_dims[i - 1]);
    }
    out_dims = phi::make_ddim(tmp_out_shape);
  }
  T* x_grad_data = dev_ctx.template Alloc<T>(d_x);
  if (axis == in_dims.size() - 1) {
    const int64_t input_height =
        phi::product(phi::slice_ddim(in_dims, 0, in_dims.size() - 1));
    const int64_t input_width = in_dims[in_dims.size() - 1];
    memset(x_grad_data, 0, d_x->numel() * sizeof(T));
    if (keepdim) {
      kthvalueAssign(input_height,
                     input_width,
                     in_dims.size(),
                     &d_out,
                     &indices,
                     x_grad_data);
    } else {
      DenseTensor out_grad_tmp, indices_tmp;
      out_grad_tmp.Resize(d_out.dims());
      indices_tmp.Resize(indices.dims());
      dev_ctx.template Alloc<T>(&out_grad_tmp);
      dev_ctx.template Alloc<int64_t>(&indices_tmp);
      Copy(dev_ctx, d_out, dev_ctx.GetPlace(), false, &out_grad_tmp);
      Copy(dev_ctx, indices, dev_ctx.GetPlace(), false, &indices_tmp);
      out_grad_tmp.Resize(out_dims);
      indices_tmp.Resize(out_dims);
      kthvalueAssign(input_height,
                     input_width,
                     in_dims.size(),
                     &out_grad_tmp,
                     &indices_tmp,
                     x_grad_data);
    }
  } else {
    std::vector<int> trans;
    for (int i = 0; i < axis; i++) {
      trans.emplace_back(i);
    }
    trans.emplace_back(out_dims.size() - 1);
    for (int i = axis + 1; i < out_dims.size() - 1; i++) {
      trans.emplace_back(i);
    }
    trans.emplace_back(axis);
    DDim trans_dims(out_dims);
    DDim trans_in_dims(in_dims);
    for (size_t i = 0; i < trans.size(); i++) {
      trans_dims[i] = out_dims[trans[i]];
      trans_in_dims[i] = in_dims[trans[i]];
    }
    DenseTensor trans_dO, trans_ind;
    trans_dO.Resize(trans_dims);
    trans_ind.Resize(trans_dims);
    dev_ctx.template Alloc<T>(&trans_dO);
    dev_ctx.template Alloc<int64_t>(&trans_ind);
    int ndims = trans.size();
    if (keepdim) {
      funcs::TransCompute<phi::CPUContext, T>(
          ndims, dev_ctx, d_out, &trans_dO, trans);
      funcs::TransCompute<phi::CPUContext, int64_t>(
          ndims, dev_ctx, indices, &trans_ind, trans);
    } else {
      DenseTensor out_grad_tmp, indices_tmp;
      out_grad_tmp.Resize(d_out.dims());
      indices_tmp.Resize(indices.dims());
      dev_ctx.template Alloc<T>(&out_grad_tmp);
      dev_ctx.template Alloc<int64_t>(&indices_tmp);
      Copy(dev_ctx, d_out, dev_ctx.GetPlace(), false, &out_grad_tmp);
      Copy(dev_ctx, indices, dev_ctx.GetPlace(), false, &indices_tmp);
      out_grad_tmp.Resize(out_dims);
      indices_tmp.Resize(out_dims);
      funcs::TransCompute<phi::CPUContext, T>(
          ndims, dev_ctx, out_grad_tmp, &trans_dO, trans);
      funcs::TransCompute<phi::CPUContext, int64_t>(
          ndims, dev_ctx, indices_tmp, &trans_ind, trans);
    }
    const int64_t input_height = phi::product(
        phi::slice_ddim(trans_in_dims, 0, trans_in_dims.size() - 1));
    const int64_t input_width = trans_in_dims[trans_in_dims.size() - 1];
    DenseTensor tmp_out;
    tmp_out.Resize(trans_in_dims);
    T* t_out = dev_ctx.template Alloc<T>(&tmp_out);
    memset(t_out, 0, d_x->numel() * sizeof(T));
    kthvalueAssign<T, int64_t>(input_height,
                               input_width,
                               in_dims.size(),
                               &trans_dO,
                               &trans_ind,
                               t_out);
    funcs::TransCompute<phi::CPUContext, T>(
        ndims, dev_ctx, tmp_out, d_x, trans);
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(kthvalue_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::KthvalueGradKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
