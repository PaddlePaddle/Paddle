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

#include "paddle/phi/kernels/argsort_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {

template <typename T, typename Type>
static void FullAssign(Type input_height,
                       Type input_width,
                       int input_dim,
                       const DenseTensor* input,
                       const DenseTensor* indices,
                       T* t_out) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (Type i = 0; i < input_height; ++i) {
    if (input_dim == 1) {
      auto e_input = EigenVector<T>::Flatten(*input);
      auto e_indices = EigenVector<Type>::Flatten(*indices);
      for (Type j = 0; j < input_width; ++j) {
        t_out[i * input_width + e_indices(j)] = e_input(j);
      }
    } else {
      auto e_input = EigenMatrix<T>::Reshape(*input, input_dim - 1);
      auto e_indices = EigenMatrix<Type>::Reshape(*indices, input_dim - 1);
      for (Type j = 0; j < input_width; ++j) {
        t_out[i * input_width + e_indices(i, j)] = e_input(i, j);
      }
    }
  }
}

template <typename T, typename Context>
void ArgsortGradKernel(const Context& dev_ctx,
                       const DenseTensor& indices,
                       const DenseTensor& input,
                       const DenseTensor& out_grad,
                       int axis,
                       bool descending UNUSED,
                       bool stable UNUSED,
                       DenseTensor* in_grad) {
  auto in_dims = indices.dims();
  auto rank = input.dims().size();
  axis = (axis < 0) ? (in_dims.size() + axis) : axis;
  dev_ctx.template Alloc<T>(in_grad);
  auto dxt = EigenVector<T>::Flatten(*in_grad);
  auto& place = *dev_ctx.eigen_device();
  dxt.device(place) = dxt.constant(static_cast<T>(0));
  if (out_grad.numel() == 0) return;

  if (rank == 0) {
    phi::Copy<Context>(dev_ctx, out_grad, dev_ctx.GetPlace(), false, in_grad);
    return;
  }

  // Do full assign
  if (axis == -1 || axis + 1 == in_dims.size()) {
    const int64_t input_height =
        common::product(common::slice_ddim(in_dims, 0, in_dims.size() - 1));
    const int64_t input_width = in_dims[in_dims.size() - 1];

    FullAssign<T, int64_t>(input_height,
                           input_width,
                           in_dims.size(),
                           &out_grad,
                           &indices,
                           in_grad->data<T>());
  } else {
    // If not full assign do transpose
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
      trans_dims[static_cast<int>(i)] = in_dims[trans[i]];
    }

    DenseTensor trans_dO;
    trans_dO.Resize(trans_dims);
    dev_ctx.template Alloc<T>(&trans_dO);
    DenseTensor trans_ind;
    trans_ind.Resize(trans_dims);
    dev_ctx.template Alloc<int64_t>(&trans_ind);
    TransposeKernel<T, Context>(dev_ctx, out_grad, trans, &trans_dO);
    TransposeKernel<int64_t, Context>(dev_ctx, indices, trans, &trans_ind);

    const int64_t input_height = common::product(
        common::slice_ddim(trans_dims, 0, trans_dims.size() - 1));
    const int64_t input_width = trans_dims[trans_dims.size() - 1];

    DenseTensor tmp_out;
    tmp_out.Resize(trans_dims);
    T* t_out = dev_ctx.template Alloc<T>(&tmp_out);

    FullAssign<T, int64_t>(input_height,
                           input_width,
                           in_dims.size(),
                           &trans_dO,
                           &trans_ind,
                           t_out);

    // transpose back
    TransposeKernel<T, Context>(dev_ctx, tmp_out, trans, in_grad);
  }
}

}  // namespace phi
PD_REGISTER_KERNEL(argsort_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::ArgsortGradKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
