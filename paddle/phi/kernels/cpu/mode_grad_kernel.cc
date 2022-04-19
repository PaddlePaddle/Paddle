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

#include "paddle/phi/kernels/mode_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/funcs/mode.h"

namespace phi {

template <typename T, typename Context>
void ModeGradKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& indices,
                    const DenseTensor& out_grad,
                    int axis,
                    bool keepdim,
                    DenseTensor* x_grad) {
  auto in_dims = x.dims();
  auto out_dims = indices.dims();

  // axis < 0, get the real axis
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
  T* x_grad_data = dev_ctx.template Alloc<T>(x_grad);

  if (axis == in_dims.size() - 1) {
    // allocate the memory for the input_grad
    // assign the out_grad to input_grad directly
    const int64_t input_height =
        phi::product(phi::slice_ddim(in_dims, 0, in_dims.size() - 1));
    const int64_t input_width = in_dims[in_dims.size() - 1];

    // init the output grad with 0, because some input elements has no grad
    memset(x_grad_data, 0, x_grad->numel() * sizeof(T));
    // Assign the output_grad to input_grad
    if (keepdim) {
      funcs::ModeAssign(input_height,
                        input_width,
                        in_dims.size(),
                        &out_grad,
                        &indices,
                        x_grad_data);
    } else {
      DenseTensor out_grad_tmp;
      dev_ctx.template Alloc<T>(&out_grad_tmp);
      DenseTensor indices_tmp;
      dev_ctx.template Alloc<int64_t>(&indices_tmp);

      phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, &out_grad_tmp);
      phi::Copy(dev_ctx, indices, dev_ctx.GetPlace(), false, &indices_tmp);

      out_grad_tmp.Resize(out_dims);
      indices_tmp.Resize(out_dims);

      funcs::ModeAssign(input_height,
                        input_width,
                        in_dims.size(),
                        &out_grad_tmp,
                        &indices_tmp,
                        x_grad_data);
    }
  } else {
    // can not assign grad to input_grad, must do the transpose
    std::vector<int> trans_axis;
    for (int i = 0; i < axis; i++) {
      trans_axis.emplace_back(i);
    }
    trans_axis.emplace_back(out_dims.size() - 1);
    for (int i = axis + 1; i < out_dims.size() - 1; i++) {
      trans_axis.emplace_back(i);
    }
    trans_axis.emplace_back(axis);
    DDim trans_shape(out_dims);
    DDim trans_in_shape(in_dims);
    for (size_t i = 0; i < trans_axis.size(); i++) {
      trans_shape[i] = out_dims[trans_axis[i]];
      trans_in_shape[i] = in_dims[trans_axis[i]];
    }
    // transpose the out_grad, indices
    DenseTensor trans_dO;
    trans_dO.Resize(trans_shape);
    dev_ctx.template Alloc<T>(&trans_dO);

    DenseTensor trans_ind;
    trans_ind.Resize(trans_shape);
    dev_ctx.template Alloc<int64_t>(&trans_ind);

    int ndims = trans_axis.size();

    if (keepdim) {
      // Do transpose
      funcs::TransCompute<CPUContext, T>(
          ndims, dev_ctx, out_grad, &trans_dO, trans_axis);
      funcs::TransCompute<CPUContext, int64_t>(
          ndims, dev_ctx, indices, &trans_ind, trans_axis);
    } else {
      DenseTensor out_grad_tmp;
      dev_ctx.template Alloc<T>(&out_grad_tmp);

      DenseTensor indices_tmp;
      dev_ctx.template Alloc<int64_t>(&indices_tmp);

      phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, &out_grad_tmp);
      phi::Copy(dev_ctx, indices, dev_ctx.GetPlace(), false, &indices_tmp);
      out_grad_tmp.Resize(out_dims);
      indices_tmp.Resize(out_dims);
      // Do transpose
      funcs::TransCompute<CPUContext, T>(
          ndims, dev_ctx, out_grad_tmp, &trans_dO, trans_axis);
      funcs::TransCompute<CPUContext, int64_t>(
          ndims, dev_ctx, indices_tmp, &trans_ind, trans_axis);
    }
    const int64_t input_height = phi::product(
        phi::slice_ddim(trans_in_shape, 0, trans_in_shape.size() - 1));
    const int64_t input_width = trans_in_shape[trans_in_shape.size() - 1];

    // Assign the out_grad to tranpose input_grad
    DenseTensor tmp_out;
    tmp_out.Resize(trans_in_shape);
    T* t_out = dev_ctx.template Alloc<T>(&tmp_out);
    memset(t_out, 0, x_grad->numel() * sizeof(T));

    funcs::ModeAssign<T, int64_t>(input_height,
                                  input_width,
                                  in_dims.size(),
                                  &trans_dO,
                                  &trans_ind,
                                  t_out);

    // Transpose back
    funcs::TransCompute<CPUContext, T>(
        ndims, dev_ctx, tmp_out, x_grad, trans_axis);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(mode_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::ModeGradKernel,
                   float,
                   double,
                   int32_t,
                   int64_t) {}
