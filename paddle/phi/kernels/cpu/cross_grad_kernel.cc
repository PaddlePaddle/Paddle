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

#include "paddle/phi/kernels/cross_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/for_range.h"

namespace phi {

template <typename T, typename Context>
void CrossGradKernel(const Context &dev_ctx,
                     const DenseTensor &x,
                     const DenseTensor &y,
                     const DenseTensor &out_grad,
                     int axis,
                     DenseTensor *x_grad,
                     DenseTensor *y_grad) {
  auto &input_x = x;
  auto &input_y = y;
  auto &input_out_grad = out_grad;
  auto *output_x_grad = x_grad;
  auto *output_y_grad = y_grad;
  int dim = axis;
  auto input_x_dims = input_x.dims();
  if (dim != DDim::kMaxRank) {
    PADDLE_ENFORCE_EQ(
        dim < input_x_dims.size() && dim >= (0 - input_x_dims.size()),
        true,
        errors::OutOfRange(
            "Attr(dim) is out of range, It's expected "
            "to be in range of [-%d, %d]. But received Attr(dim) = %d.",
            input_x_dims.size(),
            input_x_dims.size() - 1,
            dim));
    if (dim < 0) {
      dim += input_x_dims.size();
    }

    PADDLE_ENFORCE_EQ(
        input_x_dims[dim] == 3,
        true,
        errors::InvalidArgument(
            "Input(X/Y).dims[dim] must be equal to 3. But received: "
            "Input(X/Y).dims[dim] = [%d].",
            input_x_dims[dim]));
  } else {
    for (auto i = 0; i < input_x_dims.size(); i++) {
      if (input_x_dims[i] == 3) {
        dim = i;
        break;
      }
    }
    PADDLE_ENFORCE_EQ(
        dim == DDim::kMaxRank,
        false,
        errors::InvalidArgument("There must be at least one dimension 'd' "
                                "so that Input(X/Y).dims()[d] is equal to 3. "
                                "But received: Input(X/Y).dims() == [%s].",
                                input_x_dims));
  }
  auto outer_loops = 1;
  for (auto i = 0; i < dim; i++) {
    outer_loops *= static_cast<int>(input_x_dims[i]);
  }
  auto slice_size = 1;
  for (auto i = dim + 1; i < input_x_dims.size(); i++) {
    slice_size *= static_cast<int>(input_x_dims[i]);
  }

  int64_t numel = x.numel();
  DenseTensor x_conj, y_conj;
  DenseTensorMeta meta_xy(x.dtype(), x.dims());
  x_conj.set_meta(meta_xy);
  y_conj.set_meta(meta_xy);

  auto *input_x_conj_data = dev_ctx.template Alloc<T>(&x_conj);

  auto *input_y_conj_data = dev_ctx.template Alloc<T>(&y_conj);

  phi::funcs::ForRange<Context> for_range(dev_ctx, numel);
  phi::funcs::ConjFunctor<T> functor_x(
      input_x.data<T>(), numel, input_x_conj_data);
  phi::funcs::ConjFunctor<T> functor_y(
      input_y.data<T>(), numel, input_y_conj_data);
  for_range(functor_x);
  for_range(functor_y);

  std::vector<T> input_x_vec, input_y_vec, input_dout_vec;
  phi::TensorToVector(x_conj, dev_ctx, &input_x_vec);
  phi::TensorToVector(y_conj, dev_ctx, &input_y_vec);
  phi::TensorToVector(input_out_grad, dev_ctx, &input_dout_vec);
  std::vector<T> out_dx_vec(output_x_grad->numel());
  std::vector<T> out_dy_vec(output_y_grad->numel());

  dev_ctx.template Alloc<T>(output_x_grad);
  dev_ctx.template Alloc<T>(output_y_grad);

  for (auto i = 0; i < outer_loops; i++) {
    for (auto j = 0; j < 3; j++) {
      auto dst_pos = (3 * i + j) * slice_size;
      auto in_pos1 = (3 * i + ((j + 1) % 3)) * slice_size;
      auto in_pos2 = (3 * i + ((j + 2) % 3)) * slice_size;
      for (auto k = 0; k < slice_size; k++) {
        out_dx_vec[dst_pos + k] =
            input_dout_vec[in_pos2 + k] * input_y_vec[in_pos1 + k] -
            input_dout_vec[in_pos1 + k] * input_y_vec[in_pos2 + k];
        out_dy_vec[dst_pos + k] =
            input_dout_vec[in_pos1 + k] * input_x_vec[in_pos2 + k] -
            input_dout_vec[in_pos2 + k] * input_x_vec[in_pos1 + k];
      }
    }
  }
  phi::TensorFromVector(out_dx_vec, dev_ctx, output_x_grad);
  phi::TensorFromVector(out_dy_vec, dev_ctx, output_y_grad);
  output_x_grad->Resize(input_x_dims);
  output_y_grad->Resize(input_x_dims);
}

}  // namespace phi
PD_REGISTER_KERNEL(cross_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::CrossGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
