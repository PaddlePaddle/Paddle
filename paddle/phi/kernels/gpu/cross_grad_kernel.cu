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

#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cross_grad_kernel.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"

namespace phi {

using paddle::platform::PADDLE_CUDA_NUM_THREADS;

template <typename T, typename IndexCalculator>
__global__ void CrossGrad(const T* x,
                          const T* y,
                          const T* out,
                          T* out_dx,
                          T* out_dy,
                          const int stride,
                          const int N,
                          IndexCalculator index_calculator) {
  CUDA_KERNEL_LOOP(i, N) {
    int offset = index_calculator(i);

    auto pos0 = offset + 0 * stride;
    auto pos1 = offset + 1 * stride;
    auto pos2 = offset + 2 * stride;

    out_dx[pos0] = out[pos2] * y[pos1] - out[pos1] * y[pos2];
    out_dy[pos0] = out[pos1] * x[pos2] - out[pos2] * x[pos1];

    out_dx[pos1] = out[pos0] * y[pos2] - out[pos2] * y[pos0];
    out_dy[pos1] = out[pos2] * x[pos0] - out[pos0] * x[pos2];

    out_dx[pos2] = out[pos1] * y[pos0] - out[pos0] * y[pos1];
    out_dy[pos2] = out[pos0] * x[pos1] - out[pos1] * x[pos0];
  }
}

template <typename T, typename Context>
void CrossGradKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     const DenseTensor& out_grad,
                     int axis,
                     DenseTensor* x_grad,
                     DenseTensor* y_grad) {
  auto& input_x = x;
  auto& input_y = y;
  auto& input_out_grad = out_grad;
  auto* output_x_grad = x_grad;
  auto* output_y_grad = y_grad;
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

  std::vector<int> cal_dims;
  std::vector<int> left_strides;
  std::vector<int> full_strides;

  int full_dim = 1;
  int left_dim = 1;
  for (auto i = 0; i < input_x_dims.size(); i++) {
    full_strides.insert(full_strides.begin(), full_dim);
    full_dim *= input_x_dims[input_x_dims.size() - i - 1];
    if (i == dim) {
      continue;
    }
    cal_dims.push_back(i);
    left_strides.insert(left_strides.begin(), left_dim);
    left_dim *= input_x_dims[input_x_dims.size() - i - 1];
  }

  const auto* input_x_data = input_x.data<T>();
  const auto* input_y_data = input_y.data<T>();
  const auto* input_out_grad_data = input_out_grad.data<T>();

  auto* output_x_grad_data = dev_ctx.template Alloc<T>(x_grad);
  auto* output_y_grad_data = dev_ctx.template Alloc<T>(y_grad);

  auto index_calculator = phi::funcs::IndexCalculator(
      input_x_dims.size() - 1, cal_dims, left_strides, full_strides);

  int64_t numel = x.numel();
  int threads = PADDLE_CUDA_NUM_THREADS;
  int blocks = (numel + threads - 1) / threads;

  CrossGrad<<<blocks, threads>>>(input_x_data,
                                 input_y_data,
                                 input_out_grad_data,
                                 output_x_grad_data,
                                 output_y_grad_data,
                                 full_strides[dim],
                                 numel / 3,
                                 index_calculator);
}
}  // namespace phi

PD_REGISTER_KERNEL(cross_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::CrossGradKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
