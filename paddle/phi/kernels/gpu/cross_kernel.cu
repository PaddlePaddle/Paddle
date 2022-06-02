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

#include "paddle/phi/kernels/cross_kernel.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/common_shape.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
  template <typename T>
  __global__ void Cross(const T* x, const T* y, T* out, int outer_loops, int slice_size) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(i < outer_loops && k < slice_size){
      auto pos0 = (3 * i + 0) * slice_size;
      auto pos1 = (3 * i + 1) * slice_size;
      auto pos2 = (3 * i + 2) * slice_size;

      out[pos0 + k] = x[pos1 + k] * y[pos2 + k] - x[pos2 + k] * y[pos1 + k];
      out[pos1 + k] = x[pos2 + k] * y[pos0 + k] - x[pos0 + k] * y[pos2 + k];
      out[pos2 + k] = x[pos0 + k] * y[pos1 + k] - x[pos1 + k] * y[pos0 + k];
    }
  }

  template <typename T, typename Context>
  void GPUCrossKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   int axis,
                   DenseTensor* out) {
    auto& input_x = x;
    auto& input_y = y;
    auto* output = out;
    int dim = axis;
    
    auto input_x_dims = input_x.dims();
    auto input_y_dims = input_y.dims();
    bool dims_match = phi::funcs::CheckDims(input_x_dims, input_y_dims);
    PADDLE_ENFORCE_EQ(
        dims_match,
        true,
        phi::errors::InvalidArgument("The 'shape' of Input(X) should be equal to "
                                     "the 'shape' of Input(Y). But received "
                                     "Input(X).dimensions = [%s], "
                                     "Input(Y).dimensions = [%s]",
                                     input_x_dims,
                                     input_x_dims));
  
    if (dim != DDim::kMaxRank) {
      PADDLE_ENFORCE_EQ(
          dim < input_x_dims.size() && dim >= (0 - input_x_dims.size()),
          true,
          phi::errors::OutOfRange(
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
          phi::errors::InvalidArgument(
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
      PADDLE_ENFORCE_EQ(dim == DDim::kMaxRank,
                        false,
                        phi::errors::InvalidArgument(
                            "There must be at least one dimension 'd' so that "
                            "Input(X/Y).dims()[d] is equal to 3. "
                            "But received: Input(X/Y).dims() == [%s].",
                            input_x_dims));
    }
    auto outer_loops = 1;
    for (auto i = 0; i < dim; i++) {
      outer_loops *= input_x_dims[i];
    }
    auto slice_size = 1;
    for (auto i = dim + 1; i < input_x_dims.size(); i++) {
      slice_size *= input_x_dims[i];
    }

    const auto* input_x_data = input_x.data<T>();
    const auto* input_y_data = input_y.data<T>();
    auto* out_data = dev_ctx.template Alloc<T>(out);

    dim3 threads(16, 16);
    dim3 grid((outer_loops + 16 - 1 ) / 16, (slice_size + 16 - 1 ) / 16);

    Cross<<<grid, threads>>>(input_x_data, input_y_data, out_data, outer_loops, slice_size);
  }
}  // namespace phi
    

PD_REGISTER_KERNEL(
    cross, GPU, ALL_LAYOUT, phi::GPUCrossKernel, float, double, int, int64_t) {}
