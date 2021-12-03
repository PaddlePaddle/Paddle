/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/conv_shift_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"

namespace paddle {
namespace operators {

using framework::Tensor;

namespace {

inline int DivUp(int x, int y) { return (x + y - 1) / y; }

// Some notes on the design:
//
// Each thread is responsible for computing a single output out[k, i].
// Thread blocks are based on tiles of x with height 1 in the batch dimension.
//
// This design is based on the typical use case where the filter
// y is fairly small. For large y, it would probably be more efficient
// to also tile across y.
template <typename T>
__global__ void ConvShiftForward(const T *x, const T *y, int x_width,
                                 int y_width, int y_half_width, int batch_size,
                                 T *out) {
  extern __shared__ T mem[];

  int tx = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tx;  // global x index
  int k = blockIdx.y;                    // batch index

  // Check if we are in a boundary block with fewer x's to process than
  // blockDim.x.
  int num_x =
      (blockIdx.x == gridDim.x - 1) ? (x_width % blockDim.x) : blockDim.x;

  T *sx = mem;
  T *sx_pad = &mem[num_x];
  T *sy = &mem[blockDim.x + y_width];

  // Collaboratively load y[k, :] and length-y padding of x into shared memory.
  int pad_start = blockIdx.x * blockDim.x + num_x + x_width - y_half_width;
  for (int j = tx; j < y_width; j += blockDim.x) {
    sy[j] = y[k * y_width + j];
    sx_pad[j] = x[k * x_width + (pad_start + j) % x_width];
  }

  // Load a cyclically shifted slice of x into shared memory.
  if (tx < num_x) {
    int load_i = (i - y_half_width + x_width) % x_width;
    sx[tx] = x[k * x_width + load_i];
  }
  __syncthreads();

  if (tx < num_x) {
    // Compute dot product of sx[tx:tx + y_width] and sy.
    T sum = 0;
    for (int j = 0; j < y_width; ++j) {
      sum += sx[tx + j] * sy[j];
    }

    // Save to out[k, i].
    out[k * x_width + i] = sum;
  }
}

// Compute x gradient - initial naive implementation with atomic add.
template <typename T>
__global__ void ConvShiftGradX(const T *dout, const T *y, int x_width,
                               int y_width, int y_half_width, int batch_size,
                               T *dx) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // x index
  int j = blockIdx.y;                             // y index
  int k = blockIdx.z;                             // batch index

  if (i < x_width) {
    int index = (i + j - y_half_width + x_width) % x_width;
    atomicAdd(&dx[k * x_width + index],
              dout[k * x_width + i] * y[k * y_width + j]);
  }
}

// Compute y gradient - initial naive implementation with atomic add.
template <typename T>
__global__ void ConvShiftDy(const T *x, const T *dout, int x_width, int y_width,
                            int y_half_width, int batch_size, T *dy) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // x index
  int j = blockIdx.y;                             // y index
  int k = blockIdx.z;                             // batch index

  if (i < x_width) {
    int index = (i + j - y_half_width + x_width) % x_width;
    atomicAdd(&dy[k * y_width + j],
              x[k * x_width + index] * dout[k * x_width + i]);
  }
}
}  // namespace

template <typename T>
class ConvShiftKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const Tensor *X = context.Input<Tensor>("X");
    const Tensor *Y = context.Input<Tensor>("Y");
    Tensor *Out = context.Output<Tensor>("Out");
    const T *x_data = X->data<T>();
    const T *y_data = Y->data<T>();
    T *out_data = Out->mutable_data<T>(context.GetPlace());

    int batch_size = X->dims()[0];
    int x_width = X->dims()[1];
    int y_width = Y->dims()[1];
    int y_half_width = (y_width - 1) / 2;

    const int x_per_block = 256;
    int num_x_blocks = DivUp(x_width, x_per_block);
    int mem_per_block = (x_per_block + 2 * y_width) * sizeof(T);

    dim3 grid_dim(num_x_blocks, batch_size);

    auto stream =
        context.template device_context<platform::CUDADeviceContext>().stream();

    ConvShiftForward<T><<<grid_dim, x_per_block, mem_per_block, stream>>>(
        x_data, y_data, x_width, y_width, y_half_width, batch_size, out_data);
  }
};

template <typename T>
class ConvShiftGradKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const Tensor *X = context.Input<Tensor>("X");
    const Tensor *Y = context.Input<Tensor>("Y");
    const Tensor *dOut = context.Input<Tensor>(framework::GradVarName("Out"));
    const T *x_data = X->data<T>();
    const T *y_data = Y->data<T>();
    const T *dout_data = dOut->data<T>();

    Tensor *dX = context.Output<Tensor>(framework::GradVarName("X"));
    Tensor *dY = context.Output<Tensor>(framework::GradVarName("Y"));

    int batch_size = X->dims()[0];
    int x_width = X->dims()[1];
    int y_width = Y->dims()[1];
    int y_half_width = (y_width - 1) / 2;

    auto &device_ctx =
        context.template device_context<platform::CUDADeviceContext>();
    math::SetConstant<platform::CUDADeviceContext, T> zero;

    const int x_per_block = 256;
    int num_x_blocks = DivUp(x_width, x_per_block);
    dim3 grid_dim(num_x_blocks, y_width, batch_size);

    if (dX) {
      T *dx_data = dX->mutable_data<T>(context.GetPlace());
      zero(device_ctx, dX, static_cast<T>(0.0));
      ConvShiftGradX<T><<<grid_dim, x_per_block, 0, device_ctx.stream()>>>(
          dout_data, y_data, x_width, y_width, y_half_width, batch_size,
          dx_data);
    }
    if (dY) {
      T *dy_data = dY->mutable_data<T>(context.GetPlace());
      zero(device_ctx, dY, static_cast<T>(0.0));
      ConvShiftDy<T><<<grid_dim, x_per_block, 0, device_ctx.stream()>>>(
          x_data, dout_data, x_width, y_width, y_half_width, batch_size,
          dy_data);
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    conv_shift,
    ops::ConvShiftKernel<paddle::platform::CUDADeviceContext, float>);
REGISTER_OP_CUDA_KERNEL(
    conv_shift_grad,
    ops::ConvShiftGradKernel<paddle::platform::CUDADeviceContext, float>);
