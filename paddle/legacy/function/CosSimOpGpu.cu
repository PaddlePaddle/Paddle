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

#include "CosSimOp.h"
#include "hl_base.h"
#include "hl_device_functions.cuh"

namespace paddle {

template <int block_size>
__global__ void KeCosSim(real* output,
                         const real* input1,
                         const real* input2,
                         int width,
                         int input1_height,
                         int input2_height,
                         real scale) {
  const int ty = blockIdx.y;
  int tid = threadIdx.x;

  __shared__ real xx[block_size];
  __shared__ real yy[block_size];
  __shared__ real xy[block_size];

  xx[tid] = 0.0;
  yy[tid] = 0.0;
  xy[tid] = 0.0;
  __syncthreads();

  input1 += ty * width;
  if (input2_height > 1) {
    input2 += ty * width;
  }
  for (int index = tid; index < width; index += block_size) {
    real x = input1[index];
    real y = input2[index];
    xx[tid] += x * x;
    yy[tid] += y * y;
    xy[tid] += x * y;
  }
  __syncthreads();

  for (int s = block_size / 2; s > 0; s >>= 1) {
    if (tid < s) {
      xx[tid] += xx[tid + s];
      yy[tid] += yy[tid + s];
      xy[tid] += xy[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    output[ty] = scale * xy[0] / (sqrt(xx[0]) * sqrt(yy[0]));
  }
}

void hlCossim(real* output,
              const real* input1,
              const real* input2,
              size_t width,
              size_t input1_height,
              size_t input2_height,
              real scale) {
  CHECK_NOTNULL(output);
  CHECK_NOTNULL(input1);
  CHECK_NOTNULL(input2);
  const int block_size = 256;
  dim3 threads(block_size, 1);
  dim3 grid(1, input1_height);

  KeCosSim<block_size><<<grid, threads, 0, STREAM_DEFAULT>>>(
      output, input1, input2, width, input1_height, input2_height, scale);
  CHECK_SYNC("hlCossim failed");
}

template <>
void CosSimForward<DEVICE_TYPE_GPU>(GpuMatrix& out_mat,
                                    const GpuMatrix& in1_mat,
                                    const GpuMatrix& in2_mat,
                                    real scale) {
  CHECK(out_mat.getData() && in1_mat.getData() && in2_mat.getData());
  CHECK(in1_mat.useGpu_ == true && in2_mat.useGpu_ == true)
      << "Matrix type are not GPU";

  size_t dim = in1_mat.getWidth();
  real* out = out_mat.getData();
  const real* x = in1_mat.getData();
  const real* y = in2_mat.getData();
  hlCossim(out, x, y, dim, in1_mat.getHeight(), in2_mat.getHeight(), scale);
}

template <int block_size>
__global__ void KeCosSimDerivative(const real* grad,
                                   const real* output,
                                   const real* prev_out_x,
                                   const real* prev_out_y,
                                   real* prev_grad_x,
                                   real* prev_grad_y,
                                   size_t width,
                                   size_t input1_height,
                                   size_t input2_height,
                                   real scale) {
  const int ty = blockIdx.y;
  int tid = threadIdx.x;

  __shared__ real xx[block_size];
  __shared__ real yy[block_size];
  __shared__ real xy[block_size];

  xx[tid] = 0.0;
  yy[tid] = 0.0;
  xy[tid] = 0.0;
  __syncthreads();

  prev_out_x += ty * width;
  prev_grad_x += ty * width;
  if (input2_height > 1) {
    prev_out_y += ty * width;
    prev_grad_y += ty * width;
  }
  for (int index = tid; index < width; index += block_size) {
    real x = prev_out_x[index];
    real y = prev_out_y[index];
    xx[tid] += x * x;
    yy[tid] += y * y;
    xy[tid] += x * y;
  }
  __syncthreads();

  for (int s = block_size / 2; s > 0; s >>= 1) {
    if (tid < s) {
      xx[tid] += xx[tid + s];
      yy[tid] += yy[tid + s];
      xy[tid] += xy[tid + s];
    }
    __syncthreads();
  }
  if (xy[0] == 0) {
    real reciprocal = 1.0 / (sqrt(xx[0]) * sqrt(yy[0]));
    for (int index = tid; index < width; index += block_size) {
      prev_grad_x[index] += scale * grad[ty] * prev_out_y[index] * reciprocal;
      if (input2_height > 1) {
        prev_grad_y[index] += scale * grad[ty] * prev_out_x[index] * reciprocal;
      } else {
        paddle::paddleAtomicAdd(
            prev_grad_y + index,
            scale * grad[ty] * prev_out_x[index] * reciprocal);
      }
    }
  } else {
    real reciprocalXY = 1.0 / xy[0];
    real reciprocalSquareSumX = 1.0 / xx[0];
    real reciprocalSquareSumY = 1.0 / yy[0];
    for (int index = tid; index < width; index += block_size) {
      prev_grad_x[index] +=
          output[ty] * grad[ty] * (prev_out_y[index] * reciprocalXY -
                                   prev_out_x[index] * reciprocalSquareSumX);
      if (input2_height > 1) {
        prev_grad_y[index] +=
            output[ty] * grad[ty] * (prev_out_x[index] * reciprocalXY -
                                     prev_out_y[index] * reciprocalSquareSumY);
      } else {
        paddle::paddleAtomicAdd(
            prev_grad_y + index,
            output[ty] * grad[ty] * (prev_out_x[index] * reciprocalXY -
                                     prev_out_y[index] * reciprocalSquareSumY));
      }
    }
  }
}

void hlCossimDerivative(const real* grad,
                        const real* output,
                        const real* prev_out_x,
                        const real* prev_out_y,
                        real* prev_grad_x,
                        real* prev_grad_y,
                        size_t width,
                        size_t input1_height,
                        size_t input2_height,
                        real scale) {
  CHECK_NOTNULL(grad);
  CHECK_NOTNULL(output);
  CHECK_NOTNULL(prev_out_x);
  CHECK_NOTNULL(prev_out_y);
  CHECK_NOTNULL(prev_grad_x);
  CHECK_NOTNULL(prev_grad_y);
  const int block_size = 256;
  dim3 threads(block_size, 1);
  dim3 grid(1, input1_height);
  KeCosSimDerivative<block_size><<<grid, threads, 0, STREAM_DEFAULT>>>(
      grad,
      output,
      prev_out_x,
      prev_out_y,
      prev_grad_x,
      prev_grad_y,
      width,
      input1_height,
      input2_height,
      scale);
  CHECK_SYNC("hlCossimDerivate failed");
}

template <>
void CosSimBackward<DEVICE_TYPE_GPU>(const GpuMatrix& out_grad,
                                     const GpuMatrix& out_val,
                                     const GpuMatrix& in1_val,
                                     const GpuMatrix& in2_val,
                                     GpuMatrix& in1_grad,
                                     GpuMatrix& in2_grad,
                                     real scale) {
  CHECK(out_grad.getData() && out_val.getData() && in1_val.getData() &&
        in2_val.getData() && in1_grad.getData() && in2_grad.getData());
  CHECK(out_grad.useGpu_ && out_val.useGpu_ && in1_val.useGpu_ &&
        in2_val.useGpu_ && in1_grad.useGpu_ && in2_grad.useGpu_)
      << "Matrix types are not equally GPU";

  size_t dim = in1_val.getWidth();
  const real* grad = out_grad.getData();
  const real* out = out_val.getData();
  const real* prev_out_x = in1_val.getData();
  const real* prev_out_y = in2_val.getData();
  real* prev_grad_x = in1_grad.getData();
  real* prev_grad_y = in2_grad.getData();
  hlCossimDerivative(grad,
                     out,
                     prev_out_x,
                     prev_out_y,
                     prev_grad_x,
                     prev_grad_y,
                     dim,
                     in1_val.getHeight(),
                     in2_val.getHeight(),
                     scale);
}

}  // namespace paddle
