// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/extension.h"

#define CHECK_GPU_INPUT(x) \
  PD_CHECK(x.place() == paddle::PlaceType::kGPU, #x " must be a GPU Tensor.")

template <typename data_t>
__global__ void relu_cuda_forward_kernel(const data_t* x,
                                         data_t* y,
                                         const int num) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = gid; i < num; i += blockDim.x * gridDim.x) {
    y[i] = x[i] > static_cast<data_t>(0.) ? x[i] : static_cast<data_t>(0.);
  }
}

template <typename data_t>
__global__ void relu_cuda_backward_kernel(const data_t* dy,
                                          const data_t* y,
                                          data_t* dx,
                                          const int num) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = gid; i < num; i += blockDim.x * gridDim.x) {
    dx[i] = dy[i] * (y[i] > static_cast<data_t>(0.) ? static_cast<data_t>(1.)
                                                    : static_cast<data_t>(0.));
  }
}

template <typename data_t>
__global__ void relu_cuda_double_backward_kernel(const data_t* out_data,
                                                 const data_t* ddx_data,
                                                 data_t* ddout_data,
                                                 int64_t num) {
  int64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int64_t i = num; i < num; i += blockDim.x * gridDim.x) {
    ddout_data[i] = ddx_data[i] * (out_data[i] > static_cast<data_t>(0.)
                                       ? static_cast<data_t>(1.)
                                       : static_cast<data_t>(0.));
  }
}

std::vector<paddle::Tensor> relu_cuda_forward(const paddle::Tensor& x) {
  CHECK_GPU_INPUT(x);
  auto out = paddle::empty(x.shape(), x.dtype(), x.place());

  int numel = x.size();
  int block = 512;
  int grid = (numel + block - 1) / block;
  PD_DISPATCH_FLOATING_AND_HALF_TYPES(
      x.type(), "relu_cuda_forward_kernel", ([&] {
        relu_cuda_forward_kernel<data_t><<<grid, block, 0, x.stream()>>>(
            x.data<data_t>(), out.mutable_data<data_t>(x.place()), numel);
      }));

  return {out};
}

std::vector<paddle::Tensor> relu_cuda_backward(const paddle::Tensor& x,
                                               const paddle::Tensor& out,
                                               const paddle::Tensor& grad_out) {
  CHECK_GPU_INPUT(x);
  CHECK_GPU_INPUT(out);
  CHECK_GPU_INPUT(grad_out);
  auto grad_x = paddle::empty(x.shape(), x.dtype(), x.place());

  int numel = out.size();
  int block = 512;
  int grid = (numel + block - 1) / block;
  PD_DISPATCH_FLOATING_AND_HALF_TYPES(
      out.type(), "relu_cuda_backward_kernel", ([&] {
        relu_cuda_backward_kernel<data_t><<<grid, block, 0, x.stream()>>>(
            grad_out.data<data_t>(),
            out.data<data_t>(),
            grad_x.mutable_data<data_t>(x.place()),
            numel);
      }));

  return {grad_x};
}

std::vector<paddle::Tensor> relu_cuda_double_backward(
    const paddle::Tensor& out, const paddle::Tensor& ddx) {
  CHECK_GPU_INPUT(out);
  CHECK_GPU_INPUT(ddx);
  auto ddout = paddle::empty(out.shape(), out.dtype(), out.place());

  int64_t numel = out.size();
  int64_t block = 512;
  int64_t grid = (numel + block - 1) / block;
  PD_DISPATCH_FLOATING_AND_HALF_TYPES(
      out.type(), "relu_cuda_double_backward_kernel", ([&] {
        relu_cuda_double_backward_kernel<
            data_t><<<grid, block, 0, out.stream()>>>(
            out.data<data_t>(),
            ddx.data<data_t>(),
            ddout.mutable_data<data_t>(out.place()),
            numel);
      }));

  std::cout << "Debug info: run relu gpu double backward success." << std::endl;

  return {ddout};
}

std::vector<paddle::Tensor> relu_cuda_backward_without_x(
    const paddle::Tensor& out, const paddle::Tensor& grad_out) {
  auto grad_x = paddle::empty(out.shape(), out.dtype(), out.place());

  int numel = out.size();
  int block = 512;
  int grid = (numel + block - 1) / block;
  PD_DISPATCH_FLOATING_AND_HALF_TYPES(
      out.type(), "relu_cuda_backward_kernel", ([&] {
        relu_cuda_backward_kernel<data_t><<<grid, block, 0, out.stream()>>>(
            grad_out.data<data_t>(),
            out.data<data_t>(),
            grad_x.mutable_data<data_t>(out.place()),
            numel);
      }));

  return {grad_x};
}

void relu_cuda_forward_out(const paddle::Tensor& x, paddle::Tensor* out) {
  int numel = x.size();
  int block = 512;
  int grid = (numel + block - 1) / block;
  out->reshape(x.shape());
  PD_DISPATCH_FLOATING_AND_HALF_TYPES(
      x.type(), "relu_cuda_forward_kernel", ([&] {
        relu_cuda_forward_kernel<data_t><<<grid, block, 0, x.stream()>>>(
            x.data<data_t>(), out->mutable_data<data_t>(x.place()), numel);
      }));
}

void relu_cuda_backward_out(const paddle::Tensor& x,
                            const paddle::Tensor& out,
                            const paddle::Tensor& grad_out,
                            paddle::Tensor* grad_x) {
  int numel = out.size();
  int block = 512;
  int grid = (numel + block - 1) / block;
  grad_x->reshape(x.shape());
  PD_DISPATCH_FLOATING_AND_HALF_TYPES(
      out.type(), "relu_cuda_backward_kernel", ([&] {
        relu_cuda_backward_kernel<data_t><<<grid, block, 0, x.stream()>>>(
            grad_out.data<data_t>(),
            out.data<data_t>(),
            grad_x->mutable_data<data_t>(x.place()),
            numel);
      }));
}
