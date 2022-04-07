/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/fill_diagonal_tensor_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using CUDADeviceContext = paddle::platform::CUDADeviceContext;

template <typename T>
__global__ void fill_diagonal_tensor_kernel(int64_t size, T *out_data,
                                            const T *fill_data,
                                            int64_t *strides, int64_t *matdim,
                                            int64_t offset, int64_t fill_dims0,
                                            int64_t fill_dims1) {
  int64_t i = blockIdx.x;
  auto sumoff = matdim[i] + offset;
  for (int64_t j = threadIdx.x; j < fill_dims1; j += blockDim.x) {
    auto fill_index = j * (strides[1] + strides[0]) + sumoff;
    if (fill_index < size) {
      out_data[fill_index] = fill_data[i * fill_dims1 + j];
    }
  }
}

template <typename T>
__global__ void fill_grad_kernel(int64_t size, T *out_data, int64_t *strides,
                                 int64_t *matdim, int64_t offset,
                                 int64_t fill_dims0, int64_t fill_dims1) {
  int64_t i = blockIdx.x;
  auto sumoff = matdim[i] + offset;
  for (int64_t j = threadIdx.x; j < fill_dims1; j += blockDim.x) {
    auto fill_index = j * (strides[1] + strides[0]) + sumoff;
    if (fill_index < size) {
      out_data[fill_index] = T(0);
    }
  }
}

template <typename T>
class FillDiagonalTensorCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
#ifdef __HIPCC__
    const int64_t kMaxBlockDim = 256;
#else
    const int64_t kMaxBlockDim = 512;
#endif
    auto *out = ctx.Output<framework::Tensor>("Out");
    auto *srctensor = ctx.Input<framework::Tensor>("Y");
    auto dim1 = ctx.Attr<int>("dim1");
    auto dim2 = ctx.Attr<int>("dim2");
    auto offset = ctx.Attr<int64_t>("offset");

    auto *xin = ctx.Input<framework::Tensor>("X");
    framework::TensorCopy(*xin, ctx.GetPlace(), out);

    T *out_data = out->mutable_data<T>(ctx.GetPlace());
    const T *fill_data = srctensor->data<T>();

    auto out_dims = out->dims();
    auto matdims = srctensor->dims();
    auto fill_dims = phi::flatten_to_2d(matdims, matdims.size() - 1);

    int64_t new_dims[2];
    std::vector<int64_t> memory_block;
    memory_block.resize(2 + fill_dims[0]);
    int64_t *strides = &(memory_block[0]);
    int64_t *matdim = &(memory_block[2]);
    CalMatDims(out_dims, dim1, dim2, &offset, new_dims, strides, matdim);
    PADDLE_ENFORCE_EQ(
        new_dims[0], fill_dims[0],
        platform::errors::InvalidArgument("The dims should be %d x %d, but get "
                                          "%d x %d in fill tensor Y",
                                          new_dims[0], new_dims[1],
                                          fill_dims[0], fill_dims[1]));
    PADDLE_ENFORCE_EQ(
        new_dims[1], fill_dims[1],
        platform::errors::InvalidArgument("The dims should be %d x %d, but get "
                                          "%d x %d in fill tensor Y",
                                          new_dims[0], new_dims[1],
                                          fill_dims[0], fill_dims[1]));

    auto size = out->numel();

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto stream = dev_ctx.stream();
    Tensor tensor_tmp;
    int64_t *memory_block_cu =
        tensor_tmp.mutable_data<int64_t>({2 + fill_dims[0]}, ctx.GetPlace());
    const auto gpu_place = ctx.GetPlace();
    memory::Copy(gpu_place, memory_block_cu, platform::CPUPlace(),
                 memory_block.data(), sizeof(int64_t) * (2 + fill_dims[0]),
                 stream);

    int64_t *strides_cu = &memory_block_cu[0], *matdim_cu = &memory_block_cu[2];

    auto kGridDim = new_dims[0];
    auto kBlockDim = std::min(int64_t(new_dims[1]), kMaxBlockDim);
    fill_diagonal_tensor_kernel<T><<<kGridDim, kBlockDim, 0, stream>>>(
        size, out_data, fill_data, strides_cu, matdim_cu, offset, fill_dims[0],
        fill_dims[1]);
  }
};

template <typename T>
class FillDiagonalTensorGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
#ifdef __HIPCC__
    const int64_t kMaxBlockDim = 256;
#else
    const int64_t kMaxBlockDim = 512;
#endif
    auto *dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto *dout = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));

    auto dim1 = ctx.Attr<int>("dim1");
    auto dim2 = ctx.Attr<int>("dim2");
    auto offset = ctx.Attr<int64_t>("offset");
    auto matrows = 1;

    if (dx) {
      auto *data = dx->mutable_data<T>(ctx.GetPlace());
      auto dx_dims = dx->dims();
      framework::TensorCopy(*dout, ctx.GetPlace(), dx);

      for (int i = 0; i < dx_dims.size(); i++) {
        if (i != dim1 && i != dim2) {
          matrows *= dx_dims[i];
        }
      }

      int64_t new_dims[2];
      std::vector<int64_t> memory_block;
      memory_block.resize(2 + matrows);
      int64_t *strides = &memory_block[0];
      int64_t *matdim = &memory_block[2];
      CalMatDims(dx_dims, dim1, dim2, &offset, new_dims, strides, matdim);

      auto size = dx->numel();

      auto &dev_ctx =
          ctx.template device_context<platform::CUDADeviceContext>();
      auto stream = dev_ctx.stream();
      Tensor tensor_tmp;
      int64_t *memory_block_cu =
          tensor_tmp.mutable_data<int64_t>({2 + matrows}, ctx.GetPlace());
      const auto gpu_place = ctx.GetPlace();
      memory::Copy(gpu_place, memory_block_cu, platform::CPUPlace(),
                   memory_block.data(), sizeof(int64_t) * (2 + matrows),
                   stream);

      int64_t *strides_cu = &memory_block_cu[0],
              *matdim_cu = &memory_block_cu[2];

      auto kGridDim = new_dims[0];
      auto kBlockDim = std::min(int64_t(new_dims[1]), kMaxBlockDim);
      fill_grad_kernel<T><<<kGridDim, kBlockDim, 0, stream>>>(
          size, data, strides_cu, matdim_cu, offset, new_dims[0], new_dims[1]);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(
    fill_diagonal_tensor, ops::FillDiagonalTensorCUDAKernel<float>,
    ops::FillDiagonalTensorCUDAKernel<double>,
    ops::FillDiagonalTensorCUDAKernel<plat::float16>,
    ops::FillDiagonalTensorCUDAKernel<int>,
    ops::FillDiagonalTensorCUDAKernel<int64_t>,
    ops::FillDiagonalTensorCUDAKernel<int8_t>,
    ops::FillDiagonalTensorCUDAKernel<uint8_t>,
    ops::FillDiagonalTensorCUDAKernel<paddle::platform::complex<float>>,
    ops::FillDiagonalTensorCUDAKernel<paddle::platform::complex<double>>,
    ops::FillDiagonalTensorCUDAKernel<bool>);

REGISTER_OP_CUDA_KERNEL(
    fill_diagonal_tensor_grad, ops::FillDiagonalTensorGradCUDAKernel<float>,
    ops::FillDiagonalTensorGradCUDAKernel<double>,
    ops::FillDiagonalTensorGradCUDAKernel<int>,
    ops::FillDiagonalTensorGradCUDAKernel<int64_t>,
    ops::FillDiagonalTensorGradCUDAKernel<plat::float16>,
    ops::FillDiagonalTensorGradCUDAKernel<int8_t>,
    ops::FillDiagonalTensorGradCUDAKernel<uint8_t>,
    ops::FillDiagonalTensorGradCUDAKernel<paddle::platform::complex<float>>,
    ops::FillDiagonalTensorGradCUDAKernel<paddle::platform::complex<double>>,
    ops::FillDiagonalTensorGradCUDAKernel<bool>);
