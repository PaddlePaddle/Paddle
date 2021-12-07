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

#include "paddle/fluid/operators/fill_diagonal_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using CUDADeviceContext = paddle::platform::CUDADeviceContext;

template <typename T>
__global__ void fill_constant_kernel(const int64_t featuresize, T* in_data,
                                     int64_t strides, int offset, T fillvar) {
  for (int64_t idx = blockIdx.x * featuresize + threadIdx.x;
       idx * strides + offset < (blockIdx.x + 1) * featuresize;
       idx += blockDim.x) {
    in_data[idx * strides + offset] = fillvar;
  }
}

template <typename T>
class FillIDiagonalCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#ifdef __HIPCC__
    const int64_t kMaxBlockDim = 256;
#else
    const int64_t kMaxBlockDim = 512;
#endif
    auto* out = ctx.Output<Tensor>("Out");
    auto offset = ctx.Attr<int>("offset");
    auto wrap = ctx.Attr<bool>("wrap");

    auto* xin = ctx.Input<framework::Tensor>("X");
    framework::TensorCopy(*xin, ctx.GetPlace(), out);

    T* out_data = out->mutable_data<T>(ctx.GetPlace());
    auto fill_val = static_cast<T>(ctx.template Attr<float>("value"));
    T temp_var = static_cast<T>(fill_val);

    auto size = out->numel();
    auto out_dims = out->dims();
    auto strides = CalStride(out_dims);

    // The wrap mode supported only the dims equels to 2; In wrap mode, the
    // value will be filled in cycles
    if (!wrap) {
      size = std::min(size, out_dims[1] * out_dims[1]);
    }

    int64_t kBlockDim = std::min(int64_t(size / strides), kMaxBlockDim);
    fill_constant_kernel<T><<<1, kBlockDim, 0>>>(size, out_data, strides,
                                                 offset, temp_var);
  }
};

template <typename T>
class FillIDiagonalGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#ifdef __HIPCC__
    const int64_t kMaxBlockDim = 256;
#else
    const int64_t kMaxBlockDim = 512;
#endif
    auto* dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* in_data = dx->mutable_data<T>(ctx.GetPlace());
    auto* dout = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto offset = ctx.Attr<int>("offset");
    auto wrap = ctx.Attr<bool>("wrap");

    framework::TensorCopy(*dout, ctx.GetPlace(), dx);

    auto size = dx->numel();
    auto out_dims = dx->dims();
    auto strides = CalStride(out_dims);

    auto wrapsize = std::min(size, out_dims[1] * out_dims[1]);
    // The wrap mode supported only the dims equels to 2; In wrap mode, the
    // value will be filled in cycles
    if (wrap) {
      wrapsize = size;
    }

    int64_t kBlockDim = std::min(int64_t(size), kMaxBlockDim);
    fill_constant_kernel<T><<<1, kBlockDim, 0>>>(wrapsize, in_data, strides,
                                                 offset, T(0));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(fill_diagonal, ops::FillIDiagonalCUDAKernel<float>,
                        ops::FillIDiagonalCUDAKernel<double>,
                        ops::FillIDiagonalCUDAKernel<plat::float16>,
                        ops::FillIDiagonalCUDAKernel<int>,
                        ops::FillIDiagonalCUDAKernel<int64_t>,
                        ops::FillIDiagonalCUDAKernel<bool>);

REGISTER_OP_CUDA_KERNEL(fill_diagonal_grad,
                        ops::FillIDiagonalGradCUDAKernel<float>,
                        ops::FillIDiagonalGradCUDAKernel<double>,
                        ops::FillIDiagonalGradCUDAKernel<int>,
                        ops::FillIDiagonalGradCUDAKernel<int64_t>,
                        ops::FillIDiagonalGradCUDAKernel<plat::float16>,
                        ops::FillIDiagonalGradCUDAKernel<bool>);
