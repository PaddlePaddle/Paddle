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

#include "paddle/fluid/operators/elementwise/elementwise_add_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.cu.h"
#include "paddle/fluid/operators/math.h"
#include "paddle/fluid/platform/float16.h"
// #include <cuda_fp16.h>

#define TILE_SIZE 512
namespace ops = paddle::operators;
namespace plat = paddle::platform;

namespace paddle {
namespace operators {

template <typename T>
class ElementwiseAddKernel<plat::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using Tensor = framework::Tensor;
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* z = ctx.Output<Tensor>("Out");
    if (x->dims() == y->dims()) {
      z->mutable_data<T>(ctx.GetPlace());
      auto* out_data = z->data<T>();
      dim3 block_size = dim3(TILE_SIZE, 1);
      auto size = x->numel();
      dim3 gird_size = dim3((size + TILE_SIZE - 1) / TILE_SIZE, 1);
      SameDimsElemwiseAddCUDAKernel<T><<<
          gird_size, block_size, 0,
          ctx.template device_context<plat::CUDADeviceContext>().stream()>>>(
          x->data<T>(), y->data<T>(), out_data, size);
    } else {
      default_elementwise_add<plat::CUDADeviceContext, T>(ctx, x, y, z);
    }
  }
};

template <>
class ElementwiseAddKernel<plat::CUDADeviceContext, plat::float16>
    : public framework::OpKernel<plat::float16> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using Tensor = framework::Tensor;
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* z = ctx.Output<Tensor>("Out");
    if (x->dims() == y->dims()) {
      z->mutable_data<plat::float16>(ctx.GetPlace());
      auto* out_data = z->data<plat::float16>();
      auto size = x->numel();
      dim3 gird_size = dim3((size / 2 + TILE_SIZE - 1) / TILE_SIZE, 1);
      dim3 block_size = dim3(TILE_SIZE, 1);
      const half* x2 = reinterpret_cast<const half*>(x->data<plat::float16>());
      const half* y2 = reinterpret_cast<const half*>(y->data<plat::float16>());
      half* z2 = reinterpret_cast<half*>(out_data);
      SameDimsElemwiseAddCUDAKernel<<<
          gird_size, block_size, 0,
          ctx.template device_context<plat::CUDADeviceContext>().stream()>>>(
          x2, y2, z2, size);
    } else {
      default_elementwise_add<plat::CUDADeviceContext, plat::float16>(ctx, x, y,
                                                                      z);
    }
  }
};

template <typename T>
static __global__ void SimpleElemwiseAddGradCUDAKernel(const T* dout,
                                                       int64_t size, T* dx,
                                                       T* dy) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  while (col < size) {
    dx[col] = dout[col];
    dy[col] = dout[col];
    col += blockDim.x * gridDim.x;
  }
}

template <typename T>
class ElementwiseAddGradKernel<plat::CUDADeviceContext, T>
    : public ElemwiseGradKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ElemwiseGradKernel<T>::Compute(ctx);
    using Tensor = framework::Tensor;

    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    // skip out, x, y
    auto* out = dout;
    auto *x = dout, *y = dout;

    if (platform::is_gpu_place(ctx.GetPlace()) && dx != nullptr &&
        dy != nullptr && (dx->dims() == dy->dims())) {
      dim3 block_size = dim3(TILE_SIZE, 1);
      auto size = x->numel();
      dim3 gird_size = dim3((size + TILE_SIZE - 1) / TILE_SIZE, 1);
      SimpleElemwiseAddGradCUDAKernel<T><<<
          gird_size, block_size, 0,
          ctx.template device_context<plat::CUDADeviceContext>().stream()>>>(
          dout->data<T>(), size, dx->mutable_data<T>(ctx.GetPlace()),
          dy->mutable_data<T>(ctx.GetPlace()));
    } else {
      default_elementwise_add_grad<plat::CUDADeviceContext, T>(ctx, x, y, out,
                                                               dout, dx, dy);
    }
  }
};

}  // namespace operators
}  // namespace paddle
REGISTER_OP_CUDA_KERNEL(
    elementwise_add, ops::ElementwiseAddKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, int64_t>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_add_grad,
    ops::ElementwiseAddGradKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseAddGradKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseAddGradKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseAddGradKernel<plat::CUDADeviceContext, int64_t>,
    ops::ElementwiseAddGradKernel<plat::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_add_grad_grad,
    ops::ElementwiseAddDoubleGradKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseAddDoubleGradKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseAddDoubleGradKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseAddDoubleGradKernel<plat::CUDADeviceContext, int64_t>);
