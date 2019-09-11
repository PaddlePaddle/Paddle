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
#include "paddle/fluid/operators/elementwise/elementwise_div_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.cu.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/platform/float16.h"

#define TILE_SIZE 512
namespace ops = paddle::operators;
namespace plat = paddle::platform;

namespace paddle {
namespace operators {

template <typename T>
class ElementwiseDivKernel<plat::CUDADeviceContext, T>
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
      SameDimsElemwiseDivCUDAKernel<T><<<
          gird_size, block_size, 0,
          ctx.template device_context<plat::CUDADeviceContext>().stream()>>>(
          x->data<T>(), y->data<T>(), out_data, size);
    } else {
      default_elementwise_div<plat::CUDADeviceContext, T>(ctx, x, y, z);
    }
  }
};

template <>
class ElementwiseDivKernel<plat::CUDADeviceContext, plat::float16>
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
      SameDimsElemwiseDivCUDAKernel<<<
          gird_size, block_size, 0,
          ctx.template device_context<plat::CUDADeviceContext>().stream()>>>(
          x2, y2, z2, size);
    } else {
      default_elementwise_div<plat::CUDADeviceContext, plat::float16>(ctx, x, y,
                                                                      z);
    }
  }
};

template <typename T>
static __global__ void SimpleElemwiseDivGradCUDAKernel(const T* x, const T* y,
                                                       const T* out,
                                                       const T* dout,
                                                       int64_t size, T* dx,
                                                       T* dy) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  while (col < size) {
    T o = dout[col];
    dx[col] = o / y[col];
    dy[col] = -o * out[col] / y[col];
    col += blockDim.x * gridDim.x;
  }
}

template <typename T>
class ElementwiseDivGradKernel<plat::CUDADeviceContext, T>
    : public ElemwiseGradKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ElemwiseGradKernel<T>::Compute(ctx);
    using Tensor = framework::Tensor;

    // auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* out = ctx.Input<Tensor>("Out");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    auto* x = dout;  // fake x, not used
    int axis = ctx.Attr<int>("axis");
    if (x->dims() == y->dims() && dx && dy) {
      dim3 block_size = dim3(TILE_SIZE, 1);
      auto size = x->numel();
      dim3 gird_size = dim3((size + TILE_SIZE - 1) / TILE_SIZE, 1);
      SimpleElemwiseDivGradCUDAKernel<T><<<
          gird_size, block_size, 0,
          ctx.template device_context<plat::CUDADeviceContext>().stream()>>>(
          x->data<T>(), y->data<T>(), out->data<T>(), dout->data<T>(), size,
          dx->mutable_data<T>(ctx.GetPlace()),
          dy->mutable_data<T>(ctx.GetPlace()));
      return;
    } else {
      ElemwiseGradCompute<plat::CUDADeviceContext, T, DivGradDX<T>,
                          DivGradDY<T>>(ctx, *x, *y, *out, *dout, axis, dx, dy,
                                        DivGradDX<T>(), DivGradDY<T>());
    }
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    elementwise_div,
    ops::ElementwiseDivKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ElementwiseDivKernel<paddle::platform::CUDADeviceContext,
                              paddle::platform::float16>,
    ops::ElementwiseDivKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ElementwiseDivKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ElementwiseDivKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_div_grad,
    ops::ElementwiseDivGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ElementwiseDivGradKernel<paddle::platform::CUDADeviceContext,
                                  paddle::platform::float16>,
    ops::ElementwiseDivGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ElementwiseDivGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ElementwiseDivGradKernel<paddle::platform::CUDADeviceContext,
                                  int64_t>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_div_grad_grad,
    ops::ElementwiseDivDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        float>,
    ops::ElementwiseDivDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        double>,
    ops::ElementwiseDivDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        int>,
    ops::ElementwiseDivDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        int64_t>);
