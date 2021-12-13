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

#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast.cu.h"
#include "paddle/fluid/operators/elementwise/elementwise_sub_op.h"
#include "paddle/fluid/operators/reduce_ops/reduce_functor_op.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.cu.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/fluid/platform/float16.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

namespace paddle {
namespace operators {

template <typename T>
static __global__ void SimpleElemwiseSubGradCUDAKernel(const T* dout,
                                                       int64_t size, T* dx,
                                                       T* dy) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  while (col < size) {
    if (dx != nullptr) {
      dx[col] = dout[col];
    }
    dy[col] = -dout[col];
    col += blockDim.x * gridDim.x;
  }
}

template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_same<DeviceContext, platform::CUDADeviceContext>::value>::type
default_elementwise_sub_grad(const framework::ExecutionContext& ctx,
                             const framework::Tensor* x,
                             const framework::Tensor* y,
                             const framework::Tensor* out,
                             const framework::Tensor* dout,
                             framework::Tensor* dx, framework::Tensor* dy) {
  int axis = ctx.Attr<int>("axis");
  auto* dout_data = dout->data<T>();
  // dx
  if (dx != nullptr) {
    auto* dx_data = dx->mutable_data<T>(ctx.GetPlace());
    if (dx->dims() == dout->dims()) {
      if (dx_data != dout_data) {
        framework::TensorCopy(
            *dout, ctx.GetPlace(),
            ctx.template device_context<platform::DeviceContext>(), dx);
      }
    } else {
      // For inplace strategy, dx will be stored in addr of dout, which makes
      // the result of dy wrong.
      if (dx->IsSharedBufferWith(*dout)) {
        dx->clear();
        dx->mutable_data<T>(x->dims(), ctx.GetPlace());
      }
      std::vector<int> reduce_dims = GetReduceDim(x->dims(), out->dims(), axis);
      gpuStream_t stream = ctx.cuda_device_context().stream();
      TensorReduceFunctorImpl<T, T, CustomSum>(*dout, dx, reduce_dims, stream);
    }
  }
  // dy
  if (dy != nullptr) {
    auto* dy_data = dy->mutable_data<T>(ctx.GetPlace());
    if (dy->dims() == dout->dims()) {
      if (dy_data != dout_data) {
        dim3 block_size = dim3(ELEMENTWISE_BLOCK_SIZE, 1);
        auto size = dy->numel();
        dim3 grid_size = dim3(
            (size + ELEMENTWISE_BLOCK_SIZE - 1) / ELEMENTWISE_BLOCK_SIZE, 1);
        SimpleElemwiseSubGradCUDAKernel<T><<<
            grid_size, block_size, 0,
            ctx.template device_context<plat::CUDADeviceContext>().stream()>>>(
            dout->data<T>(), size, nullptr,
            dy->mutable_data<T>(ctx.GetPlace()));
      }
    } else {
      std::vector<int> reduce_dims = GetReduceDim(y->dims(), out->dims(), axis);
      gpuStream_t stream = ctx.cuda_device_context().stream();
      TensorReduceFunctorImpl<T, T, CustomSub>(*dout, dy, reduce_dims, stream);
    }
  }
}

template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_same<DeviceContext, plat::CUDADeviceContext>::value>::type
elementwise_sub_grad(const framework::ExecutionContext& ctx,
                     const framework::Tensor* x, const framework::Tensor* y,
                     const framework::Tensor* out,
                     const framework::Tensor* dout, framework::Tensor* dx,
                     framework::Tensor* dy) {
  dim3 block_size = dim3(ELEMENTWISE_BLOCK_SIZE, 1);
  auto size = x->numel();
  dim3 grid_size =
      dim3((size + ELEMENTWISE_BLOCK_SIZE - 1) / ELEMENTWISE_BLOCK_SIZE, 1);
  SimpleElemwiseSubGradCUDAKernel<
      T><<<grid_size, block_size, 0,
           ctx.template device_context<plat::CUDADeviceContext>().stream()>>>(
      dout->data<T>(), size, dx->mutable_data<T>(ctx.GetPlace()),
      dy->mutable_data<T>(ctx.GetPlace()));
}

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    elementwise_sub,
    ops::ElementwiseSubKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ElementwiseSubKernel<paddle::platform::CUDADeviceContext,
                              paddle::platform::float16>,
    ops::ElementwiseSubKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ElementwiseSubKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ElementwiseSubKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::ElementwiseSubKernel<paddle::platform::CUDADeviceContext,
                              paddle::platform::complex<float>>,
    ops::ElementwiseSubKernel<paddle::platform::CUDADeviceContext,
                              paddle::platform::complex<double>>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_sub_grad,
    ops::ElementwiseSubGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ElementwiseSubGradKernel<paddle::platform::CUDADeviceContext,
                                  paddle::platform::float16>,
    ops::ElementwiseSubGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ElementwiseSubGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ElementwiseSubGradKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::ElementwiseSubGradKernel<paddle::platform::CUDADeviceContext,
                                  paddle::platform::complex<float>>,
    ops::ElementwiseSubGradKernel<paddle::platform::CUDADeviceContext,
                                  paddle::platform::complex<double>>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_sub_grad_grad,
    ops::ElementwiseSubDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        float>,
    ops::ElementwiseSubDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        double>,
    ops::ElementwiseSubDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        int>,
    ops::ElementwiseSubDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        int64_t>,
    ops::ElementwiseSubDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        paddle::platform::complex<float>>,
    ops::ElementwiseSubDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        paddle::platform::complex<double>>);
