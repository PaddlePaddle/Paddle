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

#include "paddle/fluid/operators/elementwise/elementwise_mul_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast.cu.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.cu.h"
#include "paddle/fluid/platform/complex128.h"
#include "paddle/fluid/platform/complex64.h"
#include "paddle/fluid/platform/float16.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

namespace paddle {
namespace operators {

template <typename T>
struct CudaMulFunctor {
  inline HOSTDEVICE T operator()(const T* args) const {
    return args[0] * args[1];
  }
};

template <typename T>
class ElementwiseMulKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto x_var = ctx.InputVar("X");
    PADDLE_ENFORCE_NOT_NULL(
        x_var, platform::errors::InvalidArgument(
                   "Cannot get input Variable X, Variable name = %s.",
                   ctx.InputName("X")));

    framework::Tensor x;
    framework::Tensor* z;
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    FindOutMulTensors<T>(ctx, x_var, &x, y, z);

    std::vector<const framework::Tensor*> ins = {&x, y};
    std::vector<framework::Tensor*> outs = {z};
    LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
        ctx, ins, &outs, CudaMulFunctor<T>());
  }
};

template <typename T>
static __global__ void SimpleElemwiseMulGradCUDAKernel(const T* x, const T* y,
                                                       const T* out,
                                                       const T* dout,
                                                       int64_t size, T* dx,
                                                       T* dy) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  while (col < size) {
    T o = dout[col];
    dx[col] = y[col] * o;
    dy[col] = x[col] * o;
    col += blockDim.x * gridDim.x;
  }
}

template <>
__global__ void SimpleElemwiseMulGradCUDAKernel<plat::complex64>(
    const plat::complex64* x, const plat::complex64* y,
    const plat::complex64* out, const plat::complex64* dout, int64_t size,
    plat::complex64* dx, plat::complex64* dy) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  while (col < size) {
    plat::complex64 o = dout[col];
    dx[col] = plat::complex64(y[col].real, -y[col].imag) * o;
    dy[col] = plat::complex64(x[col].real, -x[col].imag) * o;
    col += blockDim.x * gridDim.x;
  }
}

template <>
__global__ void SimpleElemwiseMulGradCUDAKernel<plat::complex128>(
    const plat::complex128* x, const plat::complex128* y,
    const plat::complex128* out, const plat::complex128* dout, int64_t size,
    plat::complex128* dx, plat::complex128* dy) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  while (col < size) {
    plat::complex128 o = dout[col];
    dx[col] = plat::complex128(y[col].real, -y[col].imag) * o;
    dy[col] = plat::complex128(x[col].real, -x[col].imag) * o;
    col += blockDim.x * gridDim.x;
  }
}

template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_same<DeviceContext, plat::CUDADeviceContext>::value>::type
elementwise_mul_grad(const framework::ExecutionContext& ctx,
                     const framework::Tensor* x, const framework::Tensor* y,
                     const framework::Tensor* out,
                     const framework::Tensor* dout, framework::Tensor* dx,
                     framework::Tensor* dy) {
  dim3 block_size = dim3(PADDLE_CUDA_THREAD_SIZE, 1);
  auto size = x->numel();
  dim3 grid_size =
      dim3((size + PADDLE_CUDA_THREAD_SIZE - 1) / PADDLE_CUDA_THREAD_SIZE, 1);
  SimpleElemwiseMulGradCUDAKernel<
      T><<<grid_size, block_size, 0,
           ctx.template device_context<plat::CUDADeviceContext>().stream()>>>(
      x->data<T>(), y->data<T>(), out->data<T>(), dout->data<T>(), size,
      dx->mutable_data<T>(ctx.GetPlace()), dy->mutable_data<T>(ctx.GetPlace()));
}

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    elementwise_mul, ops::ElementwiseMulKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseMulKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseMulKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseMulKernel<plat::CUDADeviceContext, int64_t>,
    ops::ElementwiseMulKernel<plat::CUDADeviceContext, plat::float16>,
    ops::ElementwiseMulKernel<plat::CUDADeviceContext, plat::complex64>,
    ops::ElementwiseMulKernel<plat::CUDADeviceContext, plat::complex128>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_mul_grad,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext, int64_t>,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext, plat::float16>,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext, plat::complex64>,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext, plat::complex128>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_mul_grad_grad,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext, int64_t>,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext, plat::float16>,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext,
                                        plat::complex64>,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext,
                                        plat::complex128>);
