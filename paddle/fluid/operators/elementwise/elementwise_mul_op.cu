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
#include "paddle/fluid/platform/complex.h"
#include "paddle/fluid/platform/float16.h"

// only can include the headers in paddle/top/api dirs
#include "paddle/pten/api/lib/utils/tensor_utils.h"
#include "paddle/pten/include/core.h"
#include "paddle/pten/include/math.h"
namespace ops = paddle::operators;
namespace plat = paddle::platform;

namespace paddle {
namespace operators {

template <typename T>
class ElementwiseMulKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto x_var = ctx.InputVar("X");
    PADDLE_ENFORCE_EQ(x_var != nullptr, true,
                      platform::errors::InvalidArgument(
                          "Cannot get input Variable X, Variable name = %s.",
                          ctx.InputName("X")));
    const auto& cuda_ctx =
        ctx.template device_context<platform::CUDADeviceContext>();
    if (x_var->IsType<framework::SelectedRows>()) {
      framework::Tensor x_for_selectedrows;
      std::vector<const framework::Tensor*> ins;
      std::vector<framework::Tensor*> outs;
      int axis =
          PackTensorsIntoVector<T>(ctx, &ins, &outs, &x_for_selectedrows);
      LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
          cuda_ctx, ins, &outs, axis, MulFunctor<T>());
    } else if (x_var->IsType<framework::LoDTensor>()) {
      auto* x_lod = ctx.Input<framework::LoDTensor>("X");
      auto* y_lod = ctx.Input<framework::LoDTensor>("Y");
      auto* z_lod = ctx.Output<framework::LoDTensor>("Out");
      z_lod->mutable_data<T>(ctx.GetPlace());

      int axis = ctx.Attr<int>("axis");
      auto pt_x = paddle::experimental::MakePtenDenseTensor(*x_lod);
      auto pt_y = paddle::experimental::MakePtenDenseTensor(*y_lod);
      auto pt_z = paddle::experimental::MakePtenDenseTensor(*z_lod);
      pten::Multiply<T>(cuda_ctx, *pt_x.get(), *pt_y.get(), axis, pt_z.get());
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "X's type[%s] is not supported by elementwise_op. X's type should be "
          "LoDTensor or SelectedRows.",
          framework::ToTypeName(x_var->Type())));
    }
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
__global__ void SimpleElemwiseMulGradCUDAKernel<plat::complex<float>>(
    const plat::complex<float>* x, const plat::complex<float>* y,
    const plat::complex<float>* out, const plat::complex<float>* dout,
    int64_t size, plat::complex<float>* dx, plat::complex<float>* dy) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  while (col < size) {
    plat::complex<float> o = dout[col];
    dx[col] = plat::complex<float>(y[col].real, -y[col].imag) * o;
    dy[col] = plat::complex<float>(x[col].real, -x[col].imag) * o;
    col += blockDim.x * gridDim.x;
  }
}

template <>
__global__ void SimpleElemwiseMulGradCUDAKernel<plat::complex<double>>(
    const plat::complex<double>* x, const plat::complex<double>* y,
    const plat::complex<double>* out, const plat::complex<double>* dout,
    int64_t size, plat::complex<double>* dx, plat::complex<double>* dy) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  while (col < size) {
    plat::complex<double> o = dout[col];
    dx[col] = plat::complex<double>(y[col].real, -y[col].imag) * o;
    dy[col] = plat::complex<double>(x[col].real, -x[col].imag) * o;
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
  dim3 block_size = dim3(ELEMENTWISE_BLOCK_SIZE, 1);
  auto size = x->numel();
  dim3 grid_size =
      dim3((size + ELEMENTWISE_BLOCK_SIZE - 1) / ELEMENTWISE_BLOCK_SIZE, 1);
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
    ops::ElementwiseMulKernel<plat::CUDADeviceContext, bool>,
    ops::ElementwiseMulKernel<plat::CUDADeviceContext, plat::float16>,
    ops::ElementwiseMulKernel<plat::CUDADeviceContext, plat::complex<float>>,
    ops::ElementwiseMulKernel<plat::CUDADeviceContext, plat::complex<double>>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_mul_grad,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext, int64_t>,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext, bool>,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext, plat::float16>,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext,
                                  plat::complex<float>>,
    ops::ElementwiseMulGradKernel<plat::CUDADeviceContext,
                                  plat::complex<double>>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_mul_grad_grad,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext, int64_t>,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext, bool>,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext, plat::float16>,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext,
                                        plat::complex<float>>,
    ops::ElementwiseMulDoubleGradKernel<plat::CUDADeviceContext,
                                        plat::complex<double>>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_mul_triple_grad,
    ops::ElementwiseMulTripleGradKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseMulTripleGradKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseMulTripleGradKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseMulTripleGradKernel<plat::CUDADeviceContext, int64_t>,
    ops::ElementwiseMulTripleGradKernel<plat::CUDADeviceContext, bool>,
    ops::ElementwiseMulTripleGradKernel<plat::CUDADeviceContext, plat::float16>,
    ops::ElementwiseMulTripleGradKernel<plat::CUDADeviceContext,
                                        plat::complex<float>>,
    ops::ElementwiseMulTripleGradKernel<plat::CUDADeviceContext,
                                        plat::complex<double>>);
