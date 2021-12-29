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
#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast.cu.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.cu.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/fluid/platform/float16.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

namespace paddle {
namespace operators {

template <typename T>
static __global__ void SimpleElemwiseDivGradCUDAKernel(const T* x, const T* y,
                                                       const T* out,
                                                       const T* dout,
                                                       int64_t size, T* dx,
                                                       T* dy) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    T o = dout[i];
    dx[i] = o / y[i];
    dy[i] = -o * out[i] / y[i];
  }
}

template <>
__global__ void
SimpleElemwiseDivGradCUDAKernel<paddle::platform::complex<float>>(
    const paddle::platform::complex<float>* x,
    const paddle::platform::complex<float>* y,
    const paddle::platform::complex<float>* out,
    const paddle::platform::complex<float>* dout, int64_t size,
    paddle::platform::complex<float>* dx,
    paddle::platform::complex<float>* dy) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    paddle::platform::complex<float> o = dout[i];
    paddle::platform::complex<float> y_conj(y[i].real, -y[i].imag);
    paddle::platform::complex<float> out_div_y_conj((out[i] / y[i]).real,
                                                    -(out[i] / y[i]).imag);
    dx[i] = o / y_conj;
    dy[i] = -dout[i] * out_div_y_conj;
  }
}

template <>
__global__ void
SimpleElemwiseDivGradCUDAKernel<paddle::platform::complex<double>>(
    const paddle::platform::complex<double>* x,
    const paddle::platform::complex<double>* y,
    const paddle::platform::complex<double>* out,
    const paddle::platform::complex<double>* dout, int64_t size,
    paddle::platform::complex<double>* dx,
    paddle::platform::complex<double>* dy) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    paddle::platform::complex<double> o = dout[i];
    paddle::platform::complex<double> y_conj(y[i].real, -y[i].imag);
    paddle::platform::complex<double> out_div_y_conj((out[i] / y[i]).real,
                                                     -(out[i] / y[i]).imag);
    dx[i] = o / y_conj;
    dy[i] = -dout[i] * out_div_y_conj;
  }
}

template <typename T>
void reduce_functor(const framework::ExecutionContext& ctx,
                    const framework::Tensor* in, const framework::Tensor* out,
                    framework::Tensor* src, framework::Tensor* dst) {
  const auto& dev_ctx =
      ctx.template device_context<platform::CUDADeviceContext>();
  if (dst->dims() == out->dims()) {
    dst->ShareDataWith(*src);
    return;
  }
  int axis = ctx.Attr<int>("axis");
  std::vector<int> reduce_dims = GetReduceDim(in->dims(), out->dims(), axis);
  gpuStream_t stream = ctx.cuda_device_context().stream();
  TensorReduceFunctorImpl<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
      *src, dst, kps::IdentityFunctor<T>(), reduce_dims, stream);
}

template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_same<DeviceContext, platform::CUDADeviceContext>::value>::type
default_elementwise_div_grad(const framework::ExecutionContext& ctx,
                             const framework::Tensor* x,
                             const framework::Tensor* y,
                             const framework::Tensor* out,
                             const framework::Tensor* dout,
                             framework::Tensor* dx, framework::Tensor* dy) {
  int axis = ctx.Attr<int>("axis");
  auto* dout_data = dout->data<T>();
  dim3 block_size = dim3(ELEMENTWISE_BLOCK_SIZE, 1);
  const auto& dev_ctx =
      ctx.template device_context<platform::CUDADeviceContext>();
  framework::Tensor tmp_dx;
  tmp_dx.mutable_data<T>(dout->dims(), ctx.GetPlace());
  framework::Tensor tmp_dy;
  tmp_dy.mutable_data<T>(dout->dims(), ctx.GetPlace());
  if (dx != nullptr && dy != nullptr) {
    auto* dx_data = dx->mutable_data<T>(ctx.GetPlace());
    auto* dy_data = dy->mutable_data<T>(ctx.GetPlace());
    // For inplace strategy, dx will be stored in addr of dout, which makes
    // the result of dy wrong.
    if (dx->IsSharedBufferWith(*dout)) {
      dx->clear();
      dx->mutable_data<T>(x->dims(), ctx.GetPlace());
    }
    // dout.dims==out.dims
    std::vector<const framework::Tensor*> ins = {dout, out, y};
    std::vector<framework::Tensor*> outs = {&tmp_dx, &tmp_dy};
    auto functor = DivGradXYFunctor<T, T>();
    LaunchElementwiseCudaKernel<ElementwiseType::kTernary, T, T,
                                decltype(functor), 2>(dev_ctx, ins, &outs, axis,
                                                      functor);

    if (dx->dims() == dout->dims() && dy->dims() == dout->dims()) {
      dx->ShareDataWith(tmp_dx);
      dy->ShareDataWith(tmp_dy);
    } else {
      reduce_functor<T>(ctx, x, out, &tmp_dx, dx);
      reduce_functor<T>(ctx, y, out, &tmp_dy, dy);
    }
  } else if (dx != nullptr && dy == nullptr) {
    auto* dx_data = dx->mutable_data<T>(ctx.GetPlace());
    if (dx->IsSharedBufferWith(*dout)) {
      dx->clear();
      dx->mutable_data<T>(x->dims(), ctx.GetPlace());
    }
    std::vector<const framework::Tensor*> ins = {dout, y};
    std::vector<framework::Tensor*> outs = {&tmp_dx};
    LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
        dev_ctx, ins, &outs, axis, DivGradFunctor<T>());
    if (dx->dims() != dout->dims()) {
      reduce_functor<T>(ctx, x, out, &tmp_dx, dx);
    } else {
      dx->ShareDataWith(tmp_dx);
    }
  } else if (dy != nullptr && dx == nullptr) {
    auto* dy_data = dy->mutable_data<T>(ctx.GetPlace());
    std::vector<const framework::Tensor*> ins = {dout, out, y};
    std::vector<framework::Tensor*> outs = {&tmp_dy};
    LaunchElementwiseCudaKernel<ElementwiseType::kTernary, T, T>(
        dev_ctx, ins, &outs, axis, DivGradYFunctor<T>());
    if (dy->dims() != dout->dims()) {
      reduce_functor<T>(ctx, y, out, &tmp_dy, dy);
    } else {
      dy->ShareDataWith(tmp_dy);
    }
  }
}

template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_same<DeviceContext, plat::CUDADeviceContext>::value>::type
elementwise_div_grad(const framework::ExecutionContext& ctx,
                     const framework::Tensor* x, const framework::Tensor* y,
                     const framework::Tensor* out,
                     const framework::Tensor* dout, framework::Tensor* dx,
                     framework::Tensor* dy) {
  dim3 block_size = dim3(ELEMENTWISE_BLOCK_SIZE, 1);
  auto size = x->numel();
  dim3 grid_size =
      dim3((size + ELEMENTWISE_BLOCK_SIZE - 1) / ELEMENTWISE_BLOCK_SIZE, 1);
  SimpleElemwiseDivGradCUDAKernel<
      T><<<grid_size, block_size, 0,
           ctx.template device_context<plat::CUDADeviceContext>().stream()>>>(
      x->data<T>(), y->data<T>(), out->data<T>(), dout->data<T>(), size,
      dx->mutable_data<T>(ctx.GetPlace()), dy->mutable_data<T>(ctx.GetPlace()));
}

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    elementwise_div,
    ops::ElementwiseDivKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ElementwiseDivKernel<paddle::platform::CUDADeviceContext,
                              paddle::platform::float16>,
    ops::ElementwiseDivKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ElementwiseDivKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ElementwiseDivKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::ElementwiseDivKernel<paddle::platform::CUDADeviceContext,
                              paddle::platform::complex<float>>,
    ops::ElementwiseDivKernel<paddle::platform::CUDADeviceContext,
                              paddle::platform::complex<double>>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_div_grad,
    ops::ElementwiseDivGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ElementwiseDivGradKernel<paddle::platform::CUDADeviceContext,
                                  paddle::platform::float16>,
    ops::ElementwiseDivGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ElementwiseDivGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ElementwiseDivGradKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::ElementwiseDivGradKernel<paddle::platform::CUDADeviceContext,
                                  paddle::platform::complex<float>>,
    ops::ElementwiseDivGradKernel<paddle::platform::CUDADeviceContext,
                                  paddle::platform::complex<double>>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_div_grad_grad,
    ops::ElementwiseDivDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        float>,
    ops::ElementwiseDivDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        paddle::platform::float16>,
    ops::ElementwiseDivDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        double>,
    ops::ElementwiseDivDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        int>,
    ops::ElementwiseDivDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        int64_t>,
    ops::ElementwiseDivDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        paddle::platform::complex<float>>,
    ops::ElementwiseDivDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        paddle::platform::complex<double>>);
