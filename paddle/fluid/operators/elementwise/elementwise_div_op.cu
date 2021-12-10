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
#include "paddle/fluid/operators/reduce_ops/reduce_functor_op.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.cu.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/fluid/platform/float16.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

namespace paddle {
namespace operators {

template <typename T>
static __global__ void GARDYCUDAKernel(const T* y, int64_t size, T* t_y) {}

template <>
__global__ void GARDYCUDAKernel<paddle::platform::complex<float>>(
    const paddle::platform::complex<float>* y, int64_t size,
    paddle::platform::complex<float>* t_y) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  while (col < size) {
    t_y[col].real = y[col].real;
    t_y[col].imag = -y[col].imag;
    col += blockDim.x * gridDim.x;
  }
}

template <>
__global__ void GARDYCUDAKernel<paddle::platform::complex<double>>(
    const paddle::platform::complex<double>* y, int64_t size,
    paddle::platform::complex<double>* t_y) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  while (col < size) {
    t_y[col].real = y[col].real;
    t_y[col].imag = -y[col].imag;
    col += blockDim.x * gridDim.x;
  }
}

template <typename T>
static __global__ void SimpleElemwiseDivGradCUDAKernel(const T* x, const T* y,
                                                       const T* out,
                                                       const T* dout,
                                                       int64_t size, T* dx,
                                                       T* dy) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  while (col < size) {
    T o = dout[col];
    if (dx != nullptr) {
      dx[col] = o / y[col];
    }
    dy[col] = -o * out[col] / y[col];
    col += blockDim.x * gridDim.x;
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
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  while (col < size) {
    paddle::platform::complex<float> o = dout[col];
    paddle::platform::complex<float> y_conj(y[col].real, -y[col].imag);
    paddle::platform::complex<float> out_div_y_conj((out[col] / y[col]).real,
                                                    -(out[col] / y[col]).imag);
    if (dx != nullptr) {
      dx[col] = o / y_conj;
    }
    dy[col] = -o * out_div_y_conj;
    col += blockDim.x * gridDim.x;
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
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  while (col < size) {
    paddle::platform::complex<double> o = dout[col];
    paddle::platform::complex<double> y_conj(y[col].real, -y[col].imag);
    paddle::platform::complex<double> out_div_y_conj((out[col] / y[col]).real,
                                                     -(out[col] / y[col]).imag);
    if (dx != nullptr) {
      dx[col] = o / y_conj;
    }
    dy[col] = -o * out_div_y_conj;
    col += blockDim.x * gridDim.x;
  }
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
  // dx
  if (dx != nullptr) {
    auto* dx_data = dx->mutable_data<T>(ctx.GetPlace());
    if (dx->dims() == dout->dims()) {
      if (framework::IsComplexType(y->type())) {
        int size = y->numel();
        framework::Tensor t_y;
        t_y.Resize(y->dims());
        dim3 grid_size = dim3(
            (size + ELEMENTWISE_BLOCK_SIZE - 1) / ELEMENTWISE_BLOCK_SIZE, 1);
        GARDYCUDAKernel<T><<<
            grid_size, block_size, 0,
            ctx.template device_context<plat::CUDADeviceContext>().stream()>>>(
            y->data<T>(), size, t_y.mutable_data<T>(ctx.GetPlace()));
        const framework::Tensor* const_t_y =
            const_cast<const framework::Tensor*>(&t_y);
        default_elementwise_div<DeviceContext, T>(ctx, dout, const_t_y,
                                                  dx);  //  dout/y
      } else {
        default_elementwise_div<DeviceContext, T>(ctx, dout, y, dx);  //  dout/y
      }
    } else {
      // For inplace strategy, dx will be stored in addr of dout, which makes
      // the result of dy wrong.
      if (dx->IsSharedBufferWith(*dout)) {
        dx->clear();
        dx->mutable_data<T>(x->dims(), ctx.GetPlace());
      }
      framework::Tensor t_dx;
      t_dx.Resize(dout->dims());
      default_elementwise_div<DeviceContext, T>(ctx, dout, y, &t_dx);  // dout/y

      std::vector<int> reduce_dims = GetReduceDim(x->dims(), out->dims(), axis);
      gpuStream_t stream = ctx.cuda_device_context().stream();
      TensorReduceFunctorImpl<T, T, CustomSum>(t_dx, dx, reduce_dims, stream);
    }
  }
  // dy
  if (dy != nullptr) {
    auto* dy_data = dy->mutable_data<T>(ctx.GetPlace());
    if (dy->dims() == dout->dims()) {
      if (dy_data != dout_data) {
        // - out / y
        auto size = dy->numel();
        dim3 grid_size = dim3(
            (size + ELEMENTWISE_BLOCK_SIZE - 1) / ELEMENTWISE_BLOCK_SIZE, 1);
        SimpleElemwiseDivGradCUDAKernel<T><<<
            grid_size, block_size, 0,
            ctx.template device_context<plat::CUDADeviceContext>().stream()>>>(
            x->data<T>(), y->data<T>(), out->data<T>(), dout->data<T>(), size,
            nullptr, dy->mutable_data<T>(ctx.GetPlace()));
      }
    } else {
      framework::Tensor t_dy;
      t_dy.Resize(out->dims());
      default_elementwise_mul<DeviceContext, T>(ctx, dout, out,
                                                &t_dy);  // t_dy=dout*out
      framework::Tensor t_dyy;
      t_dyy.Resize(out->dims());
      const framework::Tensor* const_t_dy =
          const_cast<const framework::Tensor*>(&t_dy);
      if (framework::IsComplexType(y->type())) {
        int size = y->numel();
        framework::Tensor t_y;
        t_y.Resize(y->dims());
<<<<<<< 41291834704179377ab90d036093cf824e728ba7
        // paddle::platform::complex<double> y_conj(y[col].real, -y[col].imag);
=======
>>>>>>> add broadcast_div_bw
        dim3 grid_size = dim3(
            (size + ELEMENTWISE_BLOCK_SIZE - 1) / ELEMENTWISE_BLOCK_SIZE, 1);
        GARDYCUDAKernel<T><<<
            grid_size, block_size, 0,
            ctx.template device_context<plat::CUDADeviceContext>().stream()>>>(
            y->data<T>(), size, t_y.mutable_data<T>(ctx.GetPlace()));
        const framework::Tensor* const_t_y =
            const_cast<const framework::Tensor*>(&t_y);
        default_elementwise_div<DeviceContext, T>(ctx, const_t_dy, const_t_y,
                                                  &t_dyy);  // t_dy/y
      } else {
        default_elementwise_div<DeviceContext, T>(ctx, const_t_dy, y,
                                                  &t_dyy);  // t_dy/y
      }
      std::vector<int> reduce_dims = GetReduceDim(y->dims(), out->dims(), axis);
      gpuStream_t stream = ctx.cuda_device_context().stream();
      TensorReduceFunctorImpl<T, T, CustomSub>(t_dyy, dy, reduce_dims, stream);
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
