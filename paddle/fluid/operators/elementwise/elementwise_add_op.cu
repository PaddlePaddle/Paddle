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
#include "paddle/fluid/operators/elementwise/elementwise_op_impl.cu.h"
#include "paddle/fluid/platform/complex128.h"
#include "paddle/fluid/platform/complex64.h"
#include "paddle/fluid/platform/float16.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

namespace paddle {
namespace operators {

/*
   input: an array;
   return: the result of the math functor
   1. For Unary Op, the length of input array is 1,
      e.g. Relu: return args[0] > 0 ? args[0] : 0;
   2. For Binary Op, the length of input array is 2,
      e.g. Add: return args[0] + args[1];
*/
template <typename T>
struct CudaAddFunctor {
  __device__ __forceinline__ T operator()(const T* args) const {
    return args[0] + args[1];
  }
};

template <typename T>
struct SameDimsElemwiseAdd<platform::CUDADeviceContext, T> {
  void operator()(const framework::ExecutionContext& ctx,
                  const framework::Tensor* x, const framework::Tensor* y,
                  framework::Tensor* z) {
    std::vector<const framework::Tensor*> ins = {x, y};
    std::vector<framework::Tensor*> outs = {z};
    LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
        ctx.template device_context<platform::CUDADeviceContext>(), ins, &outs,
        CudaAddFunctor<T>());
  }
};

template <typename T>
static __global__ void SimpleElemwiseAddGradCUDAKernel(
    const T* __restrict__ dout, int size, int vec_size, T* dx, T* dy) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int loop = size / vec_size;
  int remainder = size % vec_size;
  const float4* dout_vec = reinterpret_cast<const float4*>(dout);
  float4* dx_vec = reinterpret_cast<float4*>(dx);
  float4* dy_vec = reinterpret_cast<float4*>(dy);
  float4 tmp_loop;

  for (int i = tid; i < loop; i += stride) {
    tmp_loop = dout_vec[i];
    dx_vec[i] = tmp_loop;
    dy_vec[i] = tmp_loop;
  }

  if (tid == loop && remainder != 0) {
    T tmp_rem;
    while (remainder) {
      int idx = size - remainder;
      remainder--;
      tmp_rem = dout[idx];
      dx[idx] = tmp_rem;
      dy[idx] = tmp_rem;
    }
  }
}

template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_same<DeviceContext, plat::CUDADeviceContext>::value>::type
elementwise_add_grad(const framework::ExecutionContext& ctx,
                     const framework::Tensor* x, const framework::Tensor* y,
                     const framework::Tensor* out,
                     const framework::Tensor* dout, framework::Tensor* dx,
                     framework::Tensor* dy) {
  auto* dx_data = dx->mutable_data<T>(ctx.GetPlace());
  auto* dy_data = dy->mutable_data<T>(ctx.GetPlace());
  auto* dout_data = dout->data<T>();
  if (dx_data == dout_data && dy_data != dout_data) {
    VLOG(4) << "Special case when dx_data is the same as dout_data, "
               "only need copy dout to dy";
    framework::TensorCopy(
        *dout, ctx.GetPlace(),
        ctx.template device_context<platform::DeviceContext>(), dy);
  } else if (dx_data != dout_data && dy_data == dout_data) {
    VLOG(4) << "Special case when dy_data is the same as dout_data, "
               "only need copy dout to dx";
    framework::TensorCopy(
        *dout, ctx.GetPlace(),
        ctx.template device_context<platform::DeviceContext>(), dx);
  } else if (dx_data != dout_data && dy_data != dout_data) {
    auto size = x->numel();
    int vec_size = max(static_cast<int>(sizeof(float4) / sizeof(T)), 1);
    dim3 block_size = dim3(PADDLE_CUDA_THREAD_SIZE, 1);
    dim3 grid_size =
        dim3(((size + vec_size - 1) / vec_size + PADDLE_CUDA_THREAD_SIZE - 1) /
                 PADDLE_CUDA_THREAD_SIZE,
             1);
    SimpleElemwiseAddGradCUDAKernel<
        T><<<grid_size, block_size, 0,
             ctx.template device_context<plat::CUDADeviceContext>().stream()>>>(
        dout->data<T>(), size, vec_size, dx->mutable_data<T>(ctx.GetPlace()),
        dy->mutable_data<T>(ctx.GetPlace()));
  } else {
    VLOG(4) << "Special case when dy_data is the same as dout_data, "
               "and dx_data is the same as dout_data, do not need "
               "any operator";
  }
}

}  // namespace operators
}  // namespace paddle
REGISTER_OP_CUDA_KERNEL(
    elementwise_add, ops::ElementwiseAddKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, int64_t>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, plat::float16>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, plat::complex64>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, plat::complex128>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_add_grad,
    ops::ElementwiseAddGradKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseAddGradKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseAddGradKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseAddGradKernel<plat::CUDADeviceContext, int64_t>,
    ops::ElementwiseAddGradKernel<plat::CUDADeviceContext, plat::float16>,
    ops::ElementwiseAddGradKernel<plat::CUDADeviceContext, plat::complex64>,
    ops::ElementwiseAddGradKernel<plat::CUDADeviceContext, plat::complex128>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_add_grad_grad,
    ops::ElementwiseAddDoubleGradKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseAddDoubleGradKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseAddDoubleGradKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseAddDoubleGradKernel<plat::CUDADeviceContext, int64_t>,
    ops::ElementwiseAddDoubleGradKernel<plat::CUDADeviceContext, plat::float16>,
    ops::ElementwiseAddDoubleGradKernel<plat::CUDADeviceContext,
                                        plat::complex64>,
    ops::ElementwiseAddDoubleGradKernel<plat::CUDADeviceContext,
                                        plat::complex128>);

REGISTER_OP_CUDA_KERNEL(
    grad_add, ops::ElementwiseAddKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, int64_t>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, plat::float16>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, plat::complex64>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, plat::complex128>);
