/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
Indicesou may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/p_norm_op.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

template <typename T>
__device__ __forceinline__ int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

__device__ __forceinline__ platform::float16 inline_abs(platform::float16 x) {
  return static_cast<platform::float16>(abs(static_cast<float>(x)));
}
__device__ __forceinline__ float inline_abs(float x) { return abs(x); }
__device__ __forceinline__ double inline_abs(double x) { return abs(x); }

__device__ __forceinline__ int inline_sign(platform::float16 x) {
  return sgn<platform::float16>(x);
}
__device__ __forceinline__ int inline_sign(float x) { return sgn<float>(x); }
__device__ __forceinline__ int inline_sign(double x) { return sgn<double>(x); }

__device__ __forceinline__ platform::float16 inline_pow(
    platform::float16 base, platform::float16 exponent) {
  return static_cast<platform::float16>(
      pow(static_cast<float>(base), static_cast<float>(exponent)));
}
__device__ __forceinline__ float inline_pow(float base, float exponent) {
  return pow(base, exponent);
}
__device__ __forceinline__ double inline_pow(double base, double exponent) {
  return pow(base, exponent);
}

template <typename T, int BlockDim>
__global__ void Pnorm(const T* x, const int pre,
                      const int axis_n,  // dim in axis
                      const int post, float porder, T* out_norm) {
  using MT = typename details::MPTypeTrait<T>::Type;
  typedef cub::BlockReduce<MT, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int num = pre * post;
  auto porder_t = static_cast<MT>(porder);
  auto porder_inv = static_cast<MT>(1.0 / porder);

  for (int i = blockIdx.x; i < num; i += gridDim.x) {
    int base = (i / post) * post * axis_n + (i % post);
    MT sum = static_cast<MT>(0.0);
    for (int j = threadIdx.x; j < axis_n; j += blockDim.x) {
      const MT x_ij = static_cast<MT>(x[base + j * post]);
      sum += inline_pow(inline_abs(x_ij), porder_t);
    }
    MT reduce_result = BlockReduce(temp_storage).Sum(sum);
    if (threadIdx.x == 0)
      out_norm[i] = static_cast<T>(inline_pow(reduce_result, porder_inv));
  }
}

template <typename T, int BlockDim>
__global__ void ZeorNorm(const T* x, const int pre,
                         const int axis_n,  // dim in axis
                         const int post, T* out_norm) {
  using MT = typename details::MPTypeTrait<T>::Type;
  typedef cub::BlockReduce<MT, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int num = pre * post;
  for (int i = blockIdx.x; i < num; i += gridDim.x) {
    int base = (i / post) * post * axis_n + (i % post);
    MT sum = static_cast<MT>(0.0);
    for (int j = threadIdx.x; j < axis_n; j += blockDim.x) {
      const MT x_ij = static_cast<MT>(x[base + j * post]);
      sum += static_cast<MT>(static_cast<double>(x_ij) != 0);
    }
    MT reduce_result = BlockReduce(temp_storage).Sum(sum);
    if (threadIdx.x == 0) out_norm[i] = static_cast<T>(reduce_result);
  }
}

template <typename T, int BlockDim>
__global__ void InfNorm(const T* x, const int pre,
                        const int axis_n,  // dim in axis
                        const int post, T* out_norm) {
  typedef cub::BlockReduce<T, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int num = pre * post;
  for (int i = blockIdx.x; i < num; i += gridDim.x) {
    int base = (i / post) * post * axis_n + (i % post);
    T cur_max = inline_abs(x[base]);
    for (int j = threadIdx.x; j < axis_n; j += blockDim.x) {
      T x_ij_abs = inline_abs(x[base + j * post]);
      if (cur_max < x_ij_abs) cur_max = x_ij_abs;
    }
    T reduce_result = BlockReduce(temp_storage).Reduce(cur_max, cub::Max());
    if (threadIdx.x == 0) out_norm[i] = reduce_result;
  }
}

template <typename T, int BlockDim>
__global__ void NegInfNorm(const T* x, const int pre,
                           const int axis_n,  // dim in axis
                           const int post, T* out_norm) {
  typedef cub::BlockReduce<T, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int num = pre * post;
  for (int i = blockIdx.x; i < num; i += gridDim.x) {
    int base = (i / post) * post * axis_n + (i % post);
    T cur_min = inline_abs(x[base]);
    for (int j = threadIdx.x; j < axis_n; j += blockDim.x) {
      T x_ij_abs = inline_abs(x[base + j * post]);
      if (cur_min > x_ij_abs) cur_min = x_ij_abs;
    }
    T reduce_result = BlockReduce(temp_storage).Reduce(cur_min, cub::Min());
    if (threadIdx.x == 0) out_norm[i] = reduce_result;
  }
}

template <typename DeviceContext, typename T>
class PnormCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in_x = ctx.Input<framework::Tensor>("X");
    auto* out_norm = ctx.Output<framework::Tensor>("Out");
    const T* x = in_x->data<T>();
    T* norm = out_norm->mutable_data<T>(ctx.GetPlace());

    auto xdim = in_x->dims();
    auto ndim = out_norm->dims();
    float porder = ctx.Attr<float>("porder");
    int axis = ctx.Attr<int>("axis");
    bool asvector = ctx.Attr<bool>("asvector");
    if (axis < 0) axis = xdim.size() + axis;
    int pre, n, post;
    GetDims(xdim, axis, &pre, &n, &post, asvector);

    auto& dev_ctx = ctx.cuda_device_context();

#ifdef __HIPCC__
    const int block = 256;
#else
    const int block = 512;
#endif

    int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
    const int max_blocks = std::max(max_threads / block, 1);
    int grid = std::min(max_blocks, pre * post);
    if (porder == 0) {
      ZeorNorm<T, block><<<grid, block, 0, dev_ctx.stream()>>>(x, pre, n, post,
                                                               norm);
    } else if (porder == INFINITY) {
      InfNorm<T, block><<<grid, block, 0, dev_ctx.stream()>>>(x, pre, n, post,
                                                              norm);
    } else if (porder == -INFINITY) {
      NegInfNorm<T, block><<<grid, block, 0, dev_ctx.stream()>>>(x, pre, n,
                                                                 post, norm);
    } else {
      Pnorm<T, block><<<grid, block, 0, dev_ctx.stream()>>>(x, pre, n, post,
                                                            porder, norm);
    }
  }
};

template <typename T, int BlockDim>
__global__ void PnormGradient(const T* x, const T* x_norm, const T* y_grad,
                              const float porder, const int pre,
                              const int axis_n, const int post, const T eps,
                              T* x_grad) {
  using MT = typename details::MPTypeTrait<T>::Type;
  // dx = (x/pnorm_broadcast).pow(p-1) * norm_dy.broadcast * sign(x)
  int num = pre * post;
  auto porder_grad = static_cast<MT>(porder - 1.0f);
  for (int i = blockIdx.x; i < num; i += gridDim.x) {
    __shared__ MT pnorm_i;
    __shared__ MT yout_i;

    auto base = (i / post) * post * axis_n + (i % post);

    if (threadIdx.x == 0) {
      pnorm_i = static_cast<MT>(x_norm[i]);
      yout_i = static_cast<MT>(y_grad[i]);
    }
    __syncthreads();

    for (int j = threadIdx.x; j < axis_n; j += blockDim.x) {
      int index = base + j * post;
      const MT x_ij = static_cast<MT>(inline_abs(x[index]));
      x_grad[index] = static_cast<T>(
          inline_pow(x_ij, porder_grad) /
          (inline_pow(pnorm_i, porder_grad) + static_cast<MT>(eps)) * yout_i *
          static_cast<MT>(inline_sign(x[index])));
    }
  }
}

template <typename T, int BlockDim>
__global__ void InfNormGradient(const T* x, const T* x_norm, const T* y_grad,
                                const int pre, const int axis_n, const int post,
                                T* x_grad) {
  int num = pre * post;
  for (int i = blockIdx.x; i < num; i += gridDim.x) {
    __shared__ T pnorm_i;
    __shared__ T yout_i;
    auto base = (i / post) * post * axis_n + (i % post);
    if (threadIdx.x == 0) {
      pnorm_i = x_norm[i];
      yout_i = y_grad[i];
    }
    __syncthreads();

    for (int j = threadIdx.x; j < axis_n; j += blockDim.x) {
      int index = base + j * post;
      const T x_ij = inline_abs(x[index]);
      if (x_ij == pnorm_i) {
        x_grad[index] = static_cast<T>(inline_sign(x[index])) * yout_i;
      } else {
        x_grad[index] = static_cast<T>(0);
      }
    }
  }
}

template <typename DeviceContext, typename T, typename AttrType = T>
class PnormGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in_x = ctx.Input<framework::Tensor>("X");
    auto* in_norm = ctx.Input<framework::Tensor>("Out");
    auto* in_norm_dy =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* out_dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    T* dx = out_dx->mutable_data<T>(ctx.GetPlace());
    const T* x = in_x->data<T>();
    const T* x_norm = in_norm->data<T>();
    const T* norm_dy = in_norm_dy->data<T>();

    auto xdim = in_x->dims();
    float porder = ctx.Attr<float>("porder");
    T eps = static_cast<T>(ctx.Attr<float>("epsilon"));
    int axis = ctx.Attr<int>("axis");
    bool asvector = ctx.Attr<bool>("asvector");
    if (axis < 0) axis = xdim.size() + axis;
    int pre, n, post;
    GetDims(xdim, axis, &pre, &n, &post, asvector);

    auto& dev_ctx = ctx.cuda_device_context();

#ifdef __HIPCC__
    const int block = 256;
#else
    const int block = 512;
#endif

    int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
    const int max_blocks = std::max(max_threads / block, 1);
    int grid = std::min(max_blocks, pre * post);
    if (porder == 0) {
      math::SetConstant<DeviceContext, T> set_zero;
      auto& dev_ctx = ctx.template device_context<DeviceContext>();
      set_zero(dev_ctx, out_dx, static_cast<T>(0));
    } else if (porder == INFINITY || porder == -INFINITY) {
      InfNormGradient<T, block><<<grid, block, 0, dev_ctx.stream()>>>(
          x, x_norm, norm_dy, pre, n, post, dx);
    } else {
      PnormGradient<T, block><<<grid, block, 0, dev_ctx.stream()>>>(
          x, x_norm, norm_dy, porder, pre, n, post, eps, dx);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CUDA = paddle::platform::CUDADeviceContext;

REGISTER_OP_CUDA_KERNEL(p_norm,
                        ops::PnormCUDAKernel<CUDA, paddle::platform::float16>,
                        ops::PnormCUDAKernel<CUDA, float>,
                        ops::PnormCUDAKernel<CUDA, double>);
REGISTER_OP_CUDA_KERNEL(
    p_norm_grad, ops::PnormGradCUDAKernel<CUDA, paddle::platform::float16>,
    ops::PnormGradCUDAKernel<CUDA, float>,
    ops::PnormGradCUDAKernel<CUDA, double>);
