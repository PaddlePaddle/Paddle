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
#include "paddle/fluid/operators/elementwise/elementwise_op_impl.cu.h"
#include "paddle/fluid/operators/fc_op.h"
#include "paddle/fluid/operators/p_norm_op.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.cu.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.h"
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

template <typename T>
struct NonzeroFunctor {
  HOSTDEVICE explicit inline NonzeroFunctor() {}
  HOSTDEVICE inline T operator()(const T x) const {
    return static_cast<T>(static_cast<double>(x) != 0);
  }
};

template <typename T>
struct AbsFunctor {
  HOSTDEVICE explicit inline AbsFunctor() {}
  HOSTDEVICE inline T operator()(const T x) const {
    return static_cast<T>(inline_abs(x));
  }
};

template <typename T>
struct UnsignedPowFunctor {
  HOSTDEVICE explicit inline UnsignedPowFunctor(float porder) {
    this->porder = porder;
  }
  HOSTDEVICE inline T operator()(const T x) const {
    return static_cast<T>(inline_pow(inline_abs(x), static_cast<T>(porder)));
  }
  float porder;
};

template <typename DeviceContext, typename T>
class PnormCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in_x = ctx.Input<framework::Tensor>("X");
    auto* out_norm = ctx.Output<framework::Tensor>("Out");
    const T* x = in_x->data<T>();
    T* norm = out_norm->mutable_data<T>(ctx.GetPlace());
    auto xdim = in_x->dims();
    float porder = ctx.Attr<float>("porder");
    bool asvector = ctx.Attr<bool>("asvector");
    int axis = ctx.Attr<int>("axis");
    std::vector<int> reduce_axis = {axis};
    reduce_axis = GetReduceDim(reduce_axis, xdim.size(), asvector);
    auto stream = ctx.cuda_device_context().stream();

    using MT = typename details::MPTypeTrait<T>::Type;
    if (porder == 0) {
      TensorReduceImpl<T, T, kps::AddFunctor, NonzeroFunctor<T>>(
          ctx.cuda_device_context(), *in_x, out_norm, NonzeroFunctor<T>(),
          reduce_axis, stream);
    } else if (porder == INFINITY) {
      TensorReduceImpl<T, T, kps::MaxFunctor, AbsFunctor<T>>(
          ctx.cuda_device_context(), *in_x, out_norm, AbsFunctor<T>(),
          reduce_axis, stream);
    } else if (porder == -INFINITY) {
      TensorReduceImpl<T, T, kps::MinFunctor, AbsFunctor<T>>(
          ctx.cuda_device_context(), *in_x, out_norm, AbsFunctor<T>(),
          reduce_axis, stream);
    } else {
      TensorReduceImpl<T, T, kps::AddFunctor, UnsignedPowFunctor<T>>(
          ctx.cuda_device_context(), *in_x, out_norm,
          UnsignedPowFunctor<T>(porder), reduce_axis, stream);

      const framework::Tensor* tmp_norm = out_norm;
      std::vector<const framework::Tensor*> ins = {tmp_norm};
      std::vector<framework::Tensor*> outs = {out_norm};
      const auto& cuda_ctx =
          ctx.template device_context<platform::CUDADeviceContext>();
      paddle::operators::LaunchSameDimsElementwiseCudaKernel<T>(
          cuda_ctx, ins, &outs, UnsignedPowFunctor<T>(1. / porder));
    }
  }
};

template <typename T>
struct AbsMaxAndMinGradFunctor {
  template <typename DeviceContext, typename X, typename Y, typename DX,
            typename DY, typename Dim>
  void operator()(const DeviceContext& place, X* x, Y* y, DX* dx, DY* dy,
                  const Dim& dim, int size) {
    dx->device(place) = dy->broadcast(dim) * (*x).sign() *
                        ((*x).abs() == y->broadcast(dim)).template cast<T>();
  }
};

template <typename T>
struct PNormGradFunctor {
  HOSTDEVICE explicit inline PNormGradFunctor(float porder) {
    this->porder = static_cast<T>(porder - 1.);
  }
  template <typename DeviceContext, typename X, typename Y, typename DX,
            typename DY, typename Dim>
  void operator()(const DeviceContext& place, X* x, Y* y, DX* dx, DY* dy,
                  const Dim& dim, int size) {
    dx->device(place) = (*x).abs().pow(this->porder) * (*x).sign() *
                        dy->broadcast(dim) *
                        (*y).pow(-this->porder).broadcast(dim);
  }
  T porder;
};

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

    auto xdim = in_x->dims();
    float porder = ctx.Attr<float>("porder");
    int axis = ctx.Attr<int>("axis");
    bool reduce_all = (in_norm->numel() == 1);
    if (axis < 0) axis = xdim.size() + axis;
    const std::vector<int> dims = {axis};

    auto& cuda_ctx = ctx.template device_context<DeviceContext>();

    if (porder == 0) {
      pten::funcs::SetConstant<DeviceContext, T> set_zero;
      set_zero(cuda_ctx, out_dx, static_cast<T>(0));
    } else if (porder == INFINITY || porder == -INFINITY) {
      AbsMaxAndMinGradFunctor<T> functor;
      LaunchReduceGradKernel<DeviceContext, T, AbsMaxAndMinGradFunctor<T>>(
          ctx, in_x, in_norm, in_norm_dy, out_dx, functor, dims, reduce_all);
    } else {
      auto functor = PNormGradFunctor<T>(porder);
      LaunchReduceGradKernel<DeviceContext, T, PNormGradFunctor<T>>(
          ctx, in_x, in_norm, in_norm_dy, out_dx, functor, dims, reduce_all);
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
