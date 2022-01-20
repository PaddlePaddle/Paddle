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

template <typename Tx, typename Ty = Tx>
struct UnsignedPowFunctor {
  HOSTDEVICE explicit inline UnsignedPowFunctor(float porder) {
    this->porder = porder;
  }
  HOSTDEVICE inline Ty operator()(const Tx x) const {
    return static_cast<Ty>(inline_pow(inline_abs(x), static_cast<Tx>(porder)));
  }
  float porder;
};

template <typename Tx, typename Ty = Tx>
struct PowFunctor {
  HOSTDEVICE explicit inline PowFunctor(float porder) { this->porder = porder; }
  HOSTDEVICE inline Ty operator()(const Tx x) const {
    return static_cast<Ty>(inline_pow(x, static_cast<Tx>(porder)));
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
    auto ndim = out_norm->dims();
    float porder = ctx.Attr<float>("porder");
    bool asvector = ctx.Attr<bool>("asvector");
    int axis = ctx.Attr<int>("axis");
    std::vector<int> reduce_axis = {axis};
    reduce_axis = GetReduceDim(reduce_axis, xdim.size(), asvector);

    auto stream = ctx.cuda_device_context().stream();

    using MT = typename details::MPTypeTrait<T>::Type;
    if (porder == 0) {
      TensorReduceFunctorImpl<T, T, kps::AddFunctor, NonzeroFunctor<T>>(
          *in_x, out_norm, NonzeroFunctor<T>(), reduce_axis, stream);
    } else if (porder == INFINITY) {
      TensorReduceFunctorImpl<T, T, kps::MaxFunctor, AbsFunctor<T>>(
          *in_x, out_norm, AbsFunctor<T>(), reduce_axis, stream);
    } else if (porder == -INFINITY) {
      TensorReduceFunctorImpl<T, T, kps::MinFunctor, AbsFunctor<T>>(
          *in_x, out_norm, AbsFunctor<T>(), reduce_axis, stream);
    } else {
      framework::Tensor tmp_x;
      tmp_x.mutable_data<T>(xdim, ctx.GetPlace());
      std::vector<const framework::Tensor*> ins = {in_x};
      std::vector<framework::Tensor*> outs = {&tmp_x};
      auto func = UnsignedPowFunctor<MT, T>(porder);
      const auto& cuda_ctx =
          ctx.template device_context<platform::CUDADeviceContext>();

      paddle::operators::LaunchSameDimsElementwiseCudaKernel<
          ElementwiseType::kUnary, MT, T, UnsignedPowFunctor<MT, T>>(
          cuda_ctx, ins, &outs, func);
      framework::Tensor tmp_y;
      tmp_y.mutable_data<T>(ndim, ctx.GetPlace());
      TensorReduceFunctorImpl<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
          tmp_x, &tmp_y, kps::IdentityFunctor<T>(), reduce_axis, stream);
      const framework::Tensor* tmp_norm = &tmp_y;
      ins = {tmp_norm};
      outs = {out_norm};
      auto func_inverse = UnsignedPowFunctor<MT, T>(1. / porder);

      paddle::operators::LaunchSameDimsElementwiseCudaKernel<
          ElementwiseType::kUnary, MT, T, UnsignedPowFunctor<MT, T>>(
          cuda_ctx, ins, &outs, func_inverse);
    }
  }
};

template <typename T>
struct AbsMaxAndMinGradFunctor {
  template <typename DeviceContext, typename X, typename Y, typename DX,
            typename DY, typename Dim>
  void operator()(const DeviceContext& place, X* x, Y* y, DX* dx, DY* dy,
                  const Dim& dim, int size) {
    auto equals = ((*x).abs() == y->broadcast(dim));
    auto ones = dx->constant(static_cast<T>(1.));
    auto negs = dx->constant(static_cast<T>(-1.));
    auto zeros = dx->constant(static_cast<T>(0.));
    auto positives = (*x) > zeros;
    dx->device(place) = dy->broadcast(dim) * equals.select(ones, zeros) *
                        positives.select(ones, negs);
  }
};

template <typename T>
struct PNormPostGradFunctor {
  template <typename DeviceContext, typename X, typename Y, typename DX,
            typename DY, typename Dim>
  void operator()(const DeviceContext& place, X* x, Y* y, DX* dx, DY* dy,
                  const Dim& dim, int size) {
    auto ones = dx->constant(static_cast<T>(1.));
    auto negs = dx->constant(static_cast<T>(-1.));
    auto zeros = dx->constant(static_cast<T>(0.));
    auto positives = (*x) > zeros;
    dx->device(place) = (*dx) * dy->broadcast(dim) * y->broadcast(dim) *
                        positives.select(ones, negs);
  }
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
      math::SetConstant<DeviceContext, T> set_zero;
      set_zero(cuda_ctx, out_dx, static_cast<T>(0));
    } else if (porder == INFINITY || porder == -INFINITY) {
      LaunchReduceGradKernel<DeviceContext, T, AbsMaxAndMinGradFunctor<T>>(
          ctx, in_x, in_norm, in_norm_dy, out_dx, dims, reduce_all);
    } else {
      framework::Tensor tmp_norm;
      tmp_norm.mutable_data<T>(in_norm->dims(), ctx.GetPlace());
      std::vector<const framework::Tensor*> ins = {in_norm};
      std::vector<framework::Tensor*> outs = {&tmp_norm};
      auto pow_functor = PowFunctor<T>(1. - porder);
      paddle::operators::LaunchSameDimsElementwiseCudaKernel<
          ElementwiseType::kUnary, T, T, PowFunctor<T>>(cuda_ctx, ins, &outs,
                                                        pow_functor);
      ins = {in_x};
      outs = {out_dx};
      auto unsigned_pow = UnsignedPowFunctor<T>(porder - 1.);
      paddle::operators::LaunchSameDimsElementwiseCudaKernel<
          ElementwiseType::kUnary, T, T, UnsignedPowFunctor<T>>(
          cuda_ctx, ins, &outs, unsigned_pow);
      const framework::Tensor* tmp_norm_const = &tmp_norm;
      LaunchReduceGradKernel<DeviceContext, T, PNormPostGradFunctor<T>>(
          ctx, in_x, tmp_norm_const, in_norm_dy, out_dx, dims, reduce_all);
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
