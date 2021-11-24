/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/gelu_op.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

DEVICE float normcdf(float x) { return ::normcdff(x); }
DEVICE double normcdf(double x) { return ::normcdf(x); }

DEVICE float exp(float x) { return ::expf(x); }
DEVICE double exp(double x) { return ::exp(x); }

template <typename T>
class GeluCUDAFunctor {
 private:
  const T* in_;
  T* out_;
  using MPType = typename details::MPTypeTrait<T>::Type;

 public:
  GeluCUDAFunctor(const T* in, T* out) : in_(in), out_(out) {}

  inline DEVICE void operator()(size_t i) {
    MPType x = static_cast<MPType>(in_[i]);
    out_[i] = static_cast<T>(x * normcdf(x));
  }
};

template <typename T>
class GeluCUDAGradFunctor {
 private:
  const T* in_;
  const T* dy_;
  T* out_;
  using MPType = typename details::MPTypeTrait<T>::Type;

 public:
  GeluCUDAGradFunctor(const T* in, const T* dy, T* out)
      : in_(in), dy_(dy), out_(out) {}

  inline DEVICE void operator()(size_t i) {
    MPType x = static_cast<MPType>(in_[i]);
    MPType dy = static_cast<MPType>(dy_[i]);
    constexpr MPType kBeta = M_2_SQRTPI * M_SQRT1_2 * static_cast<MPType>(0.5);
    const MPType cdf = normcdf(x);
    const MPType pdf = exp(static_cast<MPType>(-0.5) * x * x) * kBeta;
    out_[i] = static_cast<T>(dy * (cdf + x * pdf));
  }
};

template <typename DeviceContext, typename T>
class GeluCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* out = context.Output<framework::Tensor>("Out");
    auto* in = context.Input<framework::Tensor>("X");
    auto approximate = context.Attr<bool>("approximate");
    out->mutable_data<T>(in->place());

    platform::ForRange<DeviceContext> for_range(
        static_cast<const DeviceContext&>(context.device_context()),
        in->numel());
    GeluCUDAFunctor<T> functor(in->data<T>(), out->data<T>());
    for_range(functor);
  }
};

template <typename DeviceContext, typename T>
class GeluGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<framework::Tensor>("X");
    auto* dout =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* dx = context.Output<framework::Tensor>(framework::GradVarName("X"));
    auto approximate = context.Attr<bool>("approximate");
    dx->mutable_data<T>(dout->place());

    platform::ForRange<DeviceContext> for_range(
        static_cast<const DeviceContext&>(context.device_context()),
        x->numel());
    GeluCUDAGradFunctor<T> functor(x->data<T>(), dout->data<T>(),
                                   dx->data<T>());
    for_range(functor);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    gelu, ops::GeluCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::GeluCUDAKernel<paddle::platform::CUDADeviceContext, double>,
    ops::GeluCUDAKernel<paddle::platform::CUDADeviceContext,
                        paddle::platform::float16>);
REGISTER_OP_CUDA_KERNEL(
    gelu_grad,
    ops::GeluGradCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::GeluGradCUDAKernel<paddle::platform::CUDADeviceContext, double>,
    ops::GeluGradCUDAKernel<paddle::platform::CUDADeviceContext,
                            paddle::platform::float16>);
