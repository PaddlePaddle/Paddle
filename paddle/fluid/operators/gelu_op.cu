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
#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast.cu.h"
#include "paddle/fluid/operators/gelu_op.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

template <typename T>
struct GeluWithApproximateGradFunctor {
  using MPType = typename details::MPTypeTrait<T>::Type;
  inline HOSTDEVICE T operator()(T arg_x, T arg_dout) {
    MPType x = static_cast<MPType>(arg_x);
    MPType dout = static_cast<MPType>(arg_dout);
    MPType kAlpha = static_cast<MPType>(M_2_SQRTPI * M_SQRT1_2);
    MPType one = static_cast<MPType>(1);
    MPType half = static_cast<MPType>(0.5);
    auto tanh_out =
        tanh(kAlpha * x * (one + static_cast<MPType>(0.044715) * x * x));
    auto ans =
        half * x * ((one - tanh_out * tanh_out) *
                    (kAlpha + static_cast<MPType>(0.1070322243) * x * x)) +
        half * (one + tanh_out);
    return static_cast<T>(ans * dout);
  }
};

template <typename T>
struct GeluWithoutApproximateGradFunctor {
  using MPType = typename details::MPTypeTrait<T>::Type;
  inline HOSTDEVICE T operator()(T arg_x, T arg_dout) {
    MPType x = static_cast<MPType>(arg_x);
    MPType dout = static_cast<MPType>(arg_dout);
    MPType kAlpha = static_cast<MPType>(M_2_SQRTPI * M_SQRT1_2);
    MPType one = static_cast<MPType>(1);
    MPType half = static_cast<MPType>(0.5);
    auto ans = half * (one + erf(x * static_cast<MPType>(M_SQRT1_2))) +
               half * kAlpha * x * exp(-half * x * x);
    return static_cast<T>(ans * dout);
  }
};

template <typename T>
class GeluGradKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<framework::Tensor>("X");
    auto* dout =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* dx = context.Output<framework::Tensor>(framework::GradVarName("X"));
    auto approximate = context.Attr<bool>("approximate");
    dx->mutable_data<T>(dout->place());

    std::vector<const framework::Tensor*> ins = {x, dout};
    std::vector<framework::Tensor*> outs = {dx};
    const auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();
    if (approximate) {
      LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
          dev_ctx, ins, &outs, 0, GeluWithApproximateGradFunctor<T>());
    } else {
      LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
          dev_ctx, ins, &outs, 0, GeluWithoutApproximateGradFunctor<T>());
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    gelu, ops::GeluKernel<paddle::platform::CUDADeviceContext, float>,
    ops::GeluKernel<paddle::platform::CUDADeviceContext, double>,
    ops::GeluKernel<paddle::platform::CUDADeviceContext,
                    paddle::platform::float16>);
REGISTER_OP_CUDA_KERNEL(
    gelu_grad, ops::GeluGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::GeluGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::GeluGradKernel<paddle::platform::CUDADeviceContext,
                        paddle::platform::float16>);
