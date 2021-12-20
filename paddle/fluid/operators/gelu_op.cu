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
struct GeluWithApproximateFunctor {
  using MPType = typename details::MPTypeTrait<T>::Type;
  inline HOSTDEVICE T operator()(T arg_x) {
    // this function is tanh approximation of gelu
    MPType x = static_cast<MPType>(arg_x);
    MPType one = static_cast<MPType>(1);
    MPType out = x * static_cast<MPType>(0.5) *
                 (one + tanh(static_cast<MPType>(0.79788456) * x *
                             (one + static_cast<MPType>(0.044715) * x * x)));
    return static_cast<T>(out);
  }
};

template <typename T>
struct GeluWithoutApproximateFunctor {
  using MPType = typename details::MPTypeTrait<T>::Type;
  inline HOSTDEVICE T operator()(T arg_x) {
    // actual gelu with approximation = false
    MPType x = static_cast<MPType>(arg_x);
    MPType erf_out = erf(x * static_cast<MPType>(M_SQRT1_2));
    MPType out =
        x * static_cast<MPType>(0.5) * (static_cast<MPType>(1) + erf_out);
    return static_cast<T>(out);
  }
};

template <typename T>
class GeluKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* out = context.Output<framework::Tensor>("Out");
    auto* in = context.Input<framework::Tensor>("X");
    auto approximate = context.Attr<bool>("approximate");
    out->mutable_data<T>(in->place());

    std::vector<const framework::Tensor*> ins = {in};
    std::vector<framework::Tensor*> outs = {out};
    const auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();
    if (approximate) {
      LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
          dev_ctx, ins, &outs, 0, GeluWithApproximateFunctor<T>());
    } else {
      LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
          dev_ctx, ins, &outs, 0, GeluWithoutApproximateFunctor<T>());
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
