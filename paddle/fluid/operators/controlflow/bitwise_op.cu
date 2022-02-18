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

#include "paddle/fluid/operators/controlflow/bitwise_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast.cu.h"

namespace paddle {
namespace operators {

template <typename Functor>
class BinaryBitwiseOpKernel<platform::CUDADeviceContext, Functor>
    : public framework::OpKernel<typename Functor::ELEM_TYPE> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using T = typename Functor::ELEM_TYPE;

    auto* x = ctx.Input<framework::Tensor>("X");
    auto* y = ctx.Input<framework::Tensor>("Y");
    auto* out = ctx.Output<framework::Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    auto functor = Functor();
    std::vector<const framework::Tensor*> ins = {x, y};
    std::vector<framework::Tensor*> outs = {out};
    const auto& cuda_ctx =
        ctx.template device_context<platform::CUDADeviceContext>();
    paddle::operators::LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T,
                                                   T>(cuda_ctx, ins, &outs, -1,
                                                      functor);
  }
};

template <typename Functor>
class UnaryBitwiseOpKernel<platform::CUDADeviceContext, Functor>
    : public framework::OpKernel<typename Functor::ELEM_TYPE> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using T = typename Functor::ELEM_TYPE;

    auto* x = ctx.Input<framework::Tensor>("X");
    auto* out = ctx.Output<framework::Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    auto functor = Functor();
    std::vector<const framework::Tensor*> ins = {x};
    std::vector<framework::Tensor*> outs = {out};
    const auto& cuda_ctx =
        ctx.template device_context<platform::CUDADeviceContext>();
    paddle::operators::LaunchSameDimsElementwiseCudaKernel<T>(cuda_ctx, ins,
                                                              &outs, functor);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = ::paddle::operators;
namespace plat = ::paddle::platform;

REGISTER_BINARY_BITWISE_KERNEL(bitwise_and, CUDA, ops::BitwiseAndFunctor);
REGISTER_BINARY_BITWISE_KERNEL(bitwise_or, CUDA, ops::BitwiseOrFunctor);
REGISTER_BINARY_BITWISE_KERNEL(bitwise_xor, CUDA, ops::BitwiseXorFunctor);
REGISTER_UNARY_BITWISE_KERNEL(bitwise_not, CUDA, ops::BitwiseNotFunctor);
