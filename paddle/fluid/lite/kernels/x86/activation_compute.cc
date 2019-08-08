// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/operators/activation_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename Functor>
void Activate(const platform::CPUDeviceContext& context,
              const framework::LoDTensor* X, framework::LoDTensor* Out) {
  using T = typename Functor::ELEMENT_TYPE;
  auto* place = context.eigen_device();
  auto x =
      framework::EigenVector<T>::Flatten(paddle::operators::detail::Ref(X));
  auto out =
      framework::EigenVector<T>::Flatten(paddle::operators::detail::Ref(Out));
  Functor()(*place, x, out);
}

template <typename Functor>
void ActivateGrad(const platform::CPUDeviceContext& context,
                  const framework::LoDTensor* X,
                  const framework::LoDTensor* Out,
                  const framework::LoDTensor* Out_grad,
                  framework::LoDTensor* X_grad) {
  using T = typename Functor::ELEMENT_TYPE;
  auto* place = context.eigen_device();
  auto x =
      framework::EigenVector<T>::Flatten(paddle::operators::detail::Ref(X));
  auto out =
      framework::EigenVector<T>::Flatten(paddle::operators::detail::Ref(Out));
  auto x_grad = framework::EigenVector<T>::Flatten(
      paddle::operators::detail::Ref(X_grad));
  auto out_grad = framework::EigenVector<T>::Flatten(
      paddle::operators::detail::Ref(Out_grad));
  Functor()(*place, x, out, out_grad, x_grad);
}

template <typename T>
class SquareCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  void Run() override {
    auto& context = ctx_->As<X86Context>();
    auto& param = *param_.get_mutable<operators::ActivationParam>();
    CHECK(context.x86_device_context());

    param.Out->template mutable_data<T>();
    Activate<paddle::operators::SquareFunctor<T>>(*context.x86_device_context(),
                                                  &param.X->raw_tensor(),
                                                  &param.Out->raw_tensor());
  }

  virtual ~SquareCompute() = default;
};

template <typename T>
class SquareGradCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationGradParam;

  void Run() override {
    auto& context = ctx_->As<X86Context>();
    auto& param = *param_.get_mutable<operators::ActivationGradParam>();
    CHECK(context.x86_device_context());
    param.X_grad->template mutable_data<T>();

    ActivateGrad<paddle::operators::SquareGradFunctor<T>>(
        *context.x86_device_context(), &param.X->raw_tensor(),
        &param.Out->raw_tensor(), &param.Out_grad->raw_tensor(),
        &param.X_grad->raw_tensor());
  }

  virtual ~SquareGradCompute() = default;
};

template <typename T>
class SoftsignCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  void Run() override {
    auto& context = ctx_->As<X86Context>();
    auto& param = *param_.get_mutable<operators::ActivationParam>();
    CHECK(context.x86_device_context());
    param.Out->template mutable_data<T>();

    Activate<paddle::operators::SoftsignFunctor<T>>(
        *context.x86_device_context(), &param.X->raw_tensor(),
        &param.Out->raw_tensor());
  }

  virtual ~SoftsignCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

// float
REGISTER_LITE_KERNEL(square, kX86, kFloat, kNCHW,
                     paddle::lite::kernels::x86::SquareCompute<float>, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();

REGISTER_LITE_KERNEL(square_grad, kX86, kFloat, kNCHW,
                     paddle::lite::kernels::x86::SquareGradCompute<float>, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput(paddle::framework::GradVarName("Out"),
               {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput(paddle::framework::GradVarName("X"),
                {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();

REGISTER_LITE_KERNEL(softsign, kX86, kFloat, kNCHW,
                     paddle::lite::kernels::x86::SoftsignCompute<float>, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();
