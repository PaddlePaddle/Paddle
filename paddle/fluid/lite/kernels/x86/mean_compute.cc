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

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenScalar = framework::EigenScalar<T, MajorType, IndexType>;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename T>
class MeanCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::MeanParam;

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    auto& context = ctx_->As<X86Context>();
    CHECK(context.x86_device_context());

    param.Out->template mutable_data<T>();

    auto X = EigenVector<T>::Flatten(param.X->raw_tensor());
    auto y = EigenScalar<T>::From(param.Out->raw_tensor());
    const auto& place = *(context.x86_device_context()->eigen_device());

    y.device(place) = X.mean();
  }

  virtual ~MeanCompute() = default;
};

template <typename T>
class MeanGradCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::MeanGradParam;

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    auto& context = ctx_->As<X86Context>();
    CHECK_EQ(param.Out_grad->raw_tensor().numel(), 1);
    CHECK(context.x86_device_context());

    param.X_grad->template mutable_data<T>();
    T x_grad_size = static_cast<T>(param.X_grad->raw_tensor().numel());
    Eigen::DSizes<int, 1> bcast(static_cast<int>(x_grad_size));
    EigenVector<T>::Flatten(param.X_grad->raw_tensor())
        .device(*(context.x86_device_context()->eigen_device())) =
        (EigenVector<T>::From(param.Out_grad->raw_tensor()) / x_grad_size)
            .broadcast(bcast);
  }

  virtual ~MeanGradCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

// float
REGISTER_LITE_KERNEL(mean, kX86, kFloat, kNCHW,
                     paddle::lite::kernels::x86::MeanCompute<float>, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();

REGISTER_LITE_KERNEL(mean_grad, kX86, kFloat, kNCHW,
                     paddle::lite::kernels::x86::MeanGradCompute<float>, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput(paddle::framework::GradVarName("Out"),
               {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput(paddle::framework::GradVarName("X"),
                {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();
