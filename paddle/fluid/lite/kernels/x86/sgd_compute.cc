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
#include "paddle/fluid/operators/jit/kernels.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T>
class SGDCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  void Run() override {
    auto &context = ctx_->As<X86Context>();
    auto &sgd_param = *param_.get_mutable<operators::SGDParam>();
    CHECK(context.x86_device_context());

    // param.Out->template mutable_data<T>();

    const auto *param = &sgd_param.Param->raw_tensor();
    const auto *grad = &sgd_param.Grad->raw_tensor();
    const auto *learning_rate = &sgd_param.LearningRate->raw_tensor();
    auto *param_out = &sgd_param.ParamOut->raw_tensor();

    auto sz = param_out->numel();
    PADDLE_ENFORCE_EQ(param->numel(), sz);
    PADDLE_ENFORCE_EQ(grad->numel(), sz);

    paddle::operators::jit::sgd_attr_t attr(1, sz, 1, sz, 1);
    const T *lr = learning_rate->template data<T>();
    const T *param_data = param->template data<T>();
    const T *grad_data = grad->template data<T>();
    int64_t rows_idx = 0;

    T *out_data = param_out->template mutable_data<T>(
        context.x86_device_context()->GetPlace());

    auto sgd =
        paddle::operators::jit::KernelFuncs<paddle::operators::jit::SgdTuple<T>,
                                            platform::CPUPlace>::Cache()
            .At(attr);
    sgd(lr, param_data, grad_data, &rows_idx, out_data, &attr);
  }

  virtual ~SGDCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

// float
REGISTER_LITE_KERNEL(sgd, kX86, kFloat, kNCHW,
                     paddle::lite::kernels::x86::SGDCompute<float>, def)
    .BindInput("Param", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("LearningRate", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("Grad", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("ParamOut", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();
