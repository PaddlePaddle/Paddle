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
#include "paddle/fluid/operators/elementwise/elementwise_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T>
struct SubFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const { return a - b; }
};

template <typename T>
class ElementwiseSubCompute
    : public KernelLite<TARGET(kHost), PRECISION(kFloat)> {
 public:
  using param_t = operators::ElementwiseParam;

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    auto& context = ctx_->As<X86Context>();
    CHECK(context.x86_device_context);

    param.Out->template mutable_data<T>();
    paddle::operators::ElementwiseComputeEx<SubFunctor<T>,
                                            platform::CPUDeviceContext, T>(
        *context.x86_execution_context, &param.X->raw_tensor(),
        &param.Y->raw_tensor(), param.axis, SubFunctor<T>(),
        &param.Out->raw_tensor());
  }

  // TargetType target() const override;
  // PrecisionType precision() const override;

  virtual ~ElementwiseSubCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

// float
REGISTER_LITE_KERNEL(square, kHost, kFloat, kNCHW,
                     paddle::lite::kernels::x86::ElementwiseSubCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();
