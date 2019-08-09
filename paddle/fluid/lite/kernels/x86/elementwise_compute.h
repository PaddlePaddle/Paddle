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
#pragma once

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
struct AddFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const { return a + b; }
};

template <typename T>
class ElementwiseSubCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ElementwiseParam;

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    auto& context = ctx_->As<X86Context>();
    CHECK(context.x86_device_context());

    param.Out->template mutable_data<T>();
    paddle::operators::ElementwiseComputeEx<SubFunctor<T>,
                                            platform::CPUDeviceContext, T>(
        *context.x86_execution_context(), &param.X->raw_tensor(),
        &param.Y->raw_tensor(), param.axis, SubFunctor<T>(),
        &param.Out->raw_tensor());
  }

  virtual ~ElementwiseSubCompute() = default;
};

template <typename T>
struct SubGradDX {
  T operator()(T x, T y, T out, T dout) const { return dout; }
};

template <typename T>
struct SubGradDY {
  T operator()(T x, T y, T out, T dout) const { return -dout; }
};

#ifdef LITE_WITH_X86
template <typename T>
class ElementwiseSubGradCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ElementwiseGradParam;
  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    auto& context = ctx_->As<X86Context>();
    CHECK(context.x86_device_context());

    param.X_grad->template mutable_data<T>();
    // skip out, x, y
    auto dout = param.Out_grad->raw_tensor();
    auto dx = param.X_grad->raw_tensor();

    framework::Tensor* dy = nullptr;
    if (param.Y_grad) {
      param.Y_grad->template mutable_data<T>();
      dy = &param.Y_grad->raw_tensor();
    }
    auto& skip = dout;
    paddle::operators::ElemwiseExplicitGradCompute<
        platform::CPUDeviceContext, T, SubGradDX<T>, SubGradDY<T>>(
        *context.x86_execution_context(), skip, skip, skip, dout, param.axis,
        &dx, dy, SubGradDX<T>(), SubGradDY<T>());
  }

  virtual ~ElementwiseSubGradCompute() = default;
};
#endif

template <typename T>
class ElementwiseAddCompute
    : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ElementwiseParam;
  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    auto& context = ctx_->As<X86Context>();
    CHECK(context.x86_device_context());
    param.Out->template mutable_data<T>();
    paddle::operators::ElementwiseComputeEx<AddFunctor<T>,
                                            platform::CPUDeviceContext, T>(
        *context.x86_execution_context(), &param.X->raw_tensor(),
        &param.Y->raw_tensor(), param.axis, AddFunctor<T>(),
        &param.Out->raw_tensor());
  }

  virtual ~ElementwiseAddCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
