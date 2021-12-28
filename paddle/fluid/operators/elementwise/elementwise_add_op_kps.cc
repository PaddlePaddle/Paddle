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
#ifdef PADDLE_WITH_XPU

#include "paddle/fluid/operators/elementwise/elementwise_add_op.h"
#include <memory>
#include <string>
#include "paddle/fluid/operators/elementwise/elementwise_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_xpu.h"

namespace paddle {
namespace operators {

// TODO(liuxiandong): add template
void ElementwiseAddXPU2Compute(const framework::ExecutionContext& ctx);

void ElementwiseAddGradXPU2Compute(const framework::ExecutionContext& ctx);

template <typename T>
class ElementwiseAddXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // ElementwiseAddXPU2Compute(ctx);
    std::cout << "lxd_debug: elementwise_add forward" << std::endl;
  }
};

template <typename T>
class ElementwiseAddGradXPUKernel
    : public ::paddle::operators::ElemwiseGradKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // ElementwiseAddGradXPU2Compute(ctx);
    std::cout << "lxd_debug: elementwise_add backward" << std::endl;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_KERNEL(elementwise_add, KP, plat::XPUPlace,
                   ops::ElementwiseAddXPUKernel<float>,
                   ops::ElementwiseAddXPUKernel<paddle::platform::float16>);
REGISTER_OP_KERNEL(elementwise_add_grad, KP, plat::XPUPlace,
                   ops::ElementwiseAddGradXPUKernel<float>,
                   ops::ElementwiseAddGradXPUKernel<paddle::platform::float16>);
#endif
