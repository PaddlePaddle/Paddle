/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_MLU

#include "paddle/fluid/operators/elementwise/elementwise_mlu.h"
#include "paddle/fluid/operators/elementwise/elementwise_op.h"
namespace paddle {
namespace operators {

template <typename T>
class ElementwiseMaxMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    MLUBinaryOp<MAXIMUM, T>(ctx);
  }
};

template <typename T>
class ElementwiseMaxGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    MLUMinMaxGradHelper<MAXIMUM_GRAD, T>(ctx);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_MLU_KERNEL(elementwise_max,
                       ops::ElementwiseMaxMLUKernel<int>,
                       ops::ElementwiseMaxMLUKernel<float>,
                       ops::ElementwiseMaxMLUKernel<paddle::platform::float16>);
REGISTER_OP_MLU_KERNEL(
    elementwise_max_grad,
    ops::ElementwiseMaxGradMLUKernel<int>,
    ops::ElementwiseMaxGradMLUKernel<float>,
    ops::ElementwiseMaxGradMLUKernel<paddle::platform::float16>);
#endif
