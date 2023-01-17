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

#include <memory>
#include <string>

#include "paddle/fluid/operators/elementwise/elementwise_mlu.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

template <typename T>
class ElementwiseMinMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    MLUBinaryOp<MINIMUM, T>(ctx);
  }
};

template <typename T>
class ElementwiseMinGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    MLUMinMaxGradHelper<MINIMUM_GRAD, T>(ctx);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(elementwise_min,
                       ops::ElementwiseMinMLUKernel<int>,
                       ops::ElementwiseMinMLUKernel<float>,
                       ops::ElementwiseMinMLUKernel<plat::float16>);

REGISTER_OP_MLU_KERNEL(elementwise_min_grad,
                       ops::ElementwiseMinGradMLUKernel<int>,
                       ops::ElementwiseMinGradMLUKernel<float>,
                       ops::ElementwiseMinGradMLUKernel<plat::float16>);
