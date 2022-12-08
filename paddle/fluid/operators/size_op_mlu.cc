// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

template <typename T>
class SizeMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("Input");
    auto* out = ctx.Output<phi::DenseTensor>("Out");
    out->mutable_data<int64_t>(ctx.GetPlace());

    int64_t size = x->numel();
    FillMLUTensorWithHostValue<int64_t>(ctx, size, out);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_MLU_KERNEL(size,
                       ops::SizeMLUKernel<int>,
                       ops::SizeMLUKernel<int64_t>,
                       ops::SizeMLUKernel<paddle::platform::float16>,
                       ops::SizeMLUKernel<float>,
                       ops::SizeMLUKernel<double>,
                       ops::SizeMLUKernel<bool>);
