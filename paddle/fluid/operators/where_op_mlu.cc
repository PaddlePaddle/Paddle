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

#ifdef PADDLE_WITH_MLU

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class WhereMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* condition = context.Input<phi::DenseTensor>("Condition");
    auto* X = context.Input<phi::DenseTensor>("X");
    auto* Y = context.Input<phi::DenseTensor>("Y");
    auto* out = context.Output<phi::DenseTensor>("Out");
    auto place = context.GetPlace();
    out->mutable_data<T>(place);
    MLUCnnlTensorDesc x_desc(*X);
    MLUCnnlTensorDesc y_desc(*Y);
    MLUCnnlTensorDesc condition_desc(*condition);
    MLUCnnlTensorDesc out_desc(*out);
    MLUCnnl::Select(context,
                    condition_desc.get(),
                    GetBasePtr(condition),
                    x_desc.get(),
                    GetBasePtr(X),
                    y_desc.get(),
                    GetBasePtr(Y),
                    out_desc.get(),
                    GetBasePtr(out));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_MLU_KERNEL(
    where,
    ops::WhereMLUKernel<paddle::platform::MLUDeviceContext, float>,
    ops::WhereMLUKernel<paddle::platform::MLUDeviceContext, int>);
#endif
