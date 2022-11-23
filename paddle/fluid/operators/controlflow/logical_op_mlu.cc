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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

template <typename T, cnnlLogicOp_t log_method>
class LogicalMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* y = ctx.Input<phi::DenseTensor>("Y");
    auto* out = ctx.Output<phi::DenseTensor>("Out");

    out->mutable_data<T>(ctx.GetPlace());

    if (log_method == CNNL_LOGIC_OP_NOT) {
      y = x;
    }

    MLUCnnlTensorDesc x_desc(*x);
    MLUCnnlTensorDesc y_desc(*y);
    MLUCnnlTensorDesc out_desc(*out);

    MLUCnnl::Logic(ctx,
                   log_method,
                   x_desc.get(),
                   GetBasePtr(x),
                   y_desc.get(),
                   GetBasePtr(y),
                   out_desc.get(),
                   GetBasePtr(out));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_MLU_KERNEL(logical_not,
                       ops::LogicalMLUKernel<bool, CNNL_LOGIC_OP_NOT>,
                       ops::LogicalMLUKernel<int8_t, CNNL_LOGIC_OP_NOT>,
                       ops::LogicalMLUKernel<int16_t, CNNL_LOGIC_OP_NOT>,
                       ops::LogicalMLUKernel<int, CNNL_LOGIC_OP_NOT>,
                       ops::LogicalMLUKernel<float, CNNL_LOGIC_OP_NOT>);

REGISTER_OP_MLU_KERNEL(logical_and,
                       ops::LogicalMLUKernel<bool, CNNL_LOGIC_OP_AND>,
                       ops::LogicalMLUKernel<int8_t, CNNL_LOGIC_OP_AND>,
                       ops::LogicalMLUKernel<int16_t, CNNL_LOGIC_OP_AND>,
                       ops::LogicalMLUKernel<int, CNNL_LOGIC_OP_AND>,
                       ops::LogicalMLUKernel<float, CNNL_LOGIC_OP_AND>);

REGISTER_OP_MLU_KERNEL(logical_or,
                       ops::LogicalMLUKernel<bool, CNNL_LOGIC_OP_OR>,
                       ops::LogicalMLUKernel<int8_t, CNNL_LOGIC_OP_OR>,
                       ops::LogicalMLUKernel<int16_t, CNNL_LOGIC_OP_OR>,
                       ops::LogicalMLUKernel<int, CNNL_LOGIC_OP_OR>,
                       ops::LogicalMLUKernel<float, CNNL_LOGIC_OP_OR>);

REGISTER_OP_MLU_KERNEL(logical_xor,
                       ops::LogicalMLUKernel<bool, CNNL_LOGIC_OP_XOR>,
                       ops::LogicalMLUKernel<int8_t, CNNL_LOGIC_OP_XOR>,
                       ops::LogicalMLUKernel<int16_t, CNNL_LOGIC_OP_XOR>,
                       ops::LogicalMLUKernel<int, CNNL_LOGIC_OP_XOR>,
                       ops::LogicalMLUKernel<float, CNNL_LOGIC_OP_XOR>);
