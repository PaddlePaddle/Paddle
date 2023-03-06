/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License. */

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/binary.h"

namespace paddle {
namespace operators {

class MoeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return phi::KernelKey(data_type, ctx.GetPlace());
  }
};

class MoeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The source input tensor of Moe op.");
    AddInput("Gate", "(Tensor), The gating input tensor of Moe op.");
    AddInput("Bmm0", "(Tensor), The bmm0 input tensor of Moe op.");
    AddInput("Bias0", "(Tensor), The eltwise0 input tensor of Moe op.");
    AddInput("Bmm1", "(Tensor), The bmm1 input tensor of Moe op.");
    AddInput("Bias1", "(Tensor), The eltwise1 input tensor of Moe op.");
    AddOutput("Out", "(Tensor), The output tensor of Moe op.");
    AddAttr<std::string>(
        "act_type",
        R"DOC(activation type, currently only support `gelu`, `relu`. Default value is: `gelu`. )DOC")
        .SetDefault("gelu");
    AddComment(
        R"DOC(FusedEcMoe kernel. For more details you can refer to `FusedEcMoE` python documents. )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(moe,
                            MoeInferShapeFunctor,
                            PD_INFER_META(phi::MoeInferMeta));
REGISTER_OPERATOR(moe, ops::MoeOp, ops::MoeOpMaker, MoeInferShapeFunctor);
