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

#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/binary.h"

namespace paddle {
namespace operators {

using framework::OpKernelType;

class BincountOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const {
    auto data_type =
        ctx.HasInput("Weights")
            ? OperatorWithKernel::IndicateVarDataType(ctx, "Weights")
            : OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class BincountOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) The input tensor of Bincount op,");
    AddInput("Weights", "(Tensor) The weights tensor of Bincount op,")
        .AsDispensable();
    AddOutput("Out", "(Tensor) The output tensor of Bincount op,");
    AddAttr<int>("minlength", "(int) The minimal numbers of bins")
        .SetDefault(0)
        .EqualGreaterThan(0)
        .SupportTensor();
    AddComment(R"DOC(
          Bincount Operator.
          Computes frequency of each value in the input tensor.
          Elements of input tensor should be non-negative ints.
      )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(bincount,
                            BincountInferShapeFunctor,
                            PD_INFER_META(phi::BincountInferMeta));
REGISTER_OPERATOR(
    bincount,
    ops::BincountOp,
    ops::BincountOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    BincountInferShapeFunctor);
