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
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

using framework::OpKernelType;

class HistogramOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class HistogramOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) The input tensor of Histogram op,");
    AddOutput("Out", "(Tensor) The output tensor of Histogram op,");
    AddAttr<int64_t>("bins", "(int) number of histogram bins")
        .SetDefault(100)
        .EqualGreaterThan(1);
    AddAttr<int>("min", "(int) lower end of the range (inclusive)")
        .SetDefault(0);
    AddAttr<int>("max", "(int) upper end of the range (inclusive)")
        .SetDefault(0);
    AddComment(R"DOC(
          Histogram Operator.
          Computes the histogram of a tensor. The elements are sorted
          into equal width bins between min and max. If min and max are
          both zero, the minimum and maximum values of the data are used.
      )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(histogram,
                            HistogramInferShapeFunctor,
                            PD_INFER_META(phi::HistogramInferMeta));

REGISTER_OPERATOR(
    histogram,
    ops::HistogramOp,
    ops::HistogramOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    HistogramInferShapeFunctor);
