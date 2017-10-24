/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/fill_constant_batch_size_like_op.h"

namespace paddle {
namespace operators {

class FillConstantBatchSizeLikeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(
        ctx->HasInput("Input"),
        "Input(Input) of FillConstantBatchSizeLikeOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("Out"),
        "Output(Out) of FillConstantBatchSizeLikeOp should not be null.");

    auto &shape = ctx->Attrs().Get<std::vector<int>>("shape");
    PADDLE_ENFORCE_GT(shape.size(), 0);
    std::vector<int64_t> shape_int64(shape.size(), 0);
    std::transform(shape.begin(), shape.end(), shape_int64.begin(),
                   [](int a) { return static_cast<int64_t>(a); });
    auto dims = framework::make_ddim(shape_int64);

    dims[0] = ctx->GetInputDim("Input")[0];
    ctx->SetOutputDim("Out", dims);
  }

 protected:
  framework::DataType IndicateDataType(
      const framework::ExecutionContext &ctx) const override {
    return static_cast<framework::DataType>(ctx.Attr<int>("data_type"));
  }
};

class FillConstantBatchSizeLikeOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  FillConstantBatchSizeLikeOpMaker(framework::OpProto *proto,
                                   framework::OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddAttr<int>("data_type",
                 "(int, default 5 (FP32)) "
                 "Output data type")
        .SetDefault(framework::DataType::FP32);
    AddAttr<std::vector<int>>("shape", "(vector<int>) The shape of the output");
    AddAttr<float>("value", "(float, default 0) The value to be filled")
        .SetDefault(0.0f);
    AddInput("Input",
             "(Tensor) Tensor "
             "whose first dimension is used to specify the batch_size");
    AddOutput("Out",
              "(Tensor) Tensor of specified shape will be filled "
              "with the specified value");
    AddComment(R"DOC(Fill up a variable with specified constant value.)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(fill_constant_batch_size_like,
                             ops::FillConstantBatchSizeLikeOp,
                             ops::FillConstantBatchSizeLikeOpMaker);
REGISTER_OP_CPU_KERNEL(
    fill_constant_batch_size_like,
    ops::FillConstantBatchSizeLikeOpKernel<paddle::platform::CPUPlace, float>);
