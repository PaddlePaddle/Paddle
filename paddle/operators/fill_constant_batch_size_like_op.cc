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
    auto output_dim = framework::make_ddim(shape_int64);

    int input_dim_idx = ctx->Attrs().Get<int>("input_dim_idx");
    PADDLE_ENFORCE_GE(input_dim_idx, 0);
    PADDLE_ENFORCE_GT(ctx->GetInputDim("Input").size(), input_dim_idx);

    int output_dim_idx = ctx->Attrs().Get<int>("output_dim_idx");
    PADDLE_ENFORCE_GE(output_dim_idx, 0);
    PADDLE_ENFORCE_GT(static_cast<int>(shape.size()), output_dim_idx);

    output_dim[output_dim_idx] = ctx->GetInputDim("Input")[input_dim_idx];
    ctx->SetOutputDim("Out", output_dim);
  }

 protected:
  framework::OpKernelType GetKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        static_cast<framework::DataType>(ctx.Attr<int>("data_type")),
        ctx.device_context());
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
    AddInput("Input",
             "(Tensor) Tensor "
             "whose dim_idx th dimension is used to specify the batch_size");
    AddOutput("Out",
              "(Tensor) Tensor of specified shape will be filled "
              "with the specified value");
    AddAttr<std::vector<int>>("shape", "(vector<int>) The shape of the output");
    AddAttr<int>("input_dim_idx",
                 "(int, default 0) The index of input's batch size dimension")
        .SetDefault(0);
    AddAttr<int>("output_dim_idx",
                 "(int, default 0) The index of output's batch size dimension")
        .SetDefault(0);
    AddAttr<float>("value", "(float, default 0) The value to be filled")
        .SetDefault(0.0f);
    AddComment(R"DOC(
FillConstantBatchSizeLike Operator.

Fill up a variable with specified constant value.

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fill_constant_batch_size_like,
                  ops::FillConstantBatchSizeLikeOp,
                  paddle::framework::EmptyGradOpMaker,
                  ops::FillConstantBatchSizeLikeOpMaker);
REGISTER_OP_CPU_KERNEL(
    fill_constant_batch_size_like,
    ops::FillConstantBatchSizeLikeOpKernel<paddle::platform::CPUPlace, float>,
    ops::FillConstantBatchSizeLikeOpKernel<paddle::platform::CPUPlace, double>);
