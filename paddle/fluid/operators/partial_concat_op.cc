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

#include "paddle/fluid/operators/partial_concat_op.h"

#include <memory>
#include <string>
#include <vector>

namespace paddle {
namespace operators {

class PartialConcatOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_GE(
        ctx->Inputs("X").size(),
        1UL,
        platform::errors::InvalidArgument(
            "Inputs(X) of Partial ConcatOp should not be empty."));

    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("Out"),
        true,
        platform::errors::InvalidArgument(
            "Output(Out) of Partial ConcatOp should not be null."));

    auto inputs_dims = ctx->GetInputsDim("X");
    PADDLE_ENFORCE_EQ(inputs_dims[0].size(),
                      2,
                      platform::errors::InvalidArgument(
                          "Only supports 2-D array with batch size in the 1st "
                          "dimension and data in the 2nd."));

    const size_t inputs_num = inputs_dims.size();
    PADDLE_ENFORCE_GT(inputs_num,
                      0,
                      platform::errors::InvalidArgument(
                          "ShapeError: Input tensors count should > 0. But "
                          "received inputs' length is 0."));
    if (inputs_num == 1) {
      VLOG(3) << "Warning: concat op have only one input, may waste memory";
    }

    int64_t batch_size = -1;
    int64_t input_len = -1;
    for (size_t i = 0; i < inputs_num; ++i) {
      PADDLE_ENFORCE_EQ(inputs_dims[i].size(),
                        2,
                        platform::errors::InvalidArgument(
                            "It only supports two dimensions input now."));
      if (i == 0) {
        batch_size = inputs_dims[0][0];
        input_len = inputs_dims[0][1];
      } else {
        PADDLE_ENFORCE_EQ(inputs_dims[i][0],
                          batch_size,
                          platform::errors::InvalidArgument(
                              "The batch size of all inputs must be same"));
        PADDLE_ENFORCE_EQ(inputs_dims[i][1],
                          input_len,
                          platform::errors::InvalidArgument(
                              "The input length of all inputs must be same"));
      }
    }

    int start_index = static_cast<int>(ComputeStartIndex(
        static_cast<int64_t>(ctx->Attrs().Get<int>("start_index")),
        inputs_dims[0][1]));
    int partial_len = ctx->Attrs().Get<int>("length");
    if (partial_len < 0) {
      partial_len = static_cast<int>(inputs_dims[0][1] - start_index);
    }

    ctx->SetOutputDim(
        "Out",
        {inputs_dims[0][0], static_cast<int64_t>(partial_len * inputs_num)});
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto inputs = ctx.MultiInput<phi::DenseTensor>("X");
    auto input_data_type = framework::proto::VarType::Type(0);
    bool flag = false;
    for (auto *input : inputs) {
      if (input->IsInitialized()) {
        input_data_type = framework::TransToProtoVarType(input->dtype());
        flag = true;
        break;
      }
    }
    PADDLE_ENFORCE_EQ(flag,
                      1,
                      platform::errors::InvalidArgument(
                          "All Inputs of PartialSum OP are Empty!"));
    return phi::KernelKey(input_data_type, ctx.GetPlace());
  }
};

class PartialConcatGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    auto in_x = "X";
    auto out_x_g_n = framework::GradVarName(in_x);
    ctx->SetOutputsDim(out_x_g_n, ctx->GetInputsDim(in_x));

    auto in_names = ctx->Inputs(in_x);
    auto out_names = ctx->Outputs(out_x_g_n);

    PADDLE_ENFORCE_EQ(
        in_names.size(),
        out_names.size(),
        platform::errors::InvalidArgument(
            "The number of arguments in %s[%d] and %s[%d] is not equal.",
            in_x,
            in_names.size(),
            out_x_g_n,
            out_names.size()));
    for (size_t i = 0; i < in_names.size(); ++i) {
      if (out_names[i] != framework::kEmptyVarName) {
        ctx->ShareLoD(in_x, out_x_g_n, i, i);
      }
    }
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(
                              ctx, framework::GradVarName("Out")),
                          ctx.device_context().GetPlace());
  }
};

class PartialConcatOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input tensors of concat operator.").AsDuplicable();
    AddOutput("Out", "Output tensor of concat operator.");
    AddAttr<int>("start_index",
                 "The start index of each instance for concatenation.")
        .SetDefault(0);
    AddAttr<int>("length",
                 "The length of each instance for concatenation."
                 " Negative values for all elements after start_index")
        .SetDefault(-1);
    AddComment(R"DOC(
Partial Concat Operator.
Partial Concatenate the input tensors along the 2nd dimension.
Only 2-D Tensor or LodTensor input is supported.
Slice and concat can only be performed along the second dimension.
Examples:
  Input[0] = [[1,2],[3,4]]
  Input[1] = [[5,6],[7,8]]
  start_index = 1
  length = 1
  Output = [[2,6],
            [4,8]]
)DOC");
  }
};

template <typename T>
class PartialConcatGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("partial_concat_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X", false));
    op->SetAttr("start_index", this->GetAttr("start_index"));
    op->SetAttr("length", this->GetAttr("length"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(partial_concat,
                  ops::PartialConcatOp,
                  ops::PartialConcatOpMaker,
                  ops::PartialConcatGradMaker<paddle::framework::OpDesc>,
                  ops::PartialConcatGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(partial_concat_grad, ops::PartialConcatGradOp);

PD_REGISTER_STRUCT_KERNEL(partial_concat,
                          CPU,
                          ALL_LAYOUT,
                          ops::PartialConcatKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}
PD_REGISTER_STRUCT_KERNEL(partial_concat_grad,
                          CPU,
                          ALL_LAYOUT,
                          ops::PartialConcatGradientOpKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}
