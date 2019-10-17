// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/sequence_ops/sequence_mask_op.h"
#include <string>

namespace paddle {
namespace operators {

class SequenceMaskOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) must exist");
    PADDLE_ENFORCE(ctx->HasOutput("Y"), "Output(Y) must exist");

    int maxlen = ctx->Attrs().Get<int>("maxlen");
    auto dim = framework::vectorize<int>(ctx->GetInputDim("X"));

    if (ctx->HasInputs("MaxLenTensor")) {
      dim.push_back(-1);
    } else {
      dim.push_back(maxlen > 0 ? maxlen : -1);
    }
    ctx->SetOutputDim("Y", framework::make_ddim(dim));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<framework::LoDTensor>("X")->type(),
                                   ctx.device_context());
  }
  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    if (var_name == "depth_tensor") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

class SequenceMaskOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor of sequence_mask op.");
    AddOutput("Y", "The output mask of sequence_mask op.");
    AddInput("MaxLenTensor",
             "Max length tensor"
             "have higher priority than maxlen attribute")
        .AsDispensable();
    AddAttr<int>("maxlen",
                 "The maximum length of the sequence. If maxlen < 0, maxlen "
                 "= max(Input(X)).")
        .SetDefault(-1)
        .AddCustomChecker([](const int& v) {
          PADDLE_ENFORCE(v < 0 || v >= 1,
                         "Attr(maxlen) must be less than 0 or larger than 1");
        });
    AddAttr<int>("out_dtype", "Output data type");
    AddComment(R"DOC(
SequenceMask Operator

This operator outputs a Mask according to Input(X) and Attr(maxlen).
Supposing Input(X) is a Tensor with shape [d_1, d_2, ..., d_n], the
Output(Y) is a mask with shape [d_1, d_2, ..., d_n, maxlen], where:

Y(i_1, i_2, ..., i_n, j) = (j < X(i_1, i_2, ..., i_n)) 

If maxlen < 0, maxlen = max(X)
    )DOC");
  }
};
}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(sequence_mask, paddle::operators::SequenceMaskOp,
                  paddle::operators::SequenceMaskOpMaker,
                  paddle::framework::EmptyGradOpMaker);

REGISTER_OP_CPU_KERNEL(
    sequence_mask,
    paddle::operators::SequenceMaskKernel<paddle::platform::CPUDeviceContext,
                                          int>,
    paddle::operators::SequenceMaskKernel<paddle::platform::CPUDeviceContext,
                                          int64_t>,
    paddle::operators::SequenceMaskKernel<paddle::platform::CPUDeviceContext,
                                          float>,
    paddle::operators::SequenceMaskKernel<paddle::platform::CPUDeviceContext,
                                          double>);
