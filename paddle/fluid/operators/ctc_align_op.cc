/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/ctc_align_op.h"

namespace paddle {
namespace operators {

class CTCAlignOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "ctc_align");
    OP_INOUT_CHECK(ctx->HasOutput("Output"), "Output", "Output", "ctc_align");

    auto input_dims = ctx->GetInputDim("Input");

    // TODO(wanghaoshuang): it is tricky to set the wrong dimension here.
    ctx->SetOutputDim("Output", input_dims);
    if (ctx->HasInput("InputLength")) {
      ctx->SetOutputDim("OutputLength", {input_dims[0], 1});
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Input"),
        ctx.device_context());
  }
};

class CTCAlignOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input",
             "2-D Tensor or LodTensor with  shape "
             "[Lp, 1], where Lp is the sum of all input sequences' length.");
    AddInput("InputLength",
             "2-D Tensor with shape [batch_size, 1], "
             " When Input is padding mode, InputLength is length of every "
             "sequence in Input.")
        .AsDispensable();
    AddOutput("Output", "(Tensor, default: Tensor<int>), The align result.");
    AddOutput("OutputLength",
              "2-D Tensor with shape [batch_size, 1], "
              "When Input is padding mode, OutputLength is length of every "
              "sequence in Output.")
        .AsDispensable();
    AddAttr<int>("blank",
                 "(int, default: 0), the blank label set in Connectionist "
                 "Temporal Classification (CTC) op.")
        .SetDefault(0);
    AddAttr<bool>("merge_repeated",
                  "(bool, default: true), whether to "
                  "merge repeated elements between two blanks. ")
        .SetDefault(true);
    // add attr padding number for tensor input
    AddAttr<int>("padding_value",
                 "(int, default: 0), padding number "
                 "use to padding tensor. ")
        .SetDefault(0);
    AddComment(R"DOC(
CTCAlign op is used to merge repeated elements between two blanks
and then delete all blanks in sequence.

Given:
    Input.data = [0, 1, 2, 2, 0, 4, 0, 4, 5, 0, 6,
                  6, 0, 0, 7, 7, 7, 0]
    Input.dims = {18, 1}
    Input.LoD = [[0, 11, 18]]

And:
    blank = 0
    merge_repeated = True

Then:
    Output.data = [1, 2, 4, 4, 5, 6,
                   6, 7]
    Output.dims = {8, 1}
    Output.LoD = [[0, 6, 8]]
or Given:
    Input.data = [[0, 1, 2, 2, 0, 4],
                  [0, 4, 5, 0, 6, 0],
                  [0, 7, 7, 7, 0, 0]]
    InputLength.data  = [[6],
                         [5],
                         [4]],
    Input.dims = {3, 6},
    Input.Lod = []
And:
    blank = 0
    merge_repeated = True
    padding_value = 0

Then:
    Output.data = [[1, 2, 4, 0, 0, 0],
                   [4, 5, 6, 0, 0, 0],
                   [7, 0, 0, 0, 0, 0]],
    OutputLength.data = [[3],
                         [3],
                         [1]],
    Output.dims = {3, 6},
    Output.Lod = []
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    ctc_align,
    ops::CTCAlignOp,
    ops::CTCAlignOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(ctc_align,
                       ops::CTCAlignKernel<phi::CPUContext, int>,
                       ops::CTCAlignKernel<phi::CPUContext, int64_t>);
