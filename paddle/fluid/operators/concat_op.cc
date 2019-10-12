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

#include "paddle/fluid/operators/concat_op.h"
#include <memory>
#include <string>
#include <vector>

#ifdef PADDLE_WITH_MKLDNN
#include <paddle/fluid/platform/mkldnn_helper.h>
#endif

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

class ConcatOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_GE(ctx->Inputs("X").size(), 1UL,
                      "Inputs(X) of ConcatOp should not be empty.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of ConcatOp should not be null.");

    auto ins = ctx->GetInputsDim("X");
    size_t axis =
        ComputeAxis(static_cast<int64_t>(ctx->Attrs().Get<int>("axis")),
                    static_cast<int64_t>(ins[0].size()));

    const size_t n = ins.size();

    PADDLE_ENFORCE_GT(n, 0, "Input tensors count should > 0.");
    if (n == 1) {
      VLOG(3) << "Warning: concat op have only one input, may waste memory";
    }

    auto out_dims = ins[0];
    size_t in_zero_dims_size = out_dims.size();
    for (size_t i = 1; i < n; i++) {
      for (size_t j = 0; j < in_zero_dims_size; j++) {
        if (j == axis) {
          if (ctx->IsRuntime()) {
            out_dims[axis] += ins[i][j];
          } else {
            if (ins[i][j] == -1) {
              out_dims[axis] = -1;
            } else {
              out_dims[axis] += ins[i][j];
            }
          }
        } else {
          bool check_shape =
              ctx->IsRuntime() || (out_dims[j] > 0 && ins[i][j] > 0);
          if (check_shape) {
            // check all shape in run time
            PADDLE_ENFORCE_EQ(out_dims[j], ins[i][j],
                              "Input tensors should have the same "
                              "elements except the specify axis.");
          }
        }
      }
    }
    if (out_dims[axis] < 0) {
      out_dims[axis] = -1;
    }
    ctx->SetOutputDim("Out", out_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto inputs = ctx.MultiInput<Tensor>("X");
    auto input_data_type = framework::proto::VarType::Type(0);
    bool flag = 0;
    for (auto *input : inputs) {
      if (input->IsInitialized() && input->numel() > 0) {
        input_data_type = input->type();
        flag = 1;
        break;
      }
    }
    if (flag == 0) {
      PADDLE_THROW("All Inputs of Concat OP are Empty!");
    }

#ifdef PADDLE_WITH_MKLDNN
    if (platform::CanMKLDNNBeUsed(ctx)) {
      return framework::OpKernelType(input_data_type, ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

class ConcatOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input tensors of concat operator.").AsDuplicable();
    AddOutput("Out", "Output tensor of concat operator.");
    AddAttr<bool>(
        "use_mkldnn",
        "(bool, default false) Indicates if MKL-DNN kernel will be used")
        .SetDefault(false);
    AddAttr<int>("axis",
                 "The axis along which the input tensors will be concatenated."
                 "The axis could also be negative numbers. Negative axis is "
                 "interpreted as counting from the end of the rank."
                 "i.e., axis + rank(X) th dimension.")
        .SetDefault(0);
    AddAttr<bool>("use_quantizer",
                  "(bool, default false) "
                  "Set to true for operators that should be quantized and use "
                  "int8 kernel. "
                  "Only used on CPU.")
        .SetDefault(false);
    AddComment(R"DOC(
Concat Operator.

Concatenate the input tensors along dimension axis.
Examples:
  Input[0] = [[1,2],[3,4]]
  Input[1] = [[5,6]]
  axis = 0
  Output = [[1,2],
            [3,4],
            [5,6]]

)DOC");
  }
};

class ConcatOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    auto in_x = "X";
    auto out_x_g_n = framework::GradVarName(in_x);
    ctx->SetOutputsDim(out_x_g_n, ctx->GetInputsDim(in_x));
    auto &in_names = ctx->Inputs(in_x);
    auto &out_names = ctx->Outputs(out_x_g_n);
    PADDLE_ENFORCE_EQ(
        in_names.size(), out_names.size(),
        "The number of arguments in %s[%d] and %s[%d] is not equal.", in_x,
        in_names.size(), out_x_g_n, out_names.size());
    for (size_t i = 0; i < in_names.size(); ++i) {
      if (out_names[i] != framework::kEmptyVarName) {
        ctx->ShareLoD(in_x, out_x_g_n, i, i);
      }
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        ctx.Input<Tensor>(framework::GradVarName("Out"))->type(),
        ctx.GetPlace());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERENCE(ConcatOpGradNoNeedBufferVarInference,
                                      "X");

class ConcatGradOpDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("concat_grad");
    op->SetInput("X", Input("X"));
    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), InputGrad("X", false));
    op->SetAttrMap(Attrs());
    return op;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(concat, ops::ConcatOp, ops::ConcatOpMaker,
                  ops::ConcatGradOpDescMaker);
REGISTER_OPERATOR(concat_grad, ops::ConcatOpGrad,
                  ops::ConcatOpGradNoNeedBufferVarInference);
REGISTER_OP_CPU_KERNEL(
    concat, ops::ConcatKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ConcatKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ConcatKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::ConcatKernel<paddle::platform::CPUDeviceContext, int>);
REGISTER_OP_CPU_KERNEL(
    concat_grad,
    ops::ConcatGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ConcatGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ConcatGradKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::ConcatGradKernel<paddle::platform::CPUDeviceContext, int>);
