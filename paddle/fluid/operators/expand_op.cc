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

#include "paddle/fluid/operators/expand_op.h"

#include <memory>
#include <string>
#include <vector>

namespace paddle {
namespace operators {

class ExpandOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Expand");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "Expand");
    auto x_dims = ctx->GetInputDim("X");
    auto expand_times = ctx->Attrs().Get<std::vector<int>>("expand_times");

    if (expand_times.size() == 0) {
      expand_times = std::vector<int>(x_dims.size(), -1);
    }

    PADDLE_ENFORCE_EQ(
        static_cast<size_t>(x_dims.size()),
        expand_times.size(),
        platform::errors::InvalidArgument(
            "The number of elements (%d) of 'expand_times' for "
            "Op(expand) must be equal to the number of dimensions "
            "(%d) of the input.",
            expand_times.size(),
            static_cast<size_t>(x_dims.size())));
    PADDLE_ENFORCE_LE(
        x_dims.size(),
        6,
        platform::errors::InvalidArgument(
            "The number of dimensions of the input for Op(expand) "
            "must not be greater than 6, but the value received is %d.",
            x_dims.size()));

    std::vector<int64_t> out_shape(x_dims.size());
    for (size_t i = 0; i < expand_times.size(); ++i) {
      if (x_dims[i] == -1 || expand_times[i] == -1) {
        out_shape[i] = -1;
      } else {
        PADDLE_ENFORCE_GT(
            expand_times[i],
            0,
            platform::errors::InvalidArgument(
                "The %uth element of 'expand_times' for Op(expand) must be "
                "greater than 0, but the value given is %d.",
                i,
                expand_times[i]));
        out_shape[i] = x_dims[i] * expand_times[i];
      }
    }

    ctx->SetOutputDim("Out", phi::make_ddim(out_shape));
    if (out_shape[0] == x_dims[0]) {
      ctx->ShareLoD("X", "Out");
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name,
      const phi::DenseTensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    if (var_name == "expand_times_tensor" || var_name == "ExpandTimes") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(
        expected_kernel_type.data_type_, tensor.place(), tensor.layout());
  }
};

class ExpandOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor, default Tensor<float>). A tensor with rank in [1, 6]."
             "X is the input to be expanded.");
    AddInput("ExpandTimes",
             "(Tensor<int>), optional). If provided, expand according to "
             "this given expand times. It has a higher priority than "
             "expand_times_tensor and expand_times.")
        .AsDispensable();
    AddInput("expand_times_tensor",
             "(Tensor Tensor<int>), epxand times for X."
             "It has a higher priority than expand_times, but a lower priority "
             "than ExpandTimes")
        .AsDuplicable()
        .AsDispensable();
    AddOutput("Out",
              "(Tensor, default Tensor<float>). A tensor with rank in [1, 6]."
              "The rank of Output(Out) have the same with Input(X). "
              "After expanding, size of each dimension of Output(Out) is equal "
              "to size of the corresponding dimension of Input(X) multiplying "
              "the corresponding value given by Attr(expand_times).");
    AddAttr<std::vector<int>>("expand_times",
                              "Expand times number for each dimension.")
        .SetDefault({});
    AddComment(R"DOC(
Expand operator tiles the input by given times number. You should set times
number for each dimension by providing attribute 'expand_times'. The rank of X
should be in [1, 6]. Please note that size of 'expand_times' must be the same
with X's rank. Following is a using case:

Input(X) is a 3-D tensor with shape [2, 3, 1]:

        [
           [[1], [2], [3]],
           [[4], [5], [6]]
        ]

Attr(expand_times):  [1, 2, 2]

Output(Out) is a 3-D tensor with shape [2, 6, 2]:

        [
            [[1, 1], [2, 2], [3, 3], [1, 1], [2, 2], [3, 3]],
            [[4, 4], [5, 5], [6, 6], [4, 4], [5, 5], [6, 6]]
        ]

)DOC");
  }
};

class ExpandGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "ExpandGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")),
                   "Input",
                   framework::GradVarName("Out"),
                   "ExpandGrad");

    auto x_dims = ctx->GetInputDim("X");
    std::vector<int> expand_times =
        ctx->Attrs().Get<std::vector<int>>("expand_times");

    auto out_dims = ctx->GetInputDim(framework::GradVarName("Out"));

    size_t start_pos = 0u;
    if (!ctx->IsRuntime() && x_dims[0] < 0) {
      PADDLE_ENFORCE_EQ(
          x_dims[0],
          out_dims[0],
          platform::errors::InvalidArgument(
              "The first dimension size (%d) of Input(Out@GRAD) should be "
              "equal to the crroresponding dimension size (%d) of Input(X)",
              out_dims[0],
              x_dims[0]));
      start_pos = 1u;
    }

    for (size_t i = start_pos; i < expand_times.size(); ++i) {
      if (expand_times[i] == -1) {
        continue;
      } else {
        if (ctx->IsRuntime()) {
          PADDLE_ENFORCE_EQ(
              x_dims[i] * expand_times[i],
              out_dims[i],
              platform::errors::InvalidArgument(
                  "The %uth dimension size (%d) of Input(Out@GRAD) should be "
                  "equal to the multiplication of the crroresponding dimension "
                  "sizes of Input(X) (%d) and expand_times (%d).",
                  i,
                  out_dims[i],
                  x_dims[i],
                  expand_times[i]));
        }
      }
    }
    auto x_grad_name = framework::GradVarName("X");

    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name,
      const phi::DenseTensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    if (var_name == "expand_times_tensor") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(
        expected_kernel_type.data_type_, tensor.place(), tensor.layout());
  }
};

template <typename T>
class ExpandGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("expand_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetInput("expand_times_tensor", this->Input("expand_times_tensor"));
    op->SetInput("ExpandTimes", this->Input("ExpandTimes"));
    op->SetAttrMap(this->Attrs());
  }
};

template <typename T>
class ExpandDoubleGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetInput("X", this->OutputGrad(framework::GradVarName("X")));
    op->SetOutput("Out", this->InputGrad(framework::GradVarName("Out")));
    if (this->HasInput("expand_times_tensor")) {
      op->SetInput("expand_times_tensor", this->Input("expand_times_tensor"));
    }
    if (this->HasInput("ExpandTimes")) {
      op->SetInput("ExpandTimes", this->Input("ExpandTimes"));
    }
    op->SetAttrMap(this->Attrs());
    op->SetType("expand");
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(ExpandGradNoNeedBufVarsInferer, "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(expand,
                  ops::ExpandOp,
                  ops::ExpandOpMaker,
                  ops::ExpandGradOpMaker<paddle::framework::OpDesc>,
                  ops::ExpandGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(expand_grad,
                  ops::ExpandGradOp,
                  ops::ExpandDoubleGradOpMaker<paddle::framework::OpDesc>,
                  ops::ExpandDoubleGradOpMaker<paddle::imperative::OpBase>,
                  ops::ExpandGradNoNeedBufVarsInferer);
REGISTER_OP_CPU_KERNEL(expand,
                       ops::ExpandKernel<phi::CPUContext, float>,
                       ops::ExpandKernel<phi::CPUContext, double>,
                       ops::ExpandKernel<phi::CPUContext, int>,
                       ops::ExpandKernel<phi::CPUContext, int64_t>,
                       ops::ExpandKernel<phi::CPUContext, bool>);
REGISTER_OP_CPU_KERNEL(expand_grad,
                       ops::ExpandGradKernel<phi::CPUContext, float>,
                       ops::ExpandGradKernel<phi::CPUContext, double>,
                       ops::ExpandGradKernel<phi::CPUContext, int>,
                       ops::ExpandGradKernel<phi::CPUContext, int64_t>);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
REGISTER_OP_CUDA_KERNEL(
    expand,
    ops::ExpandKernel<phi::GPUContext, float>,
    ops::ExpandKernel<phi::GPUContext, double>,
    ops::ExpandKernel<phi::GPUContext, paddle::platform::float16>,
    ops::ExpandKernel<phi::GPUContext, int>,
    ops::ExpandKernel<phi::GPUContext, int64_t>,
    ops::ExpandKernel<phi::GPUContext, bool>);
REGISTER_OP_CUDA_KERNEL(
    expand_grad,
    ops::ExpandGradKernel<phi::GPUContext, float>,
    ops::ExpandGradKernel<phi::GPUContext, double>,
    ops::ExpandGradKernel<phi::GPUContext, paddle::platform::float16>,
    ops::ExpandGradKernel<phi::GPUContext, int>,
    ops::ExpandGradKernel<phi::GPUContext, int64_t>);
#endif
