/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/solve_op.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/ddim.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

using framework::OpKernelType;
using framework::Tensor;

class SolveOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Solve");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "Solve");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "Solve");

    auto x_dims = ctx->GetInputDim("X");
    // auto y_dims = ctx->GetInputDim("Y");

    std::vector<int64_t> x_dims_vec =
        paddle::framework::vectorize(ctx->GetInputDim("X"));
    std::vector<int64_t> y_dims_vec =
        paddle::framework::vectorize(ctx->GetInputDim("Y"));

    auto x_dims_n = x_dims_vec.size();
    auto y_dims_n = y_dims_vec.size();

    PADDLE_ENFORCE_GT(x_dims_n, 1,
                      platform::errors::InvalidArgument(
                          "The input tensor X's dimensions of SolveOp "
                          "should be larger than 1. But received X's "
                          "dimensions = %d, X's shape = [%s]",
                          x_dims_n, x_dims));

    PADDLE_ENFORCE_EQ(x_dims[x_dims_n - 2], x_dims[x_dims_n - 1],
                      platform::errors::InvalidArgument(
                          "The inner-most 2 dimensions of Input(X) all should "
                          "be square matrices "
                          "But received X's shape[-2] = %d and shape[-1] = %d.",
                          x_dims[x_dims_n - 2], x_dims[x_dims_n - 1]));

    bool x_broadcasted = false, y_broadcasted = false;
    bool trans_x = false, trans_y = false;
    if (x_dims_n == 1) {
      x_dims_vec.insert(x_dims_vec.begin(), 1);
      x_dims_n = 2;
      x_broadcasted = true;
    }

    if (y_dims_n == 1) {
      y_dims_vec.push_back(1);
      y_dims_n = 2;
      y_broadcasted = true;
    }

    size_t M, N;
    if (trans_x) {
      M = x_dims_vec[x_dims_n - 1];
    } else {
      M = x_dims_vec[x_dims_n - 2];
    }
    if (trans_y) {
      N = y_dims_vec[y_dims_n - 2];
    } else {
      N = y_dims_vec[y_dims_n - 1];
    }

    std::vector<int64_t> new_dims;
    if (x_dims_n >= y_dims_n) {
      new_dims.assign(x_dims_vec.begin(), x_dims_vec.end() - 2);
    } else {
      new_dims.assign(y_dims_vec.begin(), y_dims_vec.end() - 2);
    }
    if (!x_broadcasted) {
      new_dims.push_back(M);
    }
    if (!y_broadcasted) {
      new_dims.push_back(N);
    }
    if (x_broadcasted && y_broadcasted) {
      new_dims.push_back(1);
    }

    auto out_dims = framework::make_ddim(new_dims);
    ctx->SetOutputDim("Out", out_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    int customized_type_value =
        framework::OpKernelType::kDefaultCustomizedTypeValue;
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
#ifdef PADDLE_WITH_MKLDNN
    if (library == framework::LibraryType::kPlain &&
        this->CanMKLDNNBeUsed(ctx, input_data_type)) {
      library = framework::LibraryType::kMKLDNN;
      layout = framework::DataLayout::kMKLDNN;

      if (input_data_type == framework::DataTypeTrait<int8_t>::DataType() ||
          input_data_type == framework::DataTypeTrait<uint8_t>::DataType()) {
        customized_type_value = kMULMKLDNNINT8;
      }
    }
#endif

    return framework::OpKernelType(input_data_type, ctx.GetPlace(), layout,
                                   library, customized_type_value);
  }
};

class SolveOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The first input tensor of solve op.");
    AddInput("Y", "(Tensor), The second input tensor of solve op.");
    AddOutput("Out", "(Tensor), The output tensor of solve op.");
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false);
    AddAttr<float>(
        "scale_x",
        "scale_x to be used for int8 solve input data x. scale_x has the"
        "same purpose as scale_in in OPs that support quantization."
        "Only to be used with MKL-DNN INT8")
        .SetDefault(1.0f);
    AddAttr<std::vector<float>>(
        "scale_y",
        "scale_y to be used for int8 solve input data y. scale_y has the"
        "same purpose as scale_weights in OPs that support quantization."
        "Only to be used with MKL-DNN INT8")
        .SetDefault({1.0f});
    AddAttr<float>("scale_out",
                   "scale_out to be used for int8 output data."
                   "Only used with MKL-DNN INT8")
        .SetDefault(1.0f);
    AddAttr<bool>(
        "force_fp32_output",
        "(bool, default false) Force quantize kernel output FP32, only "
        "used in quantized MKL-DNN.")
        .SetDefault(false);
    AddComment(R"DOC(
Solve Operator.

This operator is used to computes the solution of a square system of linear equations with a unique solution for input $X$ and $Y$.

The equation is:

$$Out = X^-1 * Y$$

Both the input $X$ and $Y$ can carry the LoD (Level of Details) information,
or not. But the output only shares the LoD information with input $X$.

)DOC");
  }
};

class SolveOpInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{{"X", /*->*/ "Out"}};
    return m;
  }
};

class SolveGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "solve");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "solve");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "solve");
    OP_INOUT_CHECK(ctx->HasInput("Out"), "Input", "Out", "solve");

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");

    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");

    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
    if (ctx->HasOutput(y_grad_name)) {
      ctx->SetOutputDim(y_grad_name, y_dims);
    }
  }
};

template <typename T>
class SolveOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("solve_grad");
    retv->SetInput("X", this->Input("X"));
    retv->SetInput("Y", this->Input("Y"));
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetInput("Out", this->Output("Out"));
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    retv->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    retv->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(solve, ops::SolveOp, ops::SolveOpMaker,
                  ops::SolveOpInferVarType,
                  ops::SolveOpGradMaker<paddle::framework::OpDesc>,
                  ops::SolveOpGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(solve_grad, ops::SolveGradOp);

REGISTER_OP_CPU_KERNEL(
    solve, ops::SolveKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SolveKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    solve_grad, ops::SolveGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SolveGradKernel<paddle::platform::CPUDeviceContext, double>);
