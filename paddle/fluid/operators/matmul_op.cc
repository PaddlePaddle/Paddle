/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/kernels/impl/matmul_kernel_impl.h"

namespace paddle {
namespace operators {

/**
 * Printing shape information into a string is easy to use.
 */
inline static std::string DumpMatrixShape(
    const phi::funcs::MatDescriptor &desc) {
  std::stringstream buffer;
  buffer << "[" << desc.batch_size_ << ", " << desc.height_ << ", "
         << desc.width_ << "]";
  return buffer.str();
}

/**
 * Get row matrix shape from a vector shape. If the rank of x_dim > 1, the
 * original x_dim is returned.
 */
static phi::DDim RowMatrixFromVector(const phi::DDim &x_dim) {
  if (x_dim.size() > 1) {
    return x_dim;
  }
  return common::make_ddim({1, x_dim[0]});
}

/**
 * Get column matrix shape from a vector shape. If the ran of y_dim > 1, the
 * original y_dim is returned.
 */
static phi::DDim ColumnMatrixFromVector(const phi::DDim &y_dim) {
  if (y_dim.size() > 1) {
    return y_dim;
  }
  return common::make_ddim({y_dim[0], 1});
}

phi::DDim GetDimForInput(const framework::InferShapeContext &ctx,
                         std::string input_name) {
  auto dim = ctx.GetInputDim(input_name);
  PADDLE_ENFORCE_GT(dim.size(),
                    0,
                    common::errors::InvalidArgument(
                        "The Input(%s) has not been initialized properly. The "
                        "shape of Input(%s) = [%s].",
                        dim));
  return dim;
}

class MatMulOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", "matmul");
    OP_INOUT_CHECK(context->HasInput("Y"), "Input", "Y", "matmul");
    OP_INOUT_CHECK(context->HasOutput("Out"), "Output", "Out", "matmul");

    auto dim_x = GetDimForInput(*context, "X");
    auto dim_y = GetDimForInput(*context, "Y");

#ifdef PADDLE_WITH_DNNL
    // For NHWC execution output shape needs to be
    // computed like instead x*y we are to do y*x
    bool channelwise_onednn =
        context->IsRunMKLDNNKernel() &&
        (phi::OneDNNContext::tls().get_cur_paddle_data_layout() ==
         phi::DataLayout::kNHWC);
    if (channelwise_onednn) {
      std::swap(dim_x, dim_y);
    }
#endif

    auto mat_dim_x = phi::funcs::CreateMatrixDescriptor(
        RowMatrixFromVector(dim_x),
        0,
        context->Attrs().Get<bool>("transpose_X"));
    auto mat_dim_y = phi::funcs::CreateMatrixDescriptor(
        ColumnMatrixFromVector(dim_y),
        0,
        context->Attrs().Get<bool>("transpose_Y"));

    if (mat_dim_x.width_ == -1) {
      mat_dim_x.width_ = mat_dim_y.height_;
    }
    if (mat_dim_y.height_ == -1) {
      mat_dim_y.height_ = mat_dim_x.width_;
    }

    if (context->IsRuntime()) {
      PADDLE_ENFORCE_EQ(
          mat_dim_x.batch_size_ == mat_dim_y.batch_size_ ||
              mat_dim_x.batch_size_ == 0 || mat_dim_y.batch_size_ == 0,
          true,
          common::errors::InvalidArgument(
              "The batch size of the two matrices should be equal, or "
              "at least one is zero.\n"
              "But received X's shape: %s, Y's shape: %s.",
              DumpMatrixShape(mat_dim_x).c_str(),
              DumpMatrixShape(mat_dim_y).c_str()));
    }
    int64_t dim_out_y = mat_dim_y.width_;
#if defined(PADDLE_WITH_MKLML) && !defined(PADDLE_WITH_CUDA) && \
    !defined(PADDLE_WITH_HIP)
    int head_number = context->Attrs().Get<int>("head_number");
    bool split_vertical_y = (mat_dim_x.width_ != mat_dim_y.height_);
    if (context->IsRuntime()) {
      PADDLE_ENFORCE_LE(
          head_number,
          mat_dim_x.width_,
          common::errors::InvalidArgument(
              "Unsatisfied mkl acceleration library requirements: "
              "The number of heads "
              "(%d) must be equal to X's width. But received X's shape: %s.",
              head_number,
              DumpMatrixShape(mat_dim_x).c_str()));

      if (!split_vertical_y && head_number > 0) {
        dim_out_y = head_number * mat_dim_y.width_;
      }
    }
#else
    PADDLE_ENFORCE_EQ(mat_dim_x.width_,
                      mat_dim_y.height_,
                      common::errors::InvalidArgument(
                          "Input X's width should be equal to the Y's height, "
                          "but received X's shape: [%s], "
                          "Y's shape: [%s].",
                          dim_x,
                          dim_y));
#endif

    std::vector<int64_t> dim_out;
    if (mat_dim_x.batch_size_ != 0) {
      dim_out = common::vectorize(dim_x);
      dim_out[dim_out.size() - 2] = mat_dim_x.height_;
      dim_out[dim_out.size() - 1] = dim_out_y;
    } else if (mat_dim_y.batch_size_ != 0) {
      dim_out = common::vectorize(dim_y);
      dim_out[dim_out.size() - 2] = mat_dim_x.height_;
      dim_out[dim_out.size() - 1] = dim_out_y;
    } else {
      dim_out = {mat_dim_x.height_, dim_out_y};
    }

    if (dim_x.size() == 1 && dim_out[dim_out.size() - 2] == 1) {
      std::swap(dim_out[dim_out.size() - 2], dim_out[dim_out.size() - 1]);
      dim_out.resize(dim_out.size() - 1);
    }

    if (dim_y.size() == 1 && dim_out[dim_out.size() - 1] == 1) {
      dim_out.resize(dim_out.size() - 1);
    }

    phi::DDim ddim_out = common::make_ddim(dim_out);

    context->SetOutputDim("Out", ddim_out);
    context->ShareLoD("X", "Out");
  }

  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type =
        OperatorWithKernel::IndicateOrPromoteVarDataTypes(ctx, "X", "Y");
    return phi::KernelKey(input_data_type, ctx.GetPlace());
  }

  phi::KernelKey GetKernelTypeForVar(
      const std::string &var_name,
      const phi::DenseTensor &tensor,
      const phi::KernelKey &expected_kernel_type) const override {
    if (framework::IsComplexType(expected_kernel_type.dtype())) {
      // only promote inputs’s types when contains complex input
      return phi::KernelKey(tensor.place(), tensor.layout(), tensor.dtype());
    } else {
#ifdef PADDLE_WITH_DNNL
      // When matmul is first oneDNN op in a chain (there was some non oneDNN op
      // previously)
      // then we also need to rotate shape NHWC -> NCWH
      if ((expected_kernel_type.layout() == phi::DataLayout::ONEDNN) &&
          (tensor.layout() != phi::DataLayout::ONEDNN) &&
          phi::OneDNNContext::tls().get_cur_paddle_data_layout() ==
              phi::DataLayout::kNHWC) {
        return phi::KernelKey(tensor.place(),
                              phi::DataLayout::kNHWC,
                              expected_kernel_type.dtype());
      }
#endif
      return phi::KernelKey(
          tensor.place(), tensor.layout(), expected_kernel_type.dtype());
    }
  }
};

class MatMulOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The first input of MatMul op");
    AddInput("Y", "The second input of MatMul op");
    AddOutput("Out", "The output of MatMul op");
    AddAttr<bool>("transpose_X",
                  R"DOC(If true, use the transpose of `X`.
        )DOC")
        .SetDefault(false);
    AddAttr<bool>("transpose_Y",
                  R"DOC(If true, use the transpose of `Y`.
        )DOC")
        .SetDefault(false);
    AddAttr<float>("alpha", "The scale of Out").SetDefault(1.0f);
    AddAttr<bool>(
        "use_mkldnn",
        "(bool, default false) Indicates if MKL-DNN kernel will be used")
        .SetDefault(false)
        .AsExtra();
    AddAttr<bool>(
        "use_quantizer",
        "(bool, default false) "
        "This parameter is no longer used. Use 'mkldnn_data_type' instead.")
        .SetDefault(false)
        .AsExtra();
    AddAttr<std::string>(
        "mkldnn_data_type",
        "(string, default \"float32\"). Data type of mkldnn kernel")
        .SetDefault("float32")
        .InEnum({"float32", "int8", "bfloat16"})
        .AsExtra();
    /* int8 parameters */
    AddAttr<float>("Scale_x",
                   "(float, default 1.0f), The quantize scale of X tensor")
        .SetDefault(1.0f)
        .AsExtra();
    AddAttr<float>("Scale_y",
                   "(float, default 1.0f), The quantize scale of Y tensor")
        .SetDefault(1.0f)
        .AsExtra();
    AddAttr<float>("Scale_out",
                   "(float, default 1.0f), The quantize scale of output data")
        .SetDefault(1.0f)
        .AsExtra();
    AddAttr<bool>("force_fp32_output",
                  "(bool, default false) Force INT8 kernel output FP32, only "
                  "used in MKL-DNN INT8")
        .SetDefault(false)
        .AsExtra();

#if defined(PADDLE_WITH_MKLML) && !defined(PADDLE_WITH_CUDA) && \
    !defined(PADDLE_WITH_HIP)
    AddAttr<int>("head_number", "The number of heads of the matrix")
        .SetDefault(1);
#endif
    AddComment(R"DOC(
MatMul Operator.
This operator is used to perform (batched) matrix multiplication
over the last two dimensions of the input tensors `X` and `Y`.
If a transpose flag is specified, the last two dimensions of the
tensor are transposed. If the tensor is rank-1 of shape [D], then
for `X` it is treated as [1, D] in nontransposed form and as [D, 1]
in transposed form, whereas for `Y` it is the opposite: It is treated
as [D, 1] in nontransposed form and as [1, D] in transposed form.
Examples without transpose:
- X: [K], Y: [K] => Out: [1]
- X: [K], Y: [K, N] => Out: [N]
- X: [B, M, K], Y: [K] => Out: [B, M]
- X: [M, K], Y: [B, K, N] => Out: [B, M, N]
- X: [B, M, K], Y: [B, K, N] => Out: [B, M, N]
- X: [B, ..., M, K], Y: [B, ..., K, N] => Out: [B, ..., M, N]
Example of matrix multiplication with head_number of H
- X: [B, M, K], Y: [B, K, N] => Out: [B, M, H * N]
The behavior is designed to be similar to the `numpy.matmul` function.
The differences are:
- When the rank of the input data is less than or equal to 3, it
  is similar to the `numpy.matmul` function.
- When the rank of the input is greater than 3, the rank of X and
  Y must be equal, and the first `rank - 2` dimensions must be equal.
- We add `transpose_X` and `transpose_Y` flags.
- We add `head_number` attribute, which is used to multiple two matrixes head
  by head, and eventually concatenates the output of several (head_number)
  small matrixes multiplication.
Both the input `X` and `Y` can carry the LoD (Level of Details) information,
or not. But the output only shares the LoD information with input `X`.
)DOC");
  }
};

class MatMulOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", "matmul");
    OP_INOUT_CHECK(context->HasInput("Y"), "Input", "Y", "matmul");
    OP_INOUT_CHECK(context->HasInput(framework::GradVarName("Out")),
                   "Input",
                   "Out@GRAD",
                   "matmul");
    auto x_dims = context->GetInputDim("X");
    auto y_dims = context->GetInputDim("Y");

    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");

    if (context->HasOutput(x_grad_name)) {
      context->SetOutputDim(x_grad_name, x_dims);
    }
    if (context->HasOutput(y_grad_name)) {
      context->SetOutputDim(y_grad_name, y_dims);
    }
  }

  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type =
        OperatorWithKernel::IndicateOrPromoteVarDataTypes(ctx, "X", "Y");
    return phi::KernelKey(input_data_type, ctx.GetPlace());
  }
};

template <typename T>
class MatMulOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("matmul_grad");
    retv->SetInput("X", this->Input("X"));
    retv->SetInput("Y", this->Input("Y"));
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    retv->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    retv->SetAttrMap(this->Attrs());
  }
};

class MatMulOpDoubleGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", "matmul");
    OP_INOUT_CHECK(context->HasInput("Y"), "Input", "Y", "matmul");
    OP_INOUT_CHECK(context->HasInput("DOut"), "Input", "DOut", "matmul");

    if (context->HasOutput("DX") && context->HasInput("DDY")) {
      context->ShareDim("X", "DX");
    }

    if (context->HasOutput("DY") && context->HasInput("DDX")) {
      context->ShareDim("Y", "DY");
    }

    if (context->HasOutput("DDOut") &&
        (context->HasInput("DDY") || context->HasInput("DDX"))) {
      context->ShareDim("DOut", "DDOut");
    }
  }
};

template <typename T>
class MatMulOpDoubleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("matmul_grad_grad");
    retv->SetInput("X", this->Input("X"));
    retv->SetInput("Y", this->Input("Y"));
    retv->SetInput("DOut", this->Input(framework::GradVarName("Out")));
    retv->SetInput("DDX", this->OutputGrad(framework::GradVarName("X")));
    retv->SetInput("DDY", this->OutputGrad(framework::GradVarName("Y")));

    auto ddx = this->OutputGrad(framework::GradVarName("X"));
    auto ddy = this->OutputGrad(framework::GradVarName("Y"));

    if (!ddx.empty() || !ddy.empty()) {
      retv->SetOutput("DDOut", this->InputGrad(framework::GradVarName("Out")));
    }
    retv->SetOutput(
        "DX", ddy.empty() ? this->EmptyInputGrad() : this->InputGrad("X"));
    retv->SetOutput(
        "DY", ddx.empty() ? this->EmptyInputGrad() : this->InputGrad("Y"));

    retv->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(matmul,
                  ops::MatMulOp,
                  ops::MatMulOpMaker,
                  ops::MatMulOpGradMaker<paddle::framework::OpDesc>,
                  ops::MatMulOpGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(matmul_grad,
                  ops::MatMulOpGrad,
                  ops::MatMulOpDoubleGradMaker<paddle::framework::OpDesc>,
                  ops::MatMulOpDoubleGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(matmul_grad_grad, ops::MatMulOpDoubleGrad);

REGISTER_OP_VERSION(matmul).AddCheckpoint(
    R"ROC(Register matmul for adding the attribute of
       fused_reshape_Y)ROC",
    paddle::framework::compatible::OpVersionDesc().NewAttr(
        "fused_reshape_Y",
        "In order to support the function of fused the input Y "
        " and input X into the input X when "
        "using the operator of matmul, and get raw shape of input Y.",
        std::vector<int>{}));
