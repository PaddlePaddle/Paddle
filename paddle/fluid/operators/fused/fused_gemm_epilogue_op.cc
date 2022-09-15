/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Copyright (c) 2022 NVIDIA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/fused/fused_gemm_epilogue_op.h"

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

class FusedGemmEpilogueOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "FusedGemmEpilogueOp");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "FusedGemmEpilogueOp");
    OP_INOUT_CHECK(
        ctx->HasInput("Bias"), "Output", "Bias", "FusedGemmEpilogueOp");
    OP_INOUT_CHECK(
        ctx->HasOutput("Out"), "Output", "Out", "FusedGemmEpilogueOp");

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");
    auto bias_dims = ctx->GetInputDim("Bias");

    auto trans_x = ctx->Attrs().Get<bool>("trans_x");
    auto trans_y = ctx->Attrs().Get<bool>("trans_y");

    PADDLE_ENFORCE_EQ(
        y_dims.size(),
        2,
        platform::errors::InvalidArgument(
            "The Input tensor Y's dimension of FusedGemmEpilogueOp "
            " should be 2, but got %d.",
            y_dims.size()));

    PADDLE_ENFORCE_GE(
        x_dims.size(),
        2,
        platform::errors::InvalidArgument(
            "The Input tensor X's dimension of FusedGemmEpilogueOp "
            " should be >= 2, but got %d.",
            x_dims.size()));

    PADDLE_ENFORCE_EQ(
        bias_dims.size(),
        1,
        platform::errors::InvalidArgument(
            "The Input tensor bias's dimension of FusedGemmEpilogueOp "
            " should be == 1, but got %d.",
            bias_dims.size()));

    PADDLE_ENFORCE_EQ(bias_dims[0],
                      trans_y ? y_dims[0] : y_dims[1],
                      platform::errors::InvalidArgument(
                          "The Input tensor bias's dimension 0"
                          " should be == Y[-1], but got bias's shape = [%s] "
                          "and Y's shape = [%s]",
                          bias_dims,
                          y_dims));

    auto x_mat_dims =
        phi::flatten_to_2d(x_dims, trans_x ? 1 : x_dims.size() - 1);

    int K_from_x = trans_x ? x_mat_dims[0] : x_mat_dims[1];
    int K_from_y = trans_y ? y_dims[1] : y_dims[0];

    PADDLE_ENFORCE_EQ(
        K_from_x,
        K_from_y,
        platform::errors::InvalidArgument(
            "The last dimension of X should be equal with Y's first dimension."
            "But received X[-1] = [%d], Y[0] = [%d].",
            K_from_x,
            K_from_y));

    auto activation = ctx->Attrs().Get<std::string>("activation");

    if ((activation != "relu") && (activation != "gelu") &&
        (activation != "none")) {
      PADDLE_ENFORCE_EQ(
          true,
          false,
          platform::errors::InvalidArgument(
              "The activation attribute of fused_gemm_epilogue op should be"
              " one of {\"none\", \"relu\", \"gelu\"}. But received %s."
              "But received activation=%s.",
              activation));
    }

    if (activation == "none" && ctx->HasOutput("ReserveSpace")) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "The ReserveSpace would not be used when activation = \"none\""));
    }

    // cublasLt's restriction for auxiliary.
    if (ctx->HasOutput("ReserveSpace") && activation != "none") {
      int min_size_of_n = activation == "relu" ? 128 : 8;
      int N_size = trans_y ? y_dims[0] : y_dims[1];
      PADDLE_ENFORCE_EQ(N_size % min_size_of_n,
                        0,
                        platform::errors::InvalidArgument(
                            "The output dimension N (X(MxK) * Y(KxN) = C(MxN)) "
                            "should be multiple of %d when auxiliary_key given "
                            "and activation=%s, but got N = %d.",
                            min_size_of_n,
                            activation,
                            N_size));
    }

    std::vector<int64_t> out_dims;
    out_dims.reserve(static_cast<size_t>(x_dims.size()));
    if (trans_x) {
      for (int i = 1; i < x_dims.size(); ++i) out_dims.push_back(x_dims[i]);
    } else {
      for (int i = 0; i < x_dims.size() - 1; ++i) out_dims.push_back(x_dims[i]);
    }

    if (trans_y) {
      out_dims.push_back(y_dims[0]);
    } else {
      out_dims.push_back(y_dims[1]);
    }

    ctx->SetOutputDim("Out", phi::make_ddim(out_dims));
    // Note (Ming Huang): Reserve space of relu is a bit-mask,
    // which cannot pass nan_and_inf checking if shape is set.
    if (activation == "gelu" && ctx->HasOutput("ReserveSpace")) {
      ctx->SetOutputDim("ReserveSpace", phi::make_ddim(out_dims));
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.GetPlace(), layout, library);
  }
};

class FusedGemmEpilogueOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor X of Out = Act((X * Y) + Bias).");
    AddInput("Y", "The input tensor Y of Out = Act((X * Y) + Bias).");
    AddInput("Bias", "The input tensor bias of Out = Act((X * Y) + Bias).");

    AddOutput("Out", "The output tensor Out of Out = Act((X * Y) + Bias).");
    AddOutput("ReserveSpace",
              R"DOC(Reserve GPU space to place
        auxiliary data pointer. It is used to pass auxiliary data pointer
        for fused_gemm_epilogue op. If not given (empty string), the
        auxiliary mode would not be enable.)DOC")
        .AsDispensable()
        .AsExtra();

    AddAttr<bool>(
        "trans_x",
        R"DOC((bool, default false), Whether to transpose input tensor X
    or not. The input tensor X coulbe be more than two dimension. When
    set trans_x=true, it would fully reverse X. For instant: X with shpae
    [d0, d1, d2, d3] -> [d3, d2, d1, d0].)DOC")
        .SetDefault(false);
    AddAttr<bool>(
        "trans_y",
        R"DOC((bool, default false), Whether to transpose input tensor Y
    or not. The input tensor Y should be two dimension. When
    set trans_y=true, it would transpose Y. For instant: Y with shpae
    [d0, d1] -> [d1, d0].)DOC")
        .SetDefault(false);

    AddAttr<std::string>(
        "activation",
        R"DOC((string, default none), The activation function. It could be
    one of {none, relu, gelu}. When none is given, Act would be null
    operations)DOC")
        .SetDefault("none");

    AddComment(R"DOC(
FusedGemmEpilogue Operator
This operator is used to perform Activeation(Elementwise_add(Matmul(X, Y), bias)).
It is equal to paddle.nn.Linear + Activation (None, ReLU or GeLU).

Note:
X could be more than two dimension and would be flatten to 2D for computing.
X with shape [d0, d1, d2, d3] -> X_2D with shape [d0*d1*d2, d3]
)DOC");
  }
};

class FusedGemmEpilogueGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(
        ctx->HasInput("DOut"), "Input", "DOut", "FusedGemmEpilogueGradOp");
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "FusedGemmEpilogueGradOp");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "FusedGemmEpilogueGradOp");
    OP_INOUT_CHECK(ctx->HasOutput("DY"), "Output", "DY", "FusedGemmEpilogueOp");

    auto dout_dims = ctx->GetInputDim("DOut");
    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");

    auto trans_x = ctx->Attrs().Get<bool>("trans_x");
    auto trans_y = ctx->Attrs().Get<bool>("trans_y");

    PADDLE_ENFORCE_GE(
        dout_dims.size(),
        2,
        platform::errors::InvalidArgument(
            "The Input tensor DOut's dimension of FusedGemmEpilogueGradOp "
            " should be >= 2, but got %d.",
            dout_dims.size()));

    PADDLE_ENFORCE_EQ(
        y_dims.size(),
        2,
        platform::errors::InvalidArgument(
            "The Input tensor Y's dimension of FusedGemmEpilogueGradOp "
            " should be 2, but got %d.",
            y_dims.size()));

    PADDLE_ENFORCE_GE(
        x_dims.size(),
        2,
        platform::errors::InvalidArgument(
            "The Input tensor X's dimension of FusedGemmEpilogueGradOp "
            " should be >= 2, but got %d.",
            x_dims.size()));

    PADDLE_ENFORCE_EQ(
        dout_dims.size(),
        x_dims.size(),
        platform::errors::InvalidArgument(
            "The Input tensor DOut's and X's dimension of "
            "FusedGemmEpilogueGradOp "
            " should be the same, but got DOut's dim = %d and X's = %d.",
            dout_dims.size(),
            x_dims.size()));

    auto dout_mat_dims = phi::flatten_to_2d(dout_dims, dout_dims.size() - 1);

    auto x_mat_dims = phi::flatten_to_2d(x_dims, x_dims.size() - 1);

    PADDLE_ENFORCE_EQ(
        dout_mat_dims[1],
        trans_y ? y_dims[0] : y_dims[1],
        platform::errors::InvalidArgument(
            "The last dimension of DOut should be equal with Y's last"
            "dimension. But received DOut[-1] = [%d], Y[1] = [%d].",
            dout_mat_dims[1],
            y_dims[1]));

    PADDLE_ENFORCE_EQ(
        dout_mat_dims[0],
        trans_x ? x_mat_dims[1] : x_mat_dims[0],
        platform::errors::InvalidArgument(
            "The first dimension of DOut should be equal with X's first"
            "dimension. But received DOut[0] = [%d], Y[0] = [%d].",
            dout_mat_dims[0],
            x_mat_dims[0]));

    auto activation_grad = ctx->Attrs().Get<std::string>("activation_grad");
    if ((activation_grad != "relu_grad") && (activation_grad != "gelu_grad") &&
        (activation_grad != "none")) {
      PADDLE_ENFORCE_EQ(
          true,
          false,
          platform::errors::InvalidArgument(
              "The activation attribute of fused_gemm_epilogue op should be"
              " one of {\"none\", \"relu\", \"gelu\"}. But received %s."
              "But received activation=%s.",
              activation_grad));
    }

    if (activation_grad != "none" && !ctx->HasInput("ReserveSpace")) {
      PADDLE_ENFORCE_EQ(true,
                        false,
                        platform::errors::InvalidArgument(
                            "The ReserveSpace should not be empty. "
                            "when activation_grad == {relu_grad, gelu_grad}."));
    }

    if (ctx->HasOutput("DX")) {
      std::vector<int64_t> dx_dims;
      dx_dims.reserve(static_cast<size_t>(x_dims.size()));
      for (int i = 0; i < x_dims.size(); ++i) {
        dx_dims.push_back(x_dims[i]);
      }
      ctx->SetOutputDim("DX", phi::make_ddim(dx_dims));
    }

    std::vector<int64_t> dy_dims(y_dims.Get(), y_dims.Get() + y_dims.size());
    ctx->SetOutputDim("DY", phi::make_ddim(dy_dims));

    if (ctx->HasOutput("DBias")) {
      std::vector<int64_t> dbias_dims;
      dbias_dims.push_back(trans_y ? y_dims[0] : y_dims[1]);
      ctx->SetOutputDim("DBias", phi::make_ddim(dbias_dims));
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "DOut");
    return framework::OpKernelType(data_type, ctx.GetPlace(), layout, library);
  }
};

class FusedGemmEpilogueGradOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("DOut",
             "The input grad tensor to Out of Out = (Act(X) * Y) + bias");
    AddInput("X", "The input tensor X of Out = (Act(X) * Y) + bias");
    AddInput("Y", "The input tensor Y of Out = (Act(X) * Y) + bias");
    AddInput("ReserveSpace",
             R"DOC(A GPU space to fetch
        auxiliary data pointer. It is used to pass auxiliary data pointer
        for fused_gemm_epilogue_grad op. If not given (empty string), the
        auxiliary mode would not be enable.)DOC")
        .AsDispensable();

    AddOutput("DX", "The output grad tensor to X of Out = (Act(X) * Y) + bias.")
        .AsDispensable();
    AddOutput("DY",
              "The output grad tensor to Y of Out = (Act(X) * Y) + bias.");
    AddOutput("DBias",
              "The output grad tensor to bias of Out = (Act(X) * Y) + bias.")
        .AsDispensable();
    AddAttr<bool>(
        "trans_x",
        R"DOC((bool, default false), Whether to transpose input tensor X
    or not. The input tensor X coulbe be more than two dimension. When
    set trans_x=true, it would fully reverse X. For instant: X with shpae
    [d0, d1, d2, d3] -> [d3, d2, d1, d0].)DOC")
        .SetDefault(false);
    AddAttr<bool>(
        "trans_y",
        R"DOC((bool, default false), Whether to transpose input tensor Y
    or not. The input tensor Y should be two dimension. When
    set trans_y=true, it would transpose Y. For instant: Y with shpae
    [d0, d1] -> [d1, d0].)DOC")
        .SetDefault(false);

    AddAttr<std::string>(
        "activation_grad",
        R"DOC((string, default none), The backward activation function. It could be
    one of {none, relu_grad, gelu_grad}. When none is given, The backward Act would
    be null operations)DOC")
        .SetDefault("none");

    AddComment(R"DOC(
FusedGemmEpilogueGrad Operator
This operator is used to perform backward of Elementwise_add(Matmul(Activeation(X), Y), bias).
It is equal to Activation (None, ReLU or GeLU) + paddle.nn.Linear.

Note:
X could be more than two dimension and would be flatten to 2D for computing.
X with shape [d0, d1, d2, d3] -> X_2D with shape [d0*d1*d2, d3]
)DOC");
  }
};

template <typename T>
class FusedGemmEpilogueOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    const auto& act_type = this->template Attr<std::string>("activation");

    op->SetType(this->ForwardOpType() + "_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Input("Y"));
    if (act_type != "none") {
      op->SetInput("ReserveSpace", this->Input("ReserveSpace"));
    }
    op->SetInput("DOut", this->OutputGrad("Out"));

    op->SetOutput("DX", this->InputGrad("X"));
    op->SetOutput("DY", this->InputGrad("Y"));
    op->SetOutput("DBias", this->InputGrad("Bias"));

    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    fused_gemm_epilogue,
    ops::FusedGemmEpilogueOp,
    ops::FusedGemmEpilogueOpMaker,
    ops::FusedGemmEpilogueOpGradMaker<paddle::framework::OpDesc>,
    ops::FusedGemmEpilogueOpGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(fused_gemm_epilogue_grad,
                  ops::FusedGemmEpilogueGradOp,
                  ops::FusedGemmEpilogueGradOpMaker);
