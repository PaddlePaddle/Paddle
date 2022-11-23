/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
Indicesou may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>
#include <unordered_map>

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class AffineChannelOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor) Feature map input can be a 4D tensor with order NCHW "
             "or NHWC. It also can be a 2D tensor and C is the second "
             "dimension.");
    AddInput("Scale",
             "(Tensor) 1D input of shape (C), the c-th element "
             "is the scale factor of the affine transformation "
             "for the c-th channel of the input.");
    AddInput("Bias",
             "(Tensor) 1D input of shape (C), the c-th element "
             "is the bias of the affine transformation for the "
             "c-th channel of the input.");
    AddAttr<std::string>(
        "data_layout",
        "(string, default NCHW) Only used in "
        "An optional string from: \"NHWC\", \"NCHW\". "
        "Defaults to \"NHWC\". Specify the data format of the output data, "
        "the input will be transformed automatically. ")
        .SetDefault("AnyLayout");
    AddOutput("Out", "(Tensor) A tensor of the same shape and order with X.");
    AddComment(R"DOC(

Applies a separate affine transformation to each channel of the input. Useful
for replacing spatial batch norm with its equivalent fixed transformation.
The input also can be 2D tensor and applies a affine transformation in second
dimension.

$$Out = Scale*X + Bias$$

)DOC");
  }
};

class AffineChannelOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "AffineChannel");
    OP_INOUT_CHECK(ctx->HasInput("Scale"), "Input", "Scale", "AffineChannel");
    OP_INOUT_CHECK(ctx->HasInput("Bias"), "Input", "Bias", "AffineChannel");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "AffineChannel");

    auto x_dims = ctx->GetInputDim("X");
    auto scale_dims = ctx->GetInputDim("Scale");
    auto b_dims = ctx->GetInputDim("Bias");
    const framework::DataLayout data_layout = framework::StringToDataLayout(
        ctx->Attrs().Get<std::string>("data_layout"));

    const int64_t C = (data_layout == framework::DataLayout::kNCHW
                           ? x_dims[1]
                           : x_dims[x_dims.size() - 1]);

    PADDLE_ENFORCE_EQ(
        scale_dims.size(),
        1UL,
        platform::errors::InvalidArgument(
            "The dimensions of Input(Scale) must be 1,"
            "But received the dimensions of Input(Scale) is [%d] ",
            scale_dims.size()));
    PADDLE_ENFORCE_EQ(b_dims.size(),
                      1UL,
                      platform::errors::InvalidArgument(
                          "The dimensions of Input(Bias) must be 1,"
                          "But received the dimensions of Input(Bias) is [%d] ",
                          scale_dims.size()));
    if (ctx->IsRuntime() || scale_dims[0] > 0) {
      PADDLE_ENFORCE_EQ(
          scale_dims[0],
          C,
          platform::errors::InvalidArgument(
              "The first dimension value of Input(Scale) must be [%d],"
              "But received [%d].",
              C,
              scale_dims[0]));
    }
    if (ctx->IsRuntime() || b_dims[0] > 0) {
      PADDLE_ENFORCE_EQ(
          b_dims[0],
          C,
          platform::errors::InvalidArgument(
              "The first dimension value of Input(Bias) must be [%d],"
              "But received [%d].",
              C,
              b_dims[0]));
    }

    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->ShareLoD("X", "Out");
  }
};

class AffineChannelOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")),
                   "Input",
                   framework::GradVarName("Out"),
                   "AffineChannelGrad");
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      OP_INOUT_CHECK(
          ctx->HasInput("Scale"), "Input", "Scale", "AffineChannelGrad");
      ctx->SetOutputDim(framework::GradVarName("X"),
                        ctx->GetInputDim(framework::GradVarName("Out")));
    }
    if (ctx->HasOutput(framework::GradVarName("Scale"))) {
      // Scale@GRAD and Bias@GRAD must exist at the same time.
      OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("Bias")),
                     "Output",
                     framework::GradVarName("Bias"),
                     "AffineChannelGrad");
      OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "AffineChannelGrad");
      ctx->SetOutputDim(framework::GradVarName("Scale"),
                        ctx->GetInputDim("Scale"));
      ctx->SetOutputDim(framework::GradVarName("Bias"),
                        ctx->GetInputDim("Scale"));
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.GetPlace());
  }
};

template <typename T>
class AffineChannelGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> op) const override {
    op->SetType("affine_channel_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetInput("Scale", this->Input("Scale"));

    op->SetAttrMap(this->Attrs());

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Scale"), this->InputGrad("Scale"));
    op->SetOutput(framework::GradVarName("Bias"), this->InputGrad("Bias"));
  }
};

template <typename T>
using EigenArrayMap =
    Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using ConstEigenArrayMap =
    Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using EigenVectorArrayMap = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>;
template <typename T>
using ConstEigenVectorArrayMap =
    Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>;

template <typename DeviceContext, typename T>
class AffineChannelKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* scale = ctx.Input<phi::DenseTensor>("Scale");
    auto* bias = ctx.Input<phi::DenseTensor>("Bias");

    auto* y = ctx.Output<phi::DenseTensor>("Out");
    y->mutable_data<T>(ctx.GetPlace());

    const framework::DataLayout layout =
        framework::StringToDataLayout(ctx.Attr<std::string>("data_layout"));

    auto dims = x->dims();
    int N = dims[0];
    int C = layout == framework::DataLayout::kNCHW ? dims[1]
                                                   : dims[dims.size() - 1];
    int HxW = x->numel() / N / C;

    auto* scale_d = scale->data<T>();
    auto* bias_d = bias->data<T>();
    ConstEigenVectorArrayMap<T> a_e(scale_d, C);
    ConstEigenVectorArrayMap<T> b_e(bias_d, C);

    auto* x_d = x->data<T>();
    auto* y_d = y->data<T>();
    if (layout == framework::DataLayout::kNCHW) {
      int stride = C * HxW;
      for (int i = 0; i < N; i++) {
        ConstEigenArrayMap<T> x_e(x_d, HxW, C);
        EigenArrayMap<T> y_e(y_d, HxW, C);
        y_e = (x_e.rowwise() * a_e.transpose()).rowwise() + b_e.transpose();
        x_d += stride;
        y_d += stride;
      }
    } else {
      int num = N * HxW;
      ConstEigenArrayMap<T> x_e(x_d, C, num);
      EigenArrayMap<T> y_e(y_d, C, num);
      y_e = (x_e.colwise() * a_e).colwise() + b_e;
    }
  }
};

template <typename DeviceContext, typename T>
class AffineChannelGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* scale = ctx.Input<phi::DenseTensor>("Scale");
    auto* dy = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));

    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto* dscale =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("Scale"));
    auto* dbias = ctx.Output<phi::DenseTensor>(framework::GradVarName("Bias"));

    const framework::DataLayout layout =
        framework::StringToDataLayout(ctx.Attr<std::string>("data_layout"));

    auto dims = x->dims();
    int N = dims[0];
    int C = layout == framework::DataLayout::kNCHW ? dims[1]
                                                   : dims[dims.size() - 1];
    int HxW = x->numel() / N / C;

    auto* dy_d = dy->data<T>();
    auto* scale_d = scale->data<T>();
    ConstEigenVectorArrayMap<T> scale_e(scale_d, C);

    T* dx_d = dx ? dx->mutable_data<T>(ctx.GetPlace()) : nullptr;
    T* dscale_d = dscale ? dscale->mutable_data<T>(ctx.GetPlace()) : nullptr;
    T* dbias_d = dbias ? dbias->mutable_data<T>(ctx.GetPlace()) : nullptr;
    EigenVectorArrayMap<T> dscale_e(dscale_d, C);
    EigenVectorArrayMap<T> dbias_e(dbias_d, C);

    if (layout == framework::DataLayout::kNCHW) {
      // compute dscale and dbias
      int stride = C * HxW;
      auto* original_dy_d = dy_d;
      if (dscale && dbias) {
        auto* x_d = x->data<T>();
        for (int i = 0; i < N; i++) {
          ConstEigenArrayMap<T> x_e(x_d, HxW, C);
          ConstEigenArrayMap<T> dy_e(dy_d, HxW, C);
          if (i == 0) {
            dscale_e = (x_e * dy_e).colwise().sum();
          } else {
            dscale_e += (x_e * dy_e).colwise().sum();
          }
          if (i == 0) {
            dbias_e = dy_e.colwise().sum();
          } else {
            dbias_e += dy_e.colwise().sum();
          }
          x_d += stride;
          dy_d += stride;
        }
      }

      // compute dx
      if (dx) {
        dy_d = original_dy_d;
        for (int i = 0; i < N; i++) {
          ConstEigenArrayMap<T> dy_e(dy_d, HxW, C);
          EigenArrayMap<T> dx_e(dx_d, HxW, C);
          dx_e = dy_e.rowwise() * scale_e.transpose();
          dy_d += stride;
          dx_d += stride;
        }
      }
    } else {
      int num = N * HxW;
      ConstEigenArrayMap<T> dy_e(dy_d, C, num);
      // compute dscale and dbias
      if (dscale && dbias) {
        auto* x_d = x->data<T>();
        ConstEigenArrayMap<T> x_e(x_d, C, num);
        dscale_e = (x_e * dy_e).rowwise().sum();
        dbias_e = dy_e.rowwise().sum();
      }

      // compute dx
      if (dx) {
        EigenArrayMap<T> dx_e(dx_d, C, num);
        dx_e = dy_e.colwise() * scale_e;
      }
    }
  }
};

class AffineChannelNoNeedBufferVarsInference
    : public framework::NoNeedBufferVarsInference {
 public:
  using framework::NoNeedBufferVarsInference::NoNeedBufferVarsInference;

  const std::unordered_set<std::string>& operator()(
      const framework::InferNoNeedBufferVarsContext& ctx) const final {
    static const std::unordered_set<std::string> kX({"X"});
    if (!ctx.HasOutput(framework::GradVarName("Scale")) &&
        !ctx.HasOutput(framework::GradVarName("Bias"))) {
      return kX;
    } else {
      return Empty();
    }
  }
};

DECLARE_INPLACE_OP_INFERER(AffineChannelInplaceInferer, {"X", "Out"});
DECLARE_INPLACE_OP_INFERER(AffineChannelGradInplaceInferer,
                           {framework::GradVarName("Out"),
                            framework::GradVarName("X")});

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = phi::CPUContext;

REGISTER_OPERATOR(affine_channel,
                  ops::AffineChannelOp,
                  ops::AffineChannelOpMaker,
                  ops::AffineChannelGradMaker<paddle::framework::OpDesc>,
                  ops::AffineChannelGradMaker<paddle::imperative::OpBase>,
                  ops::AffineChannelInplaceInferer);
REGISTER_OPERATOR(affine_channel_grad,
                  ops::AffineChannelOpGrad,
                  ops::AffineChannelNoNeedBufferVarsInference,
                  ops::AffineChannelGradInplaceInferer);

REGISTER_OP_CPU_KERNEL(affine_channel,
                       ops::AffineChannelKernel<CPU, float>,
                       ops::AffineChannelKernel<CPU, double>);
REGISTER_OP_CPU_KERNEL(affine_channel_grad,
                       ops::AffineChannelGradKernel<CPU, float>,
                       ops::AffineChannelGradKernel<CPU, double>);
