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
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of AffineChannelOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Scale"),
                   "Input(Scale) of AffineChannelOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Bias"),
                   "Input(Bias) of AffineChannelOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of AffineChannelOp should not be null.");

    auto x_dims = ctx->GetInputDim("X");
    auto scale_dims = ctx->GetInputDim("Scale");
    auto b_dims = ctx->GetInputDim("Bias");
    const framework::DataLayout data_layout = framework::StringToDataLayout(
        ctx->Attrs().Get<std::string>("data_layout"));

    const int64_t C = (data_layout == framework::DataLayout::kNCHW
                           ? x_dims[1]
                           : x_dims[x_dims.size() - 1]);

    PADDLE_ENFORCE_EQ(scale_dims.size(), 1UL);
    PADDLE_ENFORCE_EQ(b_dims.size(), 1UL);
    if (ctx->IsRuntime() || scale_dims[0] > 0) {
      PADDLE_ENFORCE_EQ(scale_dims[0], C);
    }
    if (ctx->IsRuntime() || b_dims[0] > 0) {
      PADDLE_ENFORCE_EQ(b_dims[0], C);
    }

    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->ShareLoD("X", "Out");
  }
};

class AffineChannelOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null.");
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      PADDLE_ENFORCE(ctx->HasInput("Scale"),
                     "Input(Scale) should not be null.");
      ctx->SetOutputDim(framework::GradVarName("X"),
                        ctx->GetInputDim(framework::GradVarName("Out")));
    }
    if (ctx->HasOutput(framework::GradVarName("Scale"))) {
      // Scale@GRAD and Bias@GRAD must exist at the same time.
      PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("Bias")),
                     "Output(Scale@GRAD) should not be null.");
      PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null.");
      ctx->SetOutputDim(framework::GradVarName("Scale"),
                        ctx->GetInputDim("Scale"));
      ctx->SetOutputDim(framework::GradVarName("Bias"),
                        ctx->GetInputDim("Scale"));
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"))->type(),
        ctx.GetPlace());
  }
};

class AffineChannelGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto* op = new framework::OpDesc();
    op->SetType("affine_channel_grad");
    op->SetInput("X", Input("X"));
    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    op->SetInput("Scale", Input("Scale"));

    op->SetAttrMap(Attrs());

    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op->SetOutput(framework::GradVarName("Scale"), InputGrad("Scale"));
    op->SetOutput(framework::GradVarName("Bias"), InputGrad("Bias"));

    return std::unique_ptr<framework::OpDesc>(op);
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
    auto* x = ctx.Input<framework::Tensor>("X");
    auto* scale = ctx.Input<framework::Tensor>("Scale");
    auto* bias = ctx.Input<framework::Tensor>("Bias");

    auto* y = ctx.Output<framework::Tensor>("Out");
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
    auto* x = ctx.Input<framework::Tensor>("X");
    auto* scale = ctx.Input<framework::Tensor>("Scale");
    auto* dy = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));

    auto* dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* dscale =
        ctx.Output<framework::Tensor>(framework::GradVarName("Scale"));
    auto* dbias = ctx.Output<framework::Tensor>(framework::GradVarName("Bias"));

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
      // compute dx
      int stride = C * HxW;
      if (dx) {
        for (int i = 0; i < N; i++) {
          ConstEigenArrayMap<T> dy_e(dy_d, HxW, C);
          EigenArrayMap<T> dx_e(dx_d, HxW, C);
          dx_e = dy_e.rowwise() * scale_e.transpose();
          dy_d += stride;
          dx_d += stride;
        }
      }
      // compute dscale and dbias
      if (dscale && dbias) {
        auto* x_d = x->data<T>();
        dy_d = dy->data<T>();
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
    } else {
      int num = N * HxW;
      ConstEigenArrayMap<T> dy_e(dy_d, C, num);
      // compute dx
      if (dx) {
        EigenArrayMap<T> dx_e(dx_d, C, num);
        dx_e = dy_e.colwise() * scale_e;
      }
      // compute dscale and dbias
      if (dscale && dbias) {
        auto* x_d = x->data<T>();
        ConstEigenArrayMap<T> x_e(x_d, C, num);
        dscale_e = (x_e * dy_e).rowwise().sum();
        dbias_e = dy_e.rowwise().sum();
      }
    }
  }
};

class AffineChannelNoNeedBufferVarsInference
    : public framework::NoNeedBufferVarsInference {
 public:
  using framework::NoNeedBufferVarsInference::NoNeedBufferVarsInference;

 private:
  inline bool HasInput(const std::string& name) const {
    auto& inputs = Inputs();
    auto iter = inputs.find(name);
    if (iter == inputs.end() || iter->second.empty()) {
      return false;
    } else {
      return iter->second[0] != framework::kEmptyVarName;
    }
  }

 public:
  std::unordered_set<std::string> operator()() const {
    if (!HasInput(framework::GradVarName("Scale")) &&
        !HasInput(framework::GradVarName("Bias"))) {
      return {"X"};
    } else {
      return {};
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;

REGISTER_OPERATOR(affine_channel, ops::AffineChannelOp,
                  ops::AffineChannelOpMaker, ops::AffineChannelGradMaker);
REGISTER_OPERATOR(affine_channel_grad, ops::AffineChannelOpGrad,
                  ops::AffineChannelNoNeedBufferVarsInference);

REGISTER_OP_CPU_KERNEL(affine_channel, ops::AffineChannelKernel<CPU, float>,
                       ops::AffineChannelKernel<CPU, double>);
REGISTER_OP_CPU_KERNEL(affine_channel_grad,
                       ops::AffineChannelGradKernel<CPU, float>,
                       ops::AffineChannelGradKernel<CPU, double>);
