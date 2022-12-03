//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/inplace_abn_op.h"

#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/operators/batch_norm_op.h"
#include "paddle/phi/kernels/batch_norm_grad_kernel.h"
#include "paddle/phi/kernels/batch_norm_kernel.h"

namespace paddle {
namespace operators {

class InplaceABNOp : public paddle::operators::BatchNormOp {
 public:
  using paddle::operators::BatchNormOp::BatchNormOp;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    // By default, the type of the scale, bias, mean,
    // and var tensors should both be float. (For float or float16 input tensor)
    // or double (For double input tensor).
    auto bn_param_type = framework::proto::VarType::FP32;
    if (input_data_type == framework::proto::VarType::FP64) {
      bn_param_type = framework::proto::VarType::FP64;
    }
    PADDLE_ENFORCE_EQ(bn_param_type,
                      framework::TransToProtoVarType(
                          ctx.Input<phi::DenseTensor>("Scale")->dtype()),
                      platform::errors::InvalidArgument(
                          "Scale input should be of float type"));
    PADDLE_ENFORCE_EQ(bn_param_type,
                      framework::TransToProtoVarType(
                          ctx.Input<phi::DenseTensor>("Bias")->dtype()),
                      platform::errors::InvalidArgument(
                          "Bias input should be of float type"));
    PADDLE_ENFORCE_EQ(bn_param_type,
                      framework::TransToProtoVarType(
                          ctx.Input<phi::DenseTensor>("Mean")->dtype()),
                      platform::errors::InvalidArgument(
                          "Mean input should be of float type"));
    PADDLE_ENFORCE_EQ(bn_param_type,
                      framework::TransToProtoVarType(
                          ctx.Input<phi::DenseTensor>("Variance")->dtype()),
                      platform::errors::InvalidArgument(
                          "Variance input should be of float type"));

    framework::LibraryType library = framework::LibraryType::kPlain;
    phi::DataLayout layout = phi::DataLayout::kAnyLayout;

    return framework::OpKernelType(
        input_data_type, ctx.GetPlace(), layout, library);
  }
};

class InplaceABNGradOp : public paddle::operators::BatchNormGradOp {
 public:
  using paddle::operators::BatchNormGradOp::BatchNormGradOp;

  void InferShape(framework::InferShapeContext* ctx) const override {
    // check input
    OP_INOUT_CHECK(ctx->HasInput("Scale"), "Input", "Scale", "InplaceABNGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Y")),
                   "Input",
                   "Y@GRAD",
                   "InplaceABNGrad");
    OP_INOUT_CHECK(
        ctx->HasInput("SavedMean"), "Input", "SavedMean", "InplaceABNGrad");
    OP_INOUT_CHECK(ctx->HasInput("SavedVariance"),
                   "Input",
                   "SavedVariance",
                   "InplaceABNGrad");

    // check output
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")),
                   "Output",
                   "X@GRAD",
                   "InplaceABNGrad");

    const bool has_scale_grad = ctx->HasOutput(framework::GradVarName("Scale"));
    const bool has_bias_grad = ctx->HasOutput(framework::GradVarName("Bias"));

    PADDLE_ENFORCE_EQ(
        has_scale_grad,
        has_bias_grad,
        platform::errors::InvalidArgument(
            "Output(Scale@GRAD) and Output(Bias@GRAD) must be null "
            "or not be null at same time. But now, "
            "has Scale@Grad=[%d], has Bias@GRAD=[%d]",
            has_scale_grad,
            has_bias_grad));

    const bool use_global_stats = ctx->Attrs().Get<bool>("use_global_stats");
    if (use_global_stats) {
      PADDLE_ENFORCE_EQ(
          !ctx->Attrs().Get<bool>("use_mkldnn"),
          true,
          platform::errors::InvalidArgument(
              "Using global stats during training is not supported "
              "in oneDNN version of batch_norm_gradient kernel now."));
    }

    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "InplaceABNGrad");
    const auto y_dims = ctx->GetInputDim("Y");
    const DataLayout data_layout =
        phi::StringToDataLayout(ctx->Attrs().Get<std::string>("data_layout"));

    const int C = ((ctx->IsRunMKLDNNKernel() == true) ||
                           (data_layout == DataLayout::kNCHW)
                       ? y_dims[1]
                       : y_dims[y_dims.size() - 1]);

    ctx->SetOutputDim(framework::GradVarName("X"), y_dims);
    // has_scale_grad == has_bias_grad, judge has_scale_grad is enough
    if (has_scale_grad) {
      ctx->SetOutputDim(framework::GradVarName("Scale"), {C});
      ctx->SetOutputDim(framework::GradVarName("Bias"), {C});
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    const auto* var = ctx.InputVar(framework::GradVarName("Y"));
    auto input_data_type = framework::TransToProtoVarType(
        ctx.Input<phi::DenseTensor>("Y")->dtype());
    if (var == nullptr) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "can't find gradient variable of Y"));
    }
    const phi::DenseTensor* t = nullptr;
    if (var->IsType<phi::DenseTensor>()) {
      t = &var->Get<phi::DenseTensor>();
    } else if (var->IsType<phi::DenseTensor>()) {
      t = &var->Get<phi::DenseTensor>();
    }
    if (t == nullptr) {
      PADDLE_THROW(
          platform::errors::InvalidArgument("gradient variable of Y is empty"));
    }
    framework::LibraryType library = framework::LibraryType::kPlain;
    phi::DataLayout layout = phi::DataLayout::kAnyLayout;

    return framework::OpKernelType(
        input_data_type, ctx.GetPlace(), layout, library);
  }
};

class InplaceABNOpMaker : public paddle::operators::BatchNormOpMaker {
 public:
  void Make() override {
    BatchNormOpMaker::Make();
    AddAttr<std::string>(
        "activation",
        "(enum string, default identity, can be identity|elu|leaky-relu) "
        "The activation type used for output candidate {h}_t.")
        .SetDefault("");
    AddAttr<float>("alpha",
                   "(float, default 1.0) Only used in inplace-abn kernel,"
                   "the activation type(identity|elu|leakyrelu) would be fused "
                   "with batch_norm, "
                   "this is the alpha value for elu|leakyrelu.")
        .SetDefault(0.1f);
    AddAttr<bool>("use_sync_bn",
                  "(bool, default false) Whether use synchronize batch "
                  "normalization.")
        .SetDefault(false);
  }
};

template <typename T>
class InplaceABNOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType(this->ForwardOpType() + "_grad");
    op->SetInput("Y", this->Output("Y"));
    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));

    op->SetInput("Scale", this->Input("Scale"));
    op->SetInput("Bias", this->Input("Bias"));
    op->SetInput("SavedMean", this->Output("SavedMean"));
    op->SetInput("SavedVariance", this->Output("SavedVariance"));
    if (this->HasOutput("ReserveSpace")) {
      op->SetInput("ReserveSpace", this->Output("ReserveSpace"));
    }

    // used when setting use_global_stats True during training
    if (PADDLE_GET_CONST(bool, this->GetAttr("use_global_stats"))) {
      op->SetInput("Mean", this->Output("MeanOut"));
      op->SetInput("Variance", this->Output("VarianceOut"));
    }

    op->SetAttrMap(this->Attrs());

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Scale"), this->InputGrad("Scale"));
    op->SetOutput(framework::GradVarName("Bias"), this->InputGrad("Bias"));
  }
};

template <typename DeviceContext, typename T>
class InplaceABNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* y = ctx.Output<phi::DenseTensor>("Y");
    PADDLE_ENFORCE_EQ(x,
                      y,
                      platform::errors::InvalidArgument(
                          "X and Y not inplaced in inplace mode"));
    auto activation =
        GetInplaceABNActivationType(ctx.Attr<std::string>("activation"));
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();

    auto* scale = ctx.Input<phi::DenseTensor>("Scale");
    auto* bias = ctx.Input<phi::DenseTensor>("Bias");
    auto* mean = ctx.Input<phi::DenseTensor>("Mean");
    auto* variance = ctx.Input<phi::DenseTensor>("Variance");

    auto momentum = ctx.Attr<float>("momentum");
    auto epsilon = ctx.Attr<float>("epsilon");
    auto data_layout = ctx.Attr<std::string>("data_layout");
    auto is_test = ctx.Attr<bool>("is_test");
    auto use_global_stats = ctx.Attr<bool>("use_global_stats");
    auto trainable_statistics = ctx.Attr<bool>("trainable_statistics");

    auto* mean_out = ctx.Output<phi::DenseTensor>("MeanOut");
    auto* variance_out = ctx.Output<phi::DenseTensor>("VarianceOut");
    auto* saved_mean = ctx.Output<phi::DenseTensor>("SavedMean");
    auto* saved_variance = ctx.Output<phi::DenseTensor>("SavedVariance");
    auto* reserve_space = ctx.Output<phi::DenseTensor>("ReserveSpace");

    auto& dev_ctx = ctx.device_context<DeviceContext>();
    phi::BatchNormKernel<T>(
        static_cast<const typename framework::ConvertToPhiContext<
            DeviceContext>::TYPE&>(dev_ctx),
        *x,
        *mean,
        *variance,
        *scale,
        *bias,
        is_test,
        momentum,
        epsilon,
        data_layout,
        use_global_stats,
        trainable_statistics,
        y,
        mean_out,
        variance_out,
        saved_mean,
        saved_variance,
        reserve_space);

    auto cur_y = EigenVector<T>::Flatten(*y);
    InplaceABNActivation<DeviceContext, T> functor;
    functor.Compute(ctx, activation, place, cur_y, cur_y);
  }
};

template <typename DeviceContext, typename T>
class InplaceABNGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* y = ctx.Input<phi::DenseTensor>("Y");
    auto* d_y = ctx.Input<phi::DenseTensor>(framework::GradVarName("Y"));
    auto* d_x = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    PADDLE_ENFORCE_EQ(d_x,
                      d_y,
                      platform::errors::InvalidArgument(
                          "X@GRAD and Y@GRAD not inplaced in inplace mode"));
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
    auto activation =
        GetInplaceABNActivationType(ctx.Attr<std::string>("activation"));

    auto py = *y;
    auto pd_y = *d_y;
    auto cur_y = EigenVector<T>::Flatten(py);
    auto cur_dy = EigenVector<T>::Flatten(pd_y);

    InplaceABNActivation<DeviceContext, T> functor;
    functor.GradCompute(ctx, activation, place, cur_y, cur_y, cur_dy, cur_dy);

    // BatchNormGradKernel<DeviceContext, T>::Compute(ctx);

    auto* scale = ctx.Input<phi::DenseTensor>("Scale");
    auto* bias = ctx.Input<phi::DenseTensor>("Bias");
    auto* saved_mean = ctx.Input<phi::DenseTensor>("SavedMean");
    auto* saved_variance = ctx.Input<phi::DenseTensor>("SavedVariance");

    auto momentum = ctx.Attr<float>("momentum");
    auto epsilon = ctx.Attr<float>("epsilon");
    auto data_layout = ctx.Attr<std::string>("data_layout");
    auto is_test = ctx.Attr<bool>("is_test");
    auto use_global_stats = ctx.Attr<bool>("use_global_stats");
    auto trainable_statistics = ctx.Attr<bool>("trainable_statistics");

    auto* scale_grad =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("Scale"));
    auto* bias_grad =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("Bias"));

    auto* reserve_space = ctx.Input<phi::DenseTensor>("ReserveSpace");
    auto* mean = ctx.Input<phi::DenseTensor>("ReserveSpace");
    auto* variance = ctx.Input<phi::DenseTensor>("ReserveSpace");

    paddle::optional<phi::DenseTensor> space_opt;
    paddle::optional<phi::DenseTensor> mean_opt;
    paddle::optional<phi::DenseTensor> variance_opt;

    if (reserve_space != nullptr) {
      space_opt = *reserve_space;
    }

    if (mean != nullptr) {
      mean_opt = *mean;
    }

    if (variance != nullptr) {
      variance_opt = *variance;
    }

    auto& dev_ctx = ctx.device_context<DeviceContext>();
    phi::BatchNormGradRawKernel<T>(
        static_cast<const typename framework::ConvertToPhiContext<
            DeviceContext>::TYPE&>(dev_ctx),
        *y,
        *scale,
        *bias,
        mean_opt,
        variance_opt,
        *saved_mean,
        *saved_variance,
        space_opt,
        *d_y,
        momentum,
        epsilon,
        data_layout,
        is_test,
        use_global_stats,
        trainable_statistics,
        true,
        d_x,
        scale_grad,
        bias_grad);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INPLACE_OP_INFERER(InplaceAbnOpInplaceInferer, {"X", "Y"});
REGISTER_OPERATOR(inplace_abn,
                  ops::InplaceABNOp,
                  ops::InplaceABNOpMaker,
                  ops::BatchNormOpInferVarType,
                  ops::InplaceABNOpGradMaker<paddle::framework::OpDesc>,
                  ops::InplaceABNOpGradMaker<paddle::imperative::OpBase>,
                  InplaceAbnOpInplaceInferer)
REGISTER_OPERATOR(inplace_abn_grad, ops::InplaceABNGradOp)

REGISTER_OP_CPU_KERNEL(inplace_abn,
                       ops::InplaceABNKernel<phi::CPUContext, float>,
                       ops::InplaceABNKernel<phi::CPUContext, double>);
REGISTER_OP_CPU_KERNEL(inplace_abn_grad,
                       ops::InplaceABNGradKernel<phi::CPUContext, float>,
                       ops::InplaceABNGradKernel<phi::CPUContext, double>);
