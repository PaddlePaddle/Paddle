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
#include "paddle/phi/kernels/batch_norm_grad_kernel.h"
#include "paddle/phi/kernels/batch_norm_kernel.h"

namespace paddle {
namespace operators {

class InplaceABNOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "BatchNorm");
    OP_INOUT_CHECK(ctx->HasInput("Scale"), "Input", "Scale", "BatchNorm");
    OP_INOUT_CHECK(ctx->HasInput("Bias"), "Input", "Bias", "BatchNorm");
    OP_INOUT_CHECK(ctx->HasInput("Mean"), "Input", "Mean", "BatchNorm");
    OP_INOUT_CHECK(ctx->HasInput("Variance"), "Input", "Variance", "BatchNorm");
    OP_INOUT_CHECK(ctx->HasOutput("Y"), "Output", "Y", "BatchNorm");

    bool is_test = ctx->Attrs().Get<bool>("is_test");
    bool trainable_stats = ctx->Attrs().Get<bool>("trainable_statistics");
    bool test_mode = is_test && (!trainable_stats);
    if (!test_mode) {
      OP_INOUT_CHECK(
          ctx->HasOutput("MeanOut"), "Output", "MeanOut", "BatchNorm");
      OP_INOUT_CHECK(
          ctx->HasOutput("VarianceOut"), "Output", "VarianceOut", "BatchNorm");
      OP_INOUT_CHECK(
          ctx->HasOutput("SavedMean"), "Output", "SavedMean", "BatchNorm");
      OP_INOUT_CHECK(ctx->HasOutput("SavedVariance"),
                     "Output",
                     "SavedVariance",
                     "BatchNorm");
    }

    // make sure Mean/MeanOut and Variance/VarianceOut share memory in Python
    PADDLE_ENFORCE_EQ(ctx->Inputs("Mean")[0],
                      ctx->Outputs("MeanOut")[0],
                      platform::errors::InvalidArgument(
                          "Mean and MeanOut should share the same memory"));
    PADDLE_ENFORCE_EQ(
        ctx->Inputs("Variance")[0],
        ctx->Outputs("VarianceOut")[0],
        platform::errors::InvalidArgument(
            "Variance and VarianceOut should share the same memory"));

    const auto x_dims = ctx->GetInputDim("X");

    for (int i = 0; i < x_dims.size(); i++) {
      PADDLE_ENFORCE_EQ(
          (x_dims[i] == -1) || (x_dims[i] > 0),
          true,
          platform::errors::InvalidArgument(
              "Each dimension of input tensor is expected to be -1 or a "
              "positive number, but received %d. Input's shape is [%s].",
              x_dims[i],
              x_dims));
    }

    const DataLayout data_layout =
        phi::StringToDataLayout(ctx->Attrs().Get<std::string>("data_layout"));

    if (ctx->IsRuntime() && ctx->HasInput("MomentumTensor")) {
      auto mom = ctx->Inputs("MomentumTensor");
      PADDLE_ENFORCE_EQ(mom.size(),
                        1,
                        platform::errors::InvalidArgument(
                            "The input tensor MomentumTensor's size must be 1"
                            "But received: MomentumTensor's size is [%d]",
                            mom.size()));
    }

    PADDLE_ENFORCE_GE(x_dims.size(),
                      2,
                      platform::errors::InvalidArgument(
                          "ShapeError: the dimension of input "
                          "X must greater than or equal to 2. But received: "
                          "the shape of input "
                          "X = [%s], the dimension of input X =[%d]",
                          x_dims,
                          x_dims.size()));
    PADDLE_ENFORCE_LE(x_dims.size(),
                      5,
                      platform::errors::InvalidArgument(
                          "ShapeError: the dimension of input X "
                          "must smaller than or equal to 5. But received: the "
                          "shape of input X "
                          "= [%s], the dimension of input X = [%d]",
                          x_dims,
                          x_dims.size()));
    VLOG(4) << ctx->IsRunMKLDNNKernel();
    VLOG(4) << data_layout;
    const int64_t C = ((ctx->IsRunMKLDNNKernel() == true) ||
                               (data_layout == DataLayout::kNCHW)
                           ? x_dims[1]
                           : x_dims[x_dims.size() - 1]);

    auto scale_dim = ctx->GetInputDim("Scale");
    auto bias_dim = ctx->GetInputDim("Bias");

    PADDLE_ENFORCE_EQ(
        scale_dim.size(),
        1UL,
        platform::errors::InvalidArgument(
            "ShapeError: the dimension of scale must equal to 1."
            "But received: the shape of scale is [%s], the dimension "
            "of scale is [%d]",
            scale_dim,
            scale_dim.size()));
    PADDLE_ENFORCE_EQ(
        bias_dim.size(),
        1UL,
        platform::errors::InvalidArgument(
            "ShapeError: the dimension of bias must equal to 1."
            "But received: the shape of bias is [%s],the dimension "
            "of bias is [%d]",
            bias_dim,
            bias_dim.size()));

    bool check = true;
    if ((!ctx->IsRuntime()) &&
        (phi::product(scale_dim) <= 0 || phi::product(bias_dim) <= 0)) {
      check = false;
    }

    if (check) {
      PADDLE_ENFORCE_EQ(scale_dim[0],
                        C,
                        platform::errors::InvalidArgument(
                            "ShapeError: the shape of scale must equal to [%d]"
                            "But received: the shape of scale is [%d]",
                            C,
                            scale_dim[0]));
      PADDLE_ENFORCE_EQ(bias_dim[0],
                        C,
                        platform::errors::InvalidArgument(
                            "ShapeError: the shape of bias must equal to [%d]"
                            "But received: the shape of bias is [%d]",
                            C,
                            bias_dim[0]));
    }
    ctx->SetOutputDim("Y", x_dims);
    ctx->ShareLoD("X", "Y");
    VLOG(4) << x_dims;
    ctx->SetOutputDim("MeanOut", {C});
    ctx->SetOutputDim("VarianceOut", {C});
    if (!test_mode) {
      ctx->SetOutputDim("SavedMean", {C});
      ctx->SetOutputDim("SavedVariance", {C});
    }
    if (ctx->HasOutput("ReserveSpace")) {
      ctx->SetOutputDim("ReserveSpace", {-1});
    }
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
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

    return phi::KernelKey(input_data_type, ctx.GetPlace());
  }
};

class InplaceABNGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
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

    const int C = static_cast<int>((ctx->IsRunMKLDNNKernel() == true) ||
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
  phi::KernelKey GetExpectedKernelType(
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
    }
    if (t == nullptr) {
      PADDLE_THROW(
          platform::errors::InvalidArgument("gradient variable of Y is empty"));
    }

    return phi::KernelKey(input_data_type, ctx.GetPlace());
  }
};

class InplaceABNOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddAttr<bool>("is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training. Some layers may run faster when this is true.")
        .SetDefault(false);
    AddAttr<float>("momentum", "").SetDefault(0.9);
    AddAttr<float>("epsilon", "")
        .SetDefault(1e-5)
        .AddCustomChecker([](const float& epsilon) {
          PADDLE_ENFORCE_GE(
              epsilon,
              0.0f,
              platform::errors::InvalidArgument(
                  "'epsilon' should be greater or equal than 0.0."));
          PADDLE_ENFORCE_LE(
              epsilon,
              0.001f,
              platform::errors::InvalidArgument(
                  "'epsilon' should be less or equal than 0.001."));
        });
    AddAttr<std::string>("data_layout", "").SetDefault("NCHW");
    AddInput("X", "The input tensor");
    AddInput("Scale",
             "Scale is a 1-dimensional tensor of size C "
             "that is applied to the output");
    AddInput("Bias",
             "Bias is a 1-dimensional tensor of size C "
             "that is applied to the output");
    AddInput("Mean",
             "The global mean (for training) or "
             "estimated mean (for testing)");
    AddInput("Variance",
             "The global variance (for training) "
             "or estimated Variance (for testing)");
    AddInput(
        "MomentumTensor",
        "(phi::DenseTensor<float32>, optional) If provided, batch_norm will "
        "use this as momentum, this has a higher priority than "
        "attr(momentum), the shape of this tensor MUST BE [1].")
        .AsDispensable();
    AddOutput("Y", "result after normalization");
    AddOutput("MeanOut",
              "Share memory with Mean. "
              "Store the global mean when training");
    AddOutput("VarianceOut",
              "Share memory with Variance. "
              "Store the global Variance when training");
    AddOutput("SavedMean",
              "Mean of the current mini batch, "
              "will apply to output when training")
        .AsIntermediate();
    AddOutput("SavedVariance",
              "Variance of the current mini batch, "
              "will apply to output when training")
        .AsIntermediate();
    AddOutput("ReserveSpace",
              "Reserve GPU space for triggering the new semi-persistent "
              "NHWC kernel")
        .AsDispensable()
        .AsExtra();
    AddAttr<bool>("use_global_stats",
                  "(bool, default false) Whether to use global mean and "
                  "variance. In inference or test mode, set use_global_stats "
                  "to true or is_test true. the behavior is equivalent. "
                  "In train mode, when setting use_global_stats True, the "
                  "global mean and variance are also used during train time, "
                  "the BN acts as scaling and shiffting.")
        .SetDefault(false);
    AddAttr<bool>(
        "trainable_statistics",
        "(bool, default false) Whether to calculate mean and variance "
        "in test mode. If setting true in test mode, mean and variace "
        "will be calculated by current batch statistics.")
        .SetDefault(false);
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
    AddComment(R"DOC(
Batch Normalization.

Batch Norm has been implemented as discussed in the paper:
https://arxiv.org/pdf/1502.03167.pdf
Can be used as a normalizer function for conv2d and fully_connected operations.
The required data format for this layer is one of the following:
1. NHWC `[batch, in_height, in_width, in_channels]`
2. NCHW `[batch, in_channels, in_height, in_width]`

)DOC");
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

template <typename T, typename DeviceContext>
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

template <typename T, typename DeviceContext>
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
    phi::BatchNormGradFunctor<T>(
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

class InplaceABNOpInferVarType
    : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{{"X", /*->*/ "Y"}};
    return m;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INPLACE_OP_INFERER(InplaceAbnOpInplaceInferer, {"X", "Y"});
REGISTER_OPERATOR(inplace_abn,
                  ops::InplaceABNOp,
                  ops::InplaceABNOpMaker,
                  ops::InplaceABNOpInferVarType,
                  ops::InplaceABNOpGradMaker<paddle::framework::OpDesc>,
                  ops::InplaceABNOpGradMaker<paddle::imperative::OpBase>,
                  InplaceAbnOpInplaceInferer)
REGISTER_OPERATOR(inplace_abn_grad, ops::InplaceABNGradOp)

PD_REGISTER_STRUCT_KERNEL(
    inplace_abn, CPU, ALL_LAYOUT, ops::InplaceABNKernel, float, double) {}
PD_REGISTER_STRUCT_KERNEL(inplace_abn_grad,
                          CPU,
                          ALL_LAYOUT,
                          ops::InplaceABNGradKernel,
                          float,
                          double) {}
