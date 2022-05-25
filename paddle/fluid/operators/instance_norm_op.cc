/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/instance_norm_op.h"
#include <memory>
#include <string>
#include <unordered_map>
#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

void InstanceNormOp::InferShape(framework::InferShapeContext *ctx) const {
  OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "InstanceNorm");
  OP_INOUT_CHECK(ctx->HasOutput("Y"), "Output", "Y", "InstanceNorm");
  OP_INOUT_CHECK(ctx->HasOutput("SavedMean"), "Output", "SavedMean",
                 "InstanceNorm");
  OP_INOUT_CHECK(ctx->HasOutput("SavedVariance"), "Output", "SavedVariance",
                 "InstanceNorm");

  const auto x_dims = ctx->GetInputDim("X");
  PADDLE_ENFORCE_NE(phi::product(x_dims), 0,
                    platform::errors::PreconditionNotMet(
                        "The Input variable X(%s) has not "
                        "been initialized. You may need to confirm "
                        "if you put exe.run(startup_program) "
                        "after optimizer.minimize function.",
                        ctx->Inputs("X").front()));
  PADDLE_ENFORCE_GE(
      x_dims.size(), 2,
      platform::errors::InvalidArgument(
          "ShapeError: the dimension of input X must "
          "greater than or equal to 2. But received: the shape of input "
          "X = [%s], the dimension of input X =[%d]",
          x_dims, x_dims.size()));
  PADDLE_ENFORCE_LE(
      x_dims.size(), 5,
      platform::errors::InvalidArgument(
          "ShapeError: the dimension of input X must "
          "smaller than or equal to 5, But received: the shape of input "
          "X = [%s], the dimension of input X = [%d]",
          x_dims, x_dims.size()));
  auto N = x_dims[0];
  auto C = x_dims[1];
  auto NxC = N * C;

  if (ctx->HasInput("Scale")) {
    auto scale_dim = ctx->GetInputDim("Scale");

    PADDLE_ENFORCE_EQ(
        scale_dim.size(), 1UL,
        platform::errors::InvalidArgument(
            "ShapeError: the dimension of scale must equal to 1."
            "But received: the shape of scale is [%s], the dimension "
            "of scale is [%d]",
            scale_dim, scale_dim.size()));

    bool check = !((!ctx->IsRuntime()) && (phi::product(scale_dim) <= 0));

    if (check) {
      PADDLE_ENFORCE_EQ(scale_dim[0], C,
                        platform::errors::InvalidArgument(
                            "ShapeError: the shape of scale must equal to [%d]"
                            "But received: the shape of scale is [%d]",
                            C, scale_dim[0]));
    }
  }
  if (ctx->HasInput("Bias")) {
    auto bias_dim = ctx->GetInputDim("Bias");
    PADDLE_ENFORCE_EQ(
        bias_dim.size(), 1UL,
        platform::errors::InvalidArgument(
            "ShapeError: the dimension of bias must equal to 1."
            "But received: the shape of bias is [%s],the dimension "
            "of bias is [%d]",
            bias_dim, bias_dim.size()));

    bool check = !((!ctx->IsRuntime()) && (phi::product(bias_dim) <= 0));
    if (check) {
      PADDLE_ENFORCE_EQ(bias_dim[0], C,
                        platform::errors::InvalidArgument(
                            "ShapeError: the shape of bias must equal to [%d]"
                            "But received: the shape of bias is [%d]",
                            C, bias_dim[0]));
    }
  }

  ctx->SetOutputDim("Y", x_dims);
  ctx->SetOutputDim("SavedMean", {NxC});
  ctx->SetOutputDim("SavedVariance", {NxC});
  ctx->ShareLoD("X", "Y");
}

framework::OpKernelType InstanceNormOp::GetExpectedKernelType(
    const framework::ExecutionContext &ctx) const {
  auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
  // By default, the type of the scale, bias, mean,
  // and var tensors should both be float. (For float or float16 input tensor)
  // or double (For double input tensor).
  auto in_param_type = framework::proto::VarType::FP32;
  if (input_data_type == framework::proto::VarType::FP64) {
    in_param_type = framework::proto::VarType::FP64;
  }
  if (ctx.HasInput("Scale")) {
    PADDLE_ENFORCE_EQ(in_param_type, framework::TransToProtoVarType(
                                         ctx.Input<Tensor>("Scale")->dtype()),
                      platform::errors::InvalidArgument(
                          "Scale input should be of float type"));
  }
  if (ctx.HasInput("Bias")) {
    PADDLE_ENFORCE_EQ(in_param_type, framework::TransToProtoVarType(
                                         ctx.Input<Tensor>("Bias")->dtype()),
                      platform::errors::InvalidArgument(
                          "Bias input should be of float type"));
  }

  return framework::OpKernelType(input_data_type, ctx.GetPlace());
}

void InstanceNormOpMaker::Make() {
  AddAttr<float>("epsilon", "")
      .SetDefault(1e-5)
      .AddCustomChecker([](const float &epsilon) {
        PADDLE_ENFORCE_EQ(epsilon >= 0.0f && epsilon <= 0.001f, true,
                          platform::errors::InvalidArgument(
                              "'epsilon' should be between 0.0 and 0.001."));
      });
  AddInput("X", "The input tensor");
  AddInput("Scale",
           "Scale is a 1-dimensional tensor of size C "
           "that is applied to the output")
      .AsDispensable();
  AddInput("Bias",
           "Bias is a 1-dimensional tensor of size C "
           "that is applied to the output")
      .AsDispensable();
  AddOutput("Y", "result after normalization");
  AddOutput("SavedMean",
            "Mean of the current mini batch, "
            "will apply to output when training")
      .AsIntermediate()
      .AsExtra();
  AddOutput("SavedVariance",
            "Variance of the current mini batch, "
            "will apply to output when training")
      .AsIntermediate()
      .AsExtra();
  AddComment(R"DOC(
Instance Normalization.

Instance Norm has been implemented as disscussed in the paper:
https://arxiv.org/pdf/1607.08022.pdf
Can be used as a normalizer function for conv2d and fully_connected operations.
The required data format for this layer is as following:
NCHW `[batch, in_channels, in_height, in_width]`

)DOC");
}

void InstanceNormGradOp::InferShape(framework::InferShapeContext *ctx) const {
  OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "InstanceNormGrad");
  OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Y")), "Input",
                 framework::GradVarName("Y"), "InstanceNormGrad");
  OP_INOUT_CHECK(ctx->HasInput("SavedMean"), "Input", "SavedMean",
                 "InstanceNormGrad");
  OP_INOUT_CHECK(ctx->HasInput("SavedVariance"), "Input", "SavedVariance",
                 "InstanceNormGrad");

  // check output
  OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")), "Output",
                 framework::GradVarName("X"), "InstanceNormGrad");
  const auto x_dims = ctx->GetInputDim("X");
  const int C = x_dims[1];
  ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
  if (ctx->HasOutput(framework::GradVarName("Scale"))) {
    ctx->SetOutputDim(framework::GradVarName("Scale"), {C});
  }
  if (ctx->HasOutput(framework::GradVarName("Bias"))) {
    ctx->SetOutputDim(framework::GradVarName("Bias"), {C});
  }
}

framework::OpKernelType InstanceNormGradOp::GetExpectedKernelType(
    const framework::ExecutionContext &ctx) const {
  const auto *var = ctx.InputVar(framework::GradVarName("Y"));
  if (var == nullptr) {
    PADDLE_THROW(
        platform::errors::NotFound("cannot find gradient variable of Y"));
  }
  const Tensor *t = nullptr;
  if (var->IsType<Tensor>()) {
    t = &var->Get<Tensor>();
  } else if (var->IsType<LoDTensor>()) {
    t = &var->Get<LoDTensor>();
  }
  if (t == nullptr) {
    PADDLE_THROW(
        platform::errors::InvalidArgument("gradient variable of Y is empty"));
  }
  return framework::OpKernelType(
      OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
}

void InstanceNormDoubleGradOp::InferShape(
    framework::InferShapeContext *ctx) const {
  OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "InstanceNormDoubleGrad");
  OP_INOUT_CHECK(ctx->HasInput("SavedMean"), "Input", "SavedMean",
                 "InstanceNormDoubleGrad");
  OP_INOUT_CHECK(ctx->HasInput("SavedVariance"), "Input", "SavedVariance",
                 "InstanceNormDoubleGrad");
  OP_INOUT_CHECK(ctx->HasInput("DDX"), "Input", "DDX",
                 "InstanceNormDoubleGrad");
  OP_INOUT_CHECK(ctx->HasInput("DY"), "Input", "DY", "InstanceNormDoubleGrad");

  // check output
  OP_INOUT_CHECK(ctx->HasOutput("DX"), "Output", "DX",
                 "InstanceNormDoubleGrad");

  const auto x_dims = ctx->GetInputDim("X");
  const int C = x_dims[1];
  if (ctx->HasOutput("DX")) {
    ctx->SetOutputDim("DX", x_dims);
  }
  if (ctx->HasOutput("DScale")) {
    ctx->SetOutputDim("DScale", {C});
  }
  if (ctx->HasOutput("DDY")) {
    ctx->ShareDim("X", "DDY");
  }
}

framework::OpKernelType InstanceNormDoubleGradOp::GetExpectedKernelType(
    const framework::ExecutionContext &ctx) const {
  const auto *var = ctx.InputVar("DY");
  if (var == nullptr) {
    PADDLE_THROW(
        platform::errors::NotFound("cannot find gradient variable of Y"));
  }
  const Tensor *t = nullptr;
  if (var->IsType<Tensor>()) {
    t = &var->Get<Tensor>();
  } else if (var->IsType<LoDTensor>()) {
    t = &var->Get<LoDTensor>();
  }
  if (t == nullptr) {
    PADDLE_THROW(
        platform::errors::InvalidArgument("gradient variable of Y is empty"));
  }
  return framework::OpKernelType(
      OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
}

template <typename T>
class InstanceNormDoubleGradKernel<platform::CPUDeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *X = ctx.Input<Tensor>("X");
    const auto *Scale = ctx.Input<Tensor>("Scale");
    const auto *dY = ctx.Input<Tensor>("DY");
    const auto *Saved_mean = ctx.Input<Tensor>("SavedMean");
    const auto *Saved_variance = ctx.Input<Tensor>("SavedVariance");
    const auto *ddX = ctx.Input<Tensor>("DDX");
    const auto *ddScale = ctx.Input<Tensor>("DDScale");
    const auto *ddBias = ctx.Input<Tensor>("DDBias");

    auto *dX = ctx.Output<Tensor>("DX");
    auto *dScale = ctx.Output<Tensor>("DScale");
    auto *ddY = ctx.Output<Tensor>("DDY");

    auto &dev_ctx = ctx.template device_context<platform::CPUDeviceContext>();
    phi::funcs::SetConstant<platform::CPUDeviceContext, T> set_constant;

    const auto &x_dims = X->dims();
    int N, C, H, W, D;
    ExtractNCWHD(x_dims, DataLayout::kNCHW, &N, &C, &H, &W, &D);
    const int sample_size = X->numel() / N / C;
    const int NxC = N * C;

    const T *mean_data = Saved_mean->data<T>();
    const T *inv_var_data = Saved_variance->data<T>();
    Tensor mean_tensor;
    Tensor inv_var_tensor;
    ConstEigenArrayMap<T> x_arr(X->data<T>(), sample_size, NxC);
    ConstEigenVectorArrayMap<T> mean_arr(mean_data, NxC);
    ConstEigenVectorArrayMap<T> inv_var_arr(inv_var_data, NxC);

    Tensor mean_tile;
    mean_tile.Resize({sample_size, NxC});
    mean_tile.mutable_data<T>(ctx.GetPlace());
    EigenArrayMap<T> mean_tile_data(mean_tile.mutable_data<T>(ctx.GetPlace()),
                                    sample_size, NxC);

    Tensor inv_var_tile;
    inv_var_tile.Resize({sample_size, NxC});
    inv_var_tile.mutable_data<T>(ctx.GetPlace());
    EigenArrayMap<T> inv_var_tile_data(
        inv_var_tile.mutable_data<T>(ctx.GetPlace()), sample_size, NxC);

    mean_tile_data = mean_arr.transpose().replicate(sample_size, 1);
    inv_var_tile_data = inv_var_arr.transpose().replicate(sample_size, 1);

    Tensor Scale_data;
    if (!Scale) {
      Scale_data.mutable_data<T>({C}, ctx.GetPlace());
      set_constant(dev_ctx, &Scale_data, static_cast<T>(1));
    }
    ConstEigenVectorArrayMap<T> scale_arr(
        Scale ? Scale->data<T>() : Scale_data.data<T>(), C);

    Tensor scale_tile;
    scale_tile.Resize({sample_size, NxC});
    scale_tile.mutable_data<T>(ctx.GetPlace());
    EigenArrayMap<T> scale_tile_data(scale_tile.mutable_data<T>(ctx.GetPlace()),
                                     sample_size, NxC);
    scale_tile_data = scale_arr.transpose().replicate(sample_size, N);

    ConstEigenArrayMap<T> dy_arr(dY->data<T>(), sample_size, NxC);
    ConstEigenArrayMap<T> ddx_arr(ddX->data<T>(), sample_size, NxC);

    // math: dx = scale * ((x - mean) * inv_var / HxW * (np.mean(ddx,
    // axis=(h,w)) *
    //          np.sum(dy, axis=(h,w)) -
    //          np.sum(dy * ddx, axis=(h,w)) + 3 * np.mean(dy * (x - mean),
    //          axis=(h,w)) * inv_var.pow(2) *
    //          np.sum(ddx * (x - mean), axis=(h,w))) + inv_var.pow(3) / HxW *
    //          np.sum(ddx * (x - mean)) *
    //          (np.mean(dy, axis=(h,w)) - dy) + inv_var.pow(3) / HxW *
    //          np.sum(dy,
    //          axis=(h,w)) * (x - mean) *
    //          (np.mean(ddx, axis=(h,w)) - ddx)) + ddr * (dy * inv_var -
    //          inv_var *
    //          np.mean(dy, axis=(h,w)) -
    //          inv_var.pow(3) * (x - mean) * np.mean(dy * (x - mean),
    //          axis=(h,w)))

    Tensor x_sub_mean_mul_invstd;
    x_sub_mean_mul_invstd.Resize({sample_size, NxC});
    x_sub_mean_mul_invstd.mutable_data<T>(ctx.GetPlace());
    EigenArrayMap<T> x_sub_mean_mul_invstd_arr(
        x_sub_mean_mul_invstd.mutable_data<T>(ctx.GetPlace()), sample_size,
        NxC);
    x_sub_mean_mul_invstd_arr = (x_arr - mean_tile_data) * inv_var_tile_data;

    if (dX) {
      dX->mutable_data<T>(ctx.GetPlace());
      set_constant(dev_ctx, dX, static_cast<T>(0));
      EigenArrayMap<T> dx_arr(dX->mutable_data<T>(ctx.GetPlace()), sample_size,
                              NxC);

      if (ddX) {
        dx_arr +=
            x_sub_mean_mul_invstd_arr * inv_var_tile_data * inv_var_tile_data /
            sample_size *
            (ddx_arr.colwise().sum() * dy_arr.colwise().sum() / sample_size -
             (dy_arr * ddx_arr).colwise().sum() +
             3. * (dy_arr * x_sub_mean_mul_invstd_arr).colwise().sum() *
                 (ddx_arr * x_sub_mean_mul_invstd_arr).colwise().sum() /
                 sample_size);

        dx_arr += (ddx_arr * x_sub_mean_mul_invstd_arr).colwise().sum() /
                  sample_size * inv_var_tile_data * inv_var_tile_data *
                  (dy_arr.colwise().sum() / sample_size - dy_arr);

        dx_arr += (dy_arr * x_sub_mean_mul_invstd_arr).colwise().sum() /
                  sample_size * inv_var_tile_data * inv_var_tile_data *
                  (ddx_arr.colwise().sum() / sample_size - ddx_arr);

        dx_arr = scale_tile_data * dx_arr;
      }
      if (ddScale) {
        ConstEigenVectorArrayMap<T> ddscale_arr(ddScale->data<T>(), C);

        Tensor ddscale_tile;
        ddscale_tile.Resize({sample_size, NxC});
        ddscale_tile.mutable_data<T>(ctx.GetPlace());
        EigenArrayMap<T> ddscale_tile_data(
            ddscale_tile.mutable_data<T>(ctx.GetPlace()), sample_size, NxC);
        ddscale_tile_data = ddscale_arr.transpose().replicate(sample_size, N);

        dx_arr += (dy_arr * inv_var_tile_data -
                   dy_arr.colwise().sum() / sample_size * inv_var_tile_data -
                   x_sub_mean_mul_invstd_arr * inv_var_tile_data *
                       (dy_arr * x_sub_mean_mul_invstd_arr).colwise().sum() /
                       sample_size) *
                  ddscale_tile_data;
      }
    }
    if (dScale) {
      // math: dscale = inv_var * (dy - np.mean(dy, axis=(h,w) - (x-mean) *
      //            inv_var.pow(2) * np.mean(dy * (x-mean), axis=(h,w)))) * ddx
      dScale->mutable_data<T>(ctx.GetPlace());
      set_constant(dev_ctx, dScale, static_cast<T>(0));
      EigenVectorArrayMap<T> dscale_arr(dScale->mutable_data<T>(ctx.GetPlace()),
                                        C);
      if (ddX) {
        Tensor first_grad;
        first_grad.Resize({sample_size, NxC});
        first_grad.mutable_data<T>(ctx.GetPlace());
        set_constant(dev_ctx, &first_grad, static_cast<T>(0));
        EigenArrayMap<T> first_grad_arr(
            first_grad.mutable_data<T>(ctx.GetPlace()), sample_size, NxC);

        first_grad_arr +=
            inv_var_tile_data *
            (dy_arr -
             dy_arr.colwise().sum().replicate(sample_size, 1) / sample_size -
             x_sub_mean_mul_invstd_arr *
                 (dy_arr * x_sub_mean_mul_invstd_arr)
                     .colwise()
                     .sum()
                     .replicate(sample_size, 1) /
                 sample_size);
        first_grad_arr = first_grad_arr * ddx_arr;
        for (int nc = 0; nc < NxC; ++nc) {
          int c = nc % C;
          dscale_arr(c) += first_grad_arr.colwise().sum()(nc);
        }
      }
    }
    if (ddY) {
      // math: ddy = (x - mean) * inv_var * ddscale + ddbias +
      //           scale * inv_var * (ddx - (x - mean) * inv_var.pow(2) *
      //           np.mean(ddx * (x - mean), axis=(h,w)))
      ddY->mutable_data<T>(ctx.GetPlace());
      set_constant(dev_ctx, ddY, static_cast<T>(0));
      EigenArrayMap<T> ddy_arr(ddY->mutable_data<T>(ctx.GetPlace()),
                               sample_size, NxC);
      if (ddX) {
        ddy_arr += scale_tile_data * inv_var_tile_data *
                   (ddx_arr - ddx_arr.colwise().sum() / sample_size -
                    x_sub_mean_mul_invstd_arr *
                        (ddx_arr * x_sub_mean_mul_invstd_arr).colwise().sum() /
                        sample_size);
      }
      if (ddScale && ddBias) {
        ConstEigenVectorArrayMap<T> ddscale_arr(ddScale->data<T>(), C);
        Tensor ddscale_tile;
        ddscale_tile.Resize({sample_size, NxC});
        ddscale_tile.mutable_data<T>(ctx.GetPlace());
        EigenArrayMap<T> ddscale_tile_data(
            ddscale_tile.mutable_data<T>(ctx.GetPlace()), sample_size, NxC);
        ddscale_tile_data = ddscale_arr.transpose().replicate(sample_size, N);

        ConstEigenVectorArrayMap<T> ddbias_arr(ddBias->data<T>(), C);
        Tensor ddbias_tile;
        ddbias_tile.Resize({sample_size, NxC});
        ddbias_tile.mutable_data<T>(ctx.GetPlace());
        EigenArrayMap<T> ddbias_tile_data(
            ddbias_tile.mutable_data<T>(ctx.GetPlace()), sample_size, NxC);
        ddbias_tile_data = ddbias_arr.transpose().replicate(sample_size, N);

        ddy_arr += x_sub_mean_mul_invstd_arr * ddscale_tile_data;
        ddy_arr += ddbias_tile_data;
      }
    }
  }
};

DECLARE_INPLACE_OP_INFERER(InstanceNormDoubleGradOpInplaceInferer,
                           {"DY", "DDY"});

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(instance_norm, ops::InstanceNormOp, ops::InstanceNormOpMaker,
                  ops::InstanceNormOpInferVarType,
                  ops::InstanceNormGradMaker<paddle::framework::OpDesc>,
                  ops::InstanceNormGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(instance_norm_grad, ops::InstanceNormGradOp,
                  ops::InstanceNormDoubleGradMaker<paddle::framework::OpDesc>,
                  ops::InstanceNormDoubleGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(instance_norm_grad_grad, ops::InstanceNormDoubleGradOp,
                  ops::InstanceNormDoubleGradOpInplaceInferer);

REGISTER_OP_CPU_KERNEL(
    instance_norm_grad_grad,
    ops::InstanceNormDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                      float>,
    ops::InstanceNormDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                      double>);

REGISTER_OP_VERSION(instance_norm)
    .AddCheckpoint(
        R"ROC(
      Change dispensable of attribute from False to True in instance_norm.
    )ROC",
        paddle::framework::compatible::OpVersionDesc()
            .ModifyAttr(
                "Bias",
                "The arg 'dispensable' of Input 'Bias' is changed: from "
                "'False' to 'True'.",
                true)
            .ModifyAttr(
                "Scale",
                "The arg 'dispensable' of Input 'Scale' is changed: from "
                "'False' to 'True'.",
                true));
