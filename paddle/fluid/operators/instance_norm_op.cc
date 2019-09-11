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
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

void InstanceNormOp::InferShape(framework::InferShapeContext *ctx) const {
  PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                    "Input(X) of Instance Norm Op should not be null.");
  PADDLE_ENFORCE_EQ(ctx->HasInput("Scale"), true,
                    "Input(Scale) of Instance Norm Op should not be null.");
  PADDLE_ENFORCE_EQ(ctx->HasInput("Bias"), true,
                    "Input(Bias) of Instance Norm Op should not be null.");
  PADDLE_ENFORCE_EQ(ctx->HasOutput("Y"), true,
                    "Output(Y) of Instance Norm Op should not be null.");

  PADDLE_ENFORCE_EQ(
      ctx->HasOutput("SavedMean"), true,
      "Output(SavedMean) of Instance Norm Op should not be null.");
  PADDLE_ENFORCE_EQ(
      ctx->HasOutput("SavedVariance"), true,
      "Output(SavedVariance) of Instance Norm Op should not be null.");

  const auto x_dims = ctx->GetInputDim("X");
  PADDLE_ENFORCE_GE(x_dims.size(), 2,
                    "the dimension of input X must greater than or equal to 2");
  PADDLE_ENFORCE_LE(x_dims.size(), 5,
                    "the dimension of input X must smaller than or equal to 5");
  auto N = x_dims[0];
  auto C = x_dims[1];
  auto NxC = N * C;

  auto scale_dim = ctx->GetInputDim("Scale");
  auto bias_dim = ctx->GetInputDim("Bias");

  PADDLE_ENFORCE_EQ(scale_dim.size(), 1UL);
  PADDLE_ENFORCE_EQ(bias_dim.size(), 1UL);

  bool check = !((!ctx->IsRuntime()) && (framework::product(scale_dim) <= 0 ||
                                         framework::product(bias_dim) <= 0));

  if (check) {
    PADDLE_ENFORCE_EQ(scale_dim[0], C);
    PADDLE_ENFORCE_EQ(bias_dim[0], C);
  }

  ctx->SetOutputDim("Y", x_dims);
  ctx->SetOutputDim("SavedMean", {NxC});
  ctx->SetOutputDim("SavedVariance", {NxC});
  ctx->ShareLoD("X", "Y");
}

framework::OpKernelType InstanceNormOp::GetExpectedKernelType(
    const framework::ExecutionContext &ctx) const {
  auto input_data_type = ctx.Input<Tensor>("X")->type();
  // By default, the type of the scale, bias, mean,
  // and var tensors should both be float. (For float or float16 input tensor)
  // or double (For double input tensor).
  auto in_param_type = framework::proto::VarType::FP32;
  if (input_data_type == framework::proto::VarType::FP64) {
    in_param_type = framework::proto::VarType::FP64;
  }
  PADDLE_ENFORCE_EQ(in_param_type, ctx.Input<Tensor>("Scale")->type(),
                    "Scale input should be of float type");
  PADDLE_ENFORCE_EQ(in_param_type, ctx.Input<Tensor>("Bias")->type(),
                    "Bias input should be of float type");

  return framework::OpKernelType(input_data_type, ctx.GetPlace());
}

void InstanceNormOpMaker::Make() {
  AddAttr<float>("epsilon", "")
      .SetDefault(1e-5)
      .AddCustomChecker([](const float &epsilon) {
        PADDLE_ENFORCE_EQ(epsilon >= 0.0f && epsilon <= 0.001f, true,
                          "'epsilon' should be between 0.0 and 0.001.");
      });
  AddInput("X", "The input tensor");
  AddInput("Scale",
           "Scale is a 1-dimensional tensor of size C "
           "that is applied to the output");
  AddInput("Bias",
           "Bias is a 1-dimensional tensor of size C "
           "that is applied to the output");
  AddOutput("Y", "result after normalization");
  AddOutput("SavedMean",
            "Mean of the current mini batch, "
            "will apply to output when training")
      .AsIntermediate();
  AddOutput("SavedVariance",
            "Variance of the current mini batch, "
            "will apply to output when training")
      .AsIntermediate();
  AddComment(R"DOC(
Instance Normalization.

Instance Norm has been implemented as disscussed in the paper:
https://arxiv.org/pdf/1607.08022.pdf
Can be used as a normalizer function for conv2d and fully_connected operations.
The required data format for this layer is as following:
NCHW `[batch, in_channels, in_height, in_width]`

)DOC");
}

template <typename T>
class InstanceNormKernel<platform::CPUDeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    T epsilon = static_cast<T>(ctx.Attr<float>("epsilon"));

    const auto *x = ctx.Input<Tensor>("X");
    const auto &x_dims = x->dims();

    const int N = x_dims[0];
    const int C = x_dims[1];
    const int NxC = N * C;

    const int sample_size = x->numel() / N / C;

    auto *y = ctx.Output<Tensor>("Y");
    auto *saved_mean = ctx.Output<Tensor>("SavedMean");
    auto *saved_variance = ctx.Output<Tensor>("SavedVariance");

    auto &dev_ctx = ctx.template device_context<platform::CPUDeviceContext>();
    auto *place = dev_ctx.eigen_device();

    Eigen::DSizes<int, 2> bcast(1, sample_size);
    Eigen::DSizes<int, 2> C_shape(C, 1);
    Eigen::DSizes<int, 2> NxC_shape(NxC, 1);
    Eigen::DSizes<int, 2> shape(NxC, sample_size);

    math::SetConstant<platform::CPUDeviceContext, T> set_constant;

    saved_mean->mutable_data<T>(ctx.GetPlace());
    saved_variance->mutable_data<T>(ctx.GetPlace());
    set_constant(dev_ctx, saved_mean, static_cast<T>(0));
    set_constant(dev_ctx, saved_variance, static_cast<T>(0));

    auto saved_mean_a = framework::EigenVector<T>::Flatten(*saved_mean);
    auto saved_mean_e = saved_mean_a.reshape(NxC_shape);
    auto saved_variance_a = framework::EigenVector<T>::Flatten(*saved_variance);
    auto saved_variance_e = saved_variance_a.reshape(NxC_shape);

    auto x_e = framework::EigenVector<T>::Flatten(*x);
    auto x_arr = x_e.reshape(shape);

    Eigen::DSizes<int, 1> rdims(1);

    saved_mean_e.device(*place) = x_arr.mean(rdims);
    auto saved_variance_arr =
        (x_arr - saved_mean_e.broadcast(bcast)).square().mean(rdims) + epsilon;

    saved_variance_e.device(*place) = saved_variance_arr.sqrt().inverse();

    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *bias = ctx.Input<Tensor>("Bias");
    auto scale_e = framework::EigenVector<T>::Flatten(*scale);
    auto scale_arr = scale_e.reshape(C_shape);
    auto bias_e = framework::EigenVector<T>::Flatten(*bias);
    auto bias_arr = bias_e.reshape(C_shape);

    y->mutable_data<T>(ctx.GetPlace());
    auto y_e = framework::EigenVector<T>::Flatten(*y);
    auto y_arr = y_e.reshape(shape);

    // (x - mean) * inv_std * scale + bias
    Eigen::DSizes<int, 2> bcast_param(N, sample_size);
    y_arr.device(*place) = (x_arr - saved_mean_e.broadcast(bcast)) *
                               saved_variance_e.broadcast(bcast) *
                               scale_arr.broadcast(bcast_param) +
                           bias_arr.broadcast(bcast_param);
  }
};

void InstanceNormGradOp::InferShape(framework::InferShapeContext *ctx) const {
  PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true, "Input(X) should not be null");
  PADDLE_ENFORCE_EQ(ctx->HasInput("Scale"), true,
                    "Input(scale) should not be null");

  PADDLE_ENFORCE_EQ(ctx->HasInput(framework::GradVarName("Y")), true,
                    "Input(Y@GRAD) should not be null");
  PADDLE_ENFORCE_EQ(ctx->HasInput("SavedMean"), true,
                    "Input(SavedMean) should not be null");
  PADDLE_ENFORCE_EQ(ctx->HasInput("SavedVariance"), true,
                    "Input(SavedVariance) should not be null");

  // check output
  PADDLE_ENFORCE_EQ(ctx->HasOutput(framework::GradVarName("X")), true,
                    "Output(x@GRAD) should not be null");
  if (ctx->HasOutput(framework::GradVarName("Scale"))) {
    PADDLE_ENFORCE_EQ(ctx->HasOutput(framework::GradVarName("Bias")), true,
                      "Output(Scale@GRAD) and Output(Bias@GRAD) should not be "
                      "null at the same time");
  }
  const auto x_dims = ctx->GetInputDim("X");
  const int C = x_dims[1];
  ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
  if (ctx->HasOutput(framework::GradVarName("Scale"))) {
    ctx->SetOutputDim(framework::GradVarName("Scale"), {C});
    ctx->SetOutputDim(framework::GradVarName("Bias"), {C});
  }
}

framework::OpKernelType InstanceNormGradOp::GetExpectedKernelType(
    const framework::ExecutionContext &ctx) const {
  const auto *var = ctx.InputVar(framework::GradVarName("Y"));
  if (var == nullptr) {
    PADDLE_THROW("cannot find Y@GRAD");
  }
  const Tensor *t = nullptr;
  if (var->IsType<Tensor>()) {
    t = &var->Get<Tensor>();
  } else if (var->IsType<LoDTensor>()) {
    t = &var->Get<LoDTensor>();
  }
  if (t == nullptr) {
    PADDLE_THROW("cannot find Y@GRAD");
  }
  return framework::OpKernelType(ctx.Input<Tensor>("X")->type(),
                                 ctx.GetPlace());
}

template <typename T>
class InstanceNormGradKernel<platform::CPUDeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *x = ctx.Input<Tensor>("X");
    const auto *d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *saved_mean = ctx.Input<Tensor>("SavedMean");
    const auto *saved_inv_variance = ctx.Input<Tensor>("SavedVariance");

    const auto &x_dims = x->dims();

    const int N = x_dims[0];
    const int C = x_dims[1];
    const int NxC = N * C;
    const int sample_size = x->numel() / N / C;

    auto *d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *d_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto *d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));
    d_x->mutable_data<T>(ctx.GetPlace());

    auto &dev_ctx = ctx.template device_context<platform::CPUDeviceContext>();
    auto *place = dev_ctx.eigen_device();

    Eigen::DSizes<int, 1> rdims(0);
    Eigen::DSizes<int, 1> mean_rdims(1);
    Eigen::DSizes<int, 2> rshape(NxC, sample_size);
    Eigen::DSizes<int, 2> bcast(1, sample_size);
    Eigen::DSizes<int, 2> C_shape(C, 1);
    Eigen::DSizes<int, 2> NxC_shape(NxC, 1);
    Eigen::DSizes<int, 2> param_shape(N, C);
    Eigen::DSizes<int, 2> shape(NxC, sample_size);

    auto scale_e = framework::EigenVector<T>::Flatten(*scale);
    auto mean_e = framework::EigenVector<T>::Flatten(*saved_mean);
    auto inv_var_e = framework::EigenVector<T>::Flatten(*saved_inv_variance);
    auto dy_e = framework::EigenVector<T>::Flatten(*d_y);
    auto x_e = framework::EigenVector<T>::Flatten(*x);

    auto scale_arr = scale_e.reshape(C_shape);
    auto mean_arr = mean_e.reshape(NxC_shape);
    auto inv_var_arr = inv_var_e.reshape(NxC_shape);
    auto dy_arr = dy_e.reshape(shape);
    auto x_arr = x_e.reshape(shape);

    auto tmp =
        (x_arr - mean_arr.broadcast(bcast)) * inv_var_arr.broadcast(bcast);

    math::SetConstant<platform::CPUDeviceContext, T> set_constant;
    // math: d_bias = np.sum(d_y, axis=(n,h,w))
    // math: d_scale = np.sum((X-mean) / inv_std * dy, axis=(n, h,w))
    if (d_scale && d_bias) {
      d_scale->mutable_data<T>(ctx.GetPlace());
      d_bias->mutable_data<T>(ctx.GetPlace());
      set_constant(dev_ctx, d_scale, static_cast<T>(0));
      set_constant(dev_ctx, d_bias, static_cast<T>(0));

      auto d_scale_e = framework::EigenVector<T>::Flatten(*d_scale);
      auto d_bias_e = framework::EigenVector<T>::Flatten(*d_bias);
      auto d_scale_data = d_scale_e.reshape(C_shape);
      auto d_bias_data = d_bias_e.reshape(C_shape);
      d_bias_data.device(*place) =
          dy_arr.sum(mean_rdims).reshape(param_shape).sum(rdims);
      d_scale_data.device(*place) =
          (tmp * dy_arr).sum(mean_rdims).reshape(param_shape).sum(rdims);
    }

    auto dy_mean = dy_arr.mean(mean_rdims).reshape(NxC_shape).broadcast(bcast);

    Eigen::DSizes<int, 2> bcast_param(N, sample_size);
    set_constant(dev_ctx, d_x, static_cast<T>(0));
    // math: d_x = scale * inv_var * d_y - scale * inv_var * np.sum(d_y,
    // axis=(h,w))
    //             - scale * (X - mean) * inv_var.pow(3) * np.sum(d_y * (X -
    //             mean),
    //             axis=(h,w))
    auto dx_e = framework::EigenVector<T>::Flatten(*d_x);
    auto dx_arr = dx_e.reshape(shape);
    dx_arr.device(*place) = scale_arr.broadcast(bcast_param) *
                            inv_var_arr.broadcast(bcast) *
                            (dy_arr - dy_mean -
                             tmp *
                                 (dy_arr * tmp)
                                     .mean(mean_rdims)
                                     .reshape(NxC_shape)
                                     .broadcast(bcast));
  }
};

std::unique_ptr<framework::OpDesc> InstanceNormGradMaker::Apply() const {
  auto *op = new framework::OpDesc();
  op->SetType("instance_norm_grad");
  op->SetInput("X", Input("X"));
  op->SetInput(framework::GradVarName("Y"), OutputGrad("Y"));

  op->SetInput("Scale", Input("Scale"));
  op->SetInput("Bias", Input("Bias"));
  op->SetInput("SavedMean", Output("SavedMean"));
  op->SetInput("SavedVariance", Output("SavedVariance"));

  op->SetAttrMap(Attrs());
  op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
  op->SetOutput(framework::GradVarName("Scale"), InputGrad("Scale"));
  op->SetOutput(framework::GradVarName("Bias"), InputGrad("Bias"));

  return std::unique_ptr<framework::OpDesc>(op);
}

void InstanceNormDoubleGradOp::InferShape(
    framework::InferShapeContext *ctx) const {
  PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true, "Input(X) should not be null");
  PADDLE_ENFORCE_EQ(ctx->HasInput("Scale"), true,
                    "Input(Scale) should not be null.");
  PADDLE_ENFORCE_EQ(ctx->HasInput("SavedMean"), true,
                    "Input(SavedMean) should not be null");
  PADDLE_ENFORCE_EQ(ctx->HasInput("SavedVariance"), true,
                    "Input(SavedVariance) should not be null");
  PADDLE_ENFORCE_EQ(ctx->HasInput("DDX"), true,
                    "Input(DDX) should not be null.");
  PADDLE_ENFORCE_EQ(ctx->HasInput("DY"), true,
                    "Input(Y@GRAD) should not be null");

  // check output
  PADDLE_ENFORCE_EQ(ctx->HasOutput("DX"), true,
                    "Output(DX) should not be null");

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
    PADDLE_THROW("cannot find Y@GRAD");
  }
  const Tensor *t = nullptr;
  if (var->IsType<Tensor>()) {
    t = &var->Get<Tensor>();
  } else if (var->IsType<LoDTensor>()) {
    t = &var->Get<LoDTensor>();
  }
  if (t == nullptr) {
    PADDLE_THROW("cannot find Y@GRAD");
  }
  return framework::OpKernelType(ctx.Input<Tensor>("X")->type(),
                                 ctx.GetPlace());
}

std::unique_ptr<framework::OpDesc> InstanceNormDoubleGradMaker::Apply() const {
  auto *op = new framework::OpDesc();
  op->SetType("instance_norm_grad_grad");
  op->SetInput("X", Input("X"));
  op->SetInput("Scale", Input("Scale"));
  op->SetInput("SavedMean", Input("SavedMean"));
  op->SetInput("SavedVariance", Input("SavedVariance"));
  op->SetInput("DDX", OutputGrad(framework::GradVarName("X")));
  op->SetInput("DDScale", OutputGrad(framework::GradVarName("Scale")));
  op->SetInput("DDBias", OutputGrad(framework::GradVarName("Bias")));
  op->SetInput("DY", Input(framework::GradVarName("Y")));

  op->SetAttrMap(Attrs());
  op->SetOutput("DX", InputGrad("X"));
  op->SetOutput("DScale", InputGrad("Scale"));
  op->SetOutput("DDY", InputGrad(framework::GradVarName("Y")));
  return std::unique_ptr<framework::OpDesc>(op);
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

    ConstEigenVectorArrayMap<T> scale_arr(Scale->data<T>(), C);

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
    //          (np.mean(ddx, axis=(h,w)) - ddx) + ddr * (dy * inv_var - inv_var
    //          *
    //          np.mean(dy, axis=(h,w)) -
    //          inv_var.pow(3) * (x - mean) * np.mean(dy * (x - mean),
    //          axis=(h,w))))

    auto &dev_ctx = ctx.template device_context<platform::CPUDeviceContext>();
    math::SetConstant<platform::CPUDeviceContext, T> set_constant;

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

        dx_arr = scale_tile_data * dx_arr.eval();
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
            (dy_arr - dy_arr.colwise().sum() / sample_size -
             x_sub_mean_mul_invstd_arr *
                 (dy_arr * x_sub_mean_mul_invstd_arr).colwise().sum() /
                 sample_size);
        first_grad_arr = first_grad_arr.eval() * ddx_arr;
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

DECLARE_INPLACE_OP_INFERER(InstanceNormDoubleGradOpInplaceInference,
                           {"DY", "DDY"});

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(instance_norm, ops::InstanceNormOp, ops::InstanceNormOpMaker,
                  ops::InstanceNormOpInferVarType, ops::InstanceNormGradMaker);
REGISTER_OPERATOR(instance_norm_grad, ops::InstanceNormGradOp,
                  ops::InstanceNormDoubleGradMaker);
REGISTER_OPERATOR(instance_norm_grad_grad, ops::InstanceNormDoubleGradOp,
                  ops::InstanceNormDoubleGradOpInplaceInference);

REGISTER_OP_CPU_KERNEL(
    instance_norm,
    ops::InstanceNormKernel<paddle::platform::CPUDeviceContext, float>,
    ops::InstanceNormKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    instance_norm_grad,
    ops::InstanceNormGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::InstanceNormGradKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    instance_norm_grad_grad,
    ops::InstanceNormDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                      float>,
    ops::InstanceNormDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                      double>);
