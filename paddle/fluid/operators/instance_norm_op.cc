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
  PADDLE_ENFORCE(ctx->HasInput("X"),
                 "Input(X) of Instance Norm Op should not be null.");
  PADDLE_ENFORCE(ctx->HasInput("Scale"),
                 "Input(Scale) of Instance Norm Op should not be null.");
  PADDLE_ENFORCE(ctx->HasInput("Bias"),
                 "Input(Bias) of Instance Norm Op should not be null.");
  PADDLE_ENFORCE(ctx->HasInput("Mean"),
                 "Input(Mean) of Instance Norm Op should not be null.");
  PADDLE_ENFORCE(ctx->HasInput("Variance"),
                 "Input(Variance) of Instance Norm Op should not be null.");
  PADDLE_ENFORCE(ctx->HasOutput("Y"),
                 "Output(Y) of Instance Norm Op should not be null.");
  bool is_test = ctx->Attrs().Get<bool>("is_test");

  if (!is_test) {
    PADDLE_ENFORCE(ctx->HasOutput("MeanOut"),
                   "Output(MeanOut) of Instance Norm Op should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("VarianceOut"),
        "Output(VarianceOut) of Instance Norm Op should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("SavedMean"),
                   "Output(SavedMean) of Instance Norm Op should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("SavedVariance"),
        "Output(SavedVariance) of Instance Norm Op should not be null.");
  }

  // make sure Mean/MeanOut and Variance/VarianceOut share memory in Python
  PADDLE_ENFORCE_EQ(ctx->Inputs("Mean")[0], ctx->Outputs("MeanOut")[0],
                    "Mean and MeanOut should share the same memory");
  PADDLE_ENFORCE_EQ(ctx->Inputs("Variance")[0], ctx->Outputs("VarianceOut")[0],
                    "Variance and VarianceOut should share the same memory");

  const auto x_dims = ctx->GetInputDim("X");
  PADDLE_ENFORCE(x_dims.size() >= 2 && x_dims.size() <= 5,
                 "Input X must have 2 to 5 dimensions.");
  auto N = x_dims[0];
  auto C = x_dims[1];
  auto NxC = N * C;

  auto scale_dim = ctx->GetInputDim("Scale");
  auto bias_dim = ctx->GetInputDim("Bias");

  PADDLE_ENFORCE_EQ(scale_dim.size(), 1UL);
  PADDLE_ENFORCE_EQ(bias_dim.size(), 1UL);

  bool check = true;
  if ((!ctx->IsRuntime()) && (framework::product(scale_dim) <= 0 ||
                              framework::product(bias_dim) <= 0)) {
    check = false;
  }
  if (check) {
    PADDLE_ENFORCE_EQ(scale_dim[0], C);
    PADDLE_ENFORCE_EQ(bias_dim[0], C);
  }

  ctx->SetOutputDim("Y", x_dims);
  ctx->SetOutputDim("MeanOut", {NxC});
  ctx->SetOutputDim("VarianceOut", {NxC});
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
  PADDLE_ENFORCE_EQ(in_param_type, ctx.Input<Tensor>("Mean")->type(),
                    "Mean input should be of float type");
  PADDLE_ENFORCE_EQ(in_param_type, ctx.Input<Tensor>("Variance")->type(),
                    "Variance input should be of float type");

  return framework::OpKernelType(input_data_type, ctx.GetPlace());
}

void InstanceNormOpMaker::Make() {
  AddAttr<bool>("is_test",
                "(bool, default false) Set to true for inference only, false "
                "for training. Some layers may run faster when this is true.")
      .SetDefault(false);
  AddAttr<float>("momentum", "").SetDefault(0.9);
  AddAttr<float>("epsilon", "")
      .SetDefault(1e-5)
      .AddCustomChecker([](const float &epsilon) {
        PADDLE_ENFORCE(epsilon >= 0.0f && epsilon <= 0.001f,
                       "'epsilon' should be between 0.0 and 0.001.");
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
  AddAttr<bool>("use_global_stats",
                "(bool, default false) Whether to use global mean and "
                "variance. In inference or test mode, set use_global_stats "
                "to true or is_test true. the behavior is equivalent. "
                "In train mode, when setting use_global_stats True, the "
                "global mean and variance are also used during train time, "
                "the BN acts as scaling and shiffting.")
      .SetDefault(false);
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
    const float epsilon = ctx.Attr<float>("epsilon");
    const float momentum = ctx.Attr<float>("momentum");
    const bool is_test = ctx.Attr<bool>("is_test");
    const bool use_global_stats = ctx.Attr<bool>("use_global_stats");

    bool global_stats = is_test || use_global_stats;
    const auto *x = ctx.Input<Tensor>("X");
    const auto &x_dims = x->dims();

    const int N = x_dims[0];
    const int C = x_dims[1];
    const int NxC = N * C;

    const int sample_size = x->numel() / N / C;

    auto *y = ctx.Output<Tensor>("Y");
    auto *mean_out = ctx.Output<Tensor>("MeanOut");
    auto *variance_out = ctx.Output<Tensor>("VarianceOut");
    auto *saved_mean = ctx.Output<Tensor>("SavedMean");
    auto *saved_variance = ctx.Output<Tensor>("SavedVariance");

    // alloc memory
    y->mutable_data<T>(ctx.GetPlace());
    mean_out->mutable_data<T>(ctx.GetPlace());
    variance_out->mutable_data<T>(ctx.GetPlace());
    saved_mean->mutable_data<T>(ctx.GetPlace());
    saved_variance->mutable_data<T>(ctx.GetPlace());

    if (!global_stats) {
      EigenVectorArrayMap<T> saved_mean_e(
          saved_mean->mutable_data<T>(ctx.GetPlace()), NxC);
      EigenVectorArrayMap<T> saved_variance_e(
          saved_variance->mutable_data<T>(ctx.GetPlace()), NxC);
      saved_mean_e.setZero();
      saved_variance_e.setZero();

      ConstEigenArrayMap<T> x_arr(x->data<T>(), sample_size, NxC);
      for (int nc = 0; nc < NxC; ++nc) {
        saved_mean_e(nc) = x_arr.col(nc).mean();
        saved_variance_e(nc) =
            (x_arr.col(nc) - saved_mean_e(nc)).matrix().squaredNorm();
      }
      saved_variance_e /= sample_size;

      EigenVectorArrayMap<T> running_mean_arr(
          mean_out->mutable_data<T>(ctx.GetPlace()), NxC);
      EigenVectorArrayMap<T> running_var_arr(
          variance_out->mutable_data<T>(ctx.GetPlace()), NxC);

      running_mean_arr =
          running_mean_arr * momentum + saved_mean_e * (1. - momentum);
      running_var_arr =
          running_var_arr * momentum + saved_variance_e * (1. - momentum);
    }

    Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> inv_std(N, C);
    if (global_stats) {
      ConstEigenVectorArrayMap<T> var_arr(
          ctx.Input<Tensor>("Variance")->data<T>(), NxC);
      inv_std = (var_arr + epsilon).sqrt().inverse();
    } else {
      EigenVectorArrayMap<T> saved_inv_std(
          ctx.Output<Tensor>("SavedVariance")->data<T>(), NxC);
      saved_inv_std = (saved_inv_std + epsilon).inverse().sqrt();
      inv_std = saved_inv_std;
    }
    ConstEigenVectorArrayMap<T> mean_arr(
        global_stats ? ctx.Input<Tensor>("Mean")->data<T>()
                     : ctx.Output<Tensor>("SavedMean")->data<T>(),
        NxC);

    // (x - mean) * inv_std * scale + bias
    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *bias = ctx.Input<Tensor>("Bias");
    ConstEigenVectorArrayMap<T> scale_arr(scale->data<T>(), C);
    ConstEigenVectorArrayMap<T> bias_arr(bias->data<T>(), C);

    for (int nc = 0; nc < NxC; ++nc) {
      EigenArrayMap<T> y_arr(y->mutable_data<T>(ctx.GetPlace()), sample_size,
                             NxC);
      ConstEigenArrayMap<T> x_arr(x->data<T>(), sample_size, NxC);
      y_arr.col(nc) =
          (x_arr.col(nc) - mean_arr(nc)) * inv_std(nc) * scale_arr(nc % C) +
          bias_arr(nc % C);
    }
  }
};

void InstanceNormGradOp::InferShape(framework::InferShapeContext *ctx) const {
  PADDLE_ENFORCE(ctx->HasInput("X"));
  PADDLE_ENFORCE(ctx->HasInput("Scale"), "Input(scale) should not be null");

  PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Y")),
                 "Input(Y@GRAD) should not be null");
  PADDLE_ENFORCE(ctx->HasInput("SavedMean"),
                 "Input(SavedMean) should not be null");
  PADDLE_ENFORCE(ctx->HasInput("SavedVariance"),
                 "Input(SavedVariance) should not be null");

  // check output
  PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")), "");
  if (ctx->HasOutput(framework::GradVarName("Scale"))) {
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("Bias")),
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
    const bool use_global_stats = ctx.Attr<bool>("use_global_stats");
    const float epsilon = ctx.Attr<float>("epsilon");

    const auto &x_dims = x->dims();
    PADDLE_ENFORCE(x_dims.size() >= 2 && x_dims.size() <= 5,
                   "The Input dim size should be between 2 and 5");
    const int N = x_dims[0];
    const int C = x_dims[1];
    const int sample_size = x->numel() / N / C;
    const int NxC = N * C;

    auto *d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *d_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto *d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));
    d_x->mutable_data<T>(ctx.GetPlace());

    const T *mean_data = saved_mean->data<T>();
    const T *inv_var_data = saved_inv_variance->data<T>();
    Tensor inv_var_tensor;
    if (use_global_stats) {
      const auto *running_mean = ctx.Input<Tensor>("Mean");
      const auto *running_variance = ctx.Input<Tensor>("Variance");
      mean_data = running_mean->data<T>();
      inv_var_tensor.Resize({NxC});
      T *running_inv_var_data = inv_var_tensor.mutable_data<T>(ctx.GetPlace());
      EigenVectorArrayMap<T> inv_var_tmp(running_inv_var_data, NxC);
      ConstEigenVectorArrayMap<T> var_arr(running_variance->data<T>(), NxC);

      inv_var_tmp = (var_arr + epsilon).sqrt().inverse().eval();
      inv_var_data = running_inv_var_data;
    }
    ConstEigenVectorArrayMap<T> scale_arr(scale->data<T>(), C);
    ConstEigenVectorArrayMap<T> mean_arr(mean_data, NxC);
    ConstEigenVectorArrayMap<T> inv_var_arr(inv_var_data, NxC);

    T *d_bias_data = nullptr;
    T *d_scale_data = nullptr;
    if (d_scale && d_bias) {
      d_scale->mutable_data<T>(ctx.GetPlace());
      d_bias->mutable_data<T>(ctx.GetPlace());
      d_bias_data = d_bias->mutable_data<T>(ctx.GetPlace());
      d_scale_data = d_scale->mutable_data<T>(ctx.GetPlace());
    }

    // d_bias = sum(d_y, axis=0)
    // d_scale = sum((X-mean) / inv_std * dy, axis=0)
    // d_x = scale * inv_var * d_y - scale * inv_var * sum(d_y, axis=0)
    //      - scale * (X - mean) * inv_var.pow(3) * sum(d_y * (X - mean),
    //      axis=0)
    EigenVectorArrayMap<T> d_bias_arr(d_bias_data, C);
    EigenVectorArrayMap<T> d_scale_arr(d_scale_data, C);

    if (d_scale && d_bias) {
      d_bias_arr.setZero();
      d_scale_arr.setZero();
    }
    if ((N * sample_size) == 1 && !use_global_stats) {
      framework::TensorCopy(*d_y, ctx.GetPlace(), d_x);
      return;
    }

    ConstEigenArrayMap<T> x_arr(x->data<T>(), sample_size, NxC);
    ConstEigenArrayMap<T> d_y_arr(d_y->data<T>(), sample_size, NxC);
    EigenArrayMap<T> d_x_arr(d_x->mutable_data<T>(ctx.GetPlace()), sample_size,
                             NxC);

    d_x_arr.setZero();
    if (d_scale && d_bias) {
      for (int nc = 0; nc < NxC; ++nc) {
        int c = nc % C;
        d_bias_arr(c) += d_y_arr.col(nc).sum();
        d_scale_arr(c) +=
            ((x_arr.col(nc) - mean_arr(nc)) * inv_var_arr(nc) * d_y_arr.col(nc))
                .sum();
      }
    }
    if (!use_global_stats) {
      for (int nc = 0; nc < NxC; ++nc) {
        int c = nc % C;
        inv_var_tensor.Resize({sample_size, NxC});
        T *y_data = inv_var_tensor.mutable_data<T>(ctx.GetPlace());
        EigenArrayMap<T> y_tmp(y_data, sample_size, NxC);
        y_tmp.col(nc) = (x_arr.col(nc) - mean_arr(nc)) * inv_var_arr(nc);

        for (int hw = 0; hw < sample_size; ++hw) {
          d_x_arr(hw, nc) =
              scale_arr(c) * inv_var_arr(nc) *
              (d_y_arr(hw, nc) - d_y_arr.col(nc).mean() -
               y_tmp(hw, nc) * (d_y_arr.col(nc) * y_tmp.col(nc)).mean());
        }
      }
    } else {
      for (int nc = 0; nc < NxC; ++nc) {
        int c = nc % C;
        d_x_arr.col(nc) = scale_arr(c) * inv_var_arr(nc) * d_y_arr.col(nc);
      }
    }
  }
};

std::unique_ptr<framework::OpDesc> InstanceNormGradMaker::Apply() const {
  auto *op = new framework::OpDesc();
  op->SetType(GradOpType());
  op->SetInput("X", Input("X"));
  op->SetInput(framework::GradVarName("Y"), OutputGrad("Y"));

  op->SetInput("Scale", Input("Scale"));
  op->SetInput("Bias", Input("Bias"));
  op->SetInput("SavedMean", Output("SavedMean"));
  op->SetInput("SavedVariance", Output("SavedVariance"));

  if (boost::get<bool>(GetAttr("use_global_stats"))) {
    op->SetInput("Mean", Output("MeanOut"));
    op->SetInput("Variance", Output("VarianceOut"));
  }

  op->SetAttrMap(Attrs());
  op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
  op->SetOutput(framework::GradVarName("Scale"), InputGrad("Scale"));
  op->SetOutput(framework::GradVarName("Bias"), InputGrad("Bias"));

  return std::unique_ptr<framework::OpDesc>(op);
}

void InstanceNormDoubleGradOp::InferShape(
    framework::InferShapeContext *ctx) const {
  PADDLE_ENFORCE(ctx->HasInput("X"));
  PADDLE_ENFORCE(ctx->HasInput("Scale"), "Input(Scale) should not be null.");
  PADDLE_ENFORCE(ctx->HasInput("SavedMean"),
                 "Input(SavedMean) should not be null");
  PADDLE_ENFORCE(ctx->HasInput("SavedVariance"),
                 "Input(SavedVariance) should not be null");
  PADDLE_ENFORCE(ctx->HasInput("DDX"), "Input(DDX) should not be null.");
  PADDLE_ENFORCE(ctx->HasInput("DY"), "Input(Y@GRAD) should not be null");

  // check output
  PADDLE_ENFORCE(ctx->HasOutput("DX"), "Output(DX) should not be null");

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

  if (boost::get<bool>(GetAttr("use_global_stats"))) {
    op->SetInput("Mean", Input("Mean"));
    op->SetInput("Variance", Input("Variance"));
  }

  op->SetAttrMap(Attrs());
  op->SetOutput("DX", InputGrad("X"));
  op->SetOutput("DScale", InputGrad("Scale"));
  op->SetOutput("DDY", InputGrad(framework::GradVarName("Y")));
  return std::unique_ptr<framework::OpDesc>(op);
}

template <typename DeviceContext, typename T>
class InstanceNormDoubleGradKernel : public framework::OpKernel<T> {
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
    const float epsilon = ctx.Attr<float>("epsilon");
    const bool use_global_stats = ctx.Attr<bool>("use_global_stats");

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
    ConstEigenVectorArrayMap<T> var_arr1(inv_var_data, NxC);
    if (use_global_stats) {
      const auto *running_mean = ctx.Input<Tensor>("Mean");
      const auto *running_variance = ctx.Input<Tensor>("Variance");
      mean_data = running_mean->data<T>();
      Tensor inv_var_tensor;
      inv_var_tensor.Resize({NxC});
      T *running_inv_var_data = inv_var_tensor.mutable_data<T>(ctx.GetPlace());
      EigenVectorArrayMap<T> inv_var_tmp(running_inv_var_data, NxC);
      ConstEigenVectorArrayMap<T> var_arr(running_variance->data<T>(), NxC);

      inv_var_tmp = (var_arr + epsilon).sqrt().inverse().eval();
      inv_var_data = running_inv_var_data;
    }
    ConstEigenVectorArrayMap<T> mean_arr(mean_data, NxC);
    ConstEigenVectorArrayMap<T> inv_var_arr(inv_var_data, NxC);
    ConstEigenArrayMap<T> x_arr(X->data<T>(), sample_size, NxC);

    Tensor mean_tile;
    mean_tile.Resize({sample_size, NxC});
    mean_tile.mutable_data<T>(ctx.GetPlace());
    EigenArrayMap<T> mean_tile_data(mean_tile.mutable_data<T>(ctx.GetPlace()),
                                    sample_size, NxC);
    mean_tile_data = mean_arr.transpose().replicate(sample_size, 1);

    Tensor inv_var_tile;
    inv_var_tile.Resize({sample_size, NxC});
    inv_var_tile.mutable_data<T>(ctx.GetPlace());
    EigenArrayMap<T> inv_var_tile_data(
        inv_var_tile.mutable_data<T>(ctx.GetPlace()), sample_size, NxC);
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

    // dx = scale * (x - mean) * inv_var / HxW * (np.mean(ddx, axis=(h,w)) *
    // np.sum(dy, axis=(h,w)) -
    //      np.sum(dy * ddx, axis=(h,w)) + 3 * np.mean(dy * (x - mean),
    //      axis=(h,w)) * inv_var.pow(2) *
    //      np.sum(ddx * (x - mean), axis=(h,w))) + inv_var.pow(3) / HxW *
    //      np.sum(ddx * (x - mean)) *
    //      (np.mean(dy, axis=(h,w)) - dy) + inv_var.pow(3) / HxW * np.sum(dy,
    //      axis=(h,w)) * (x - mean) *
    //      (np.mean(ddx, axis=(h,w)) - ddx) + ddr * (dy * inv_var - inv_var *
    //      np.mean(dy, axis=(h,w)) -
    //      inv_var.pow(3) * (x - mean) * np.mean(dy * (x - mean), axis=(h,w)))

    math::SetConstant<DeviceContext, T> set_constant;

    Tensor x_sub_mean_mul_invstd;
    x_sub_mean_mul_invstd.Resize({sample_size, NxC});
    x_sub_mean_mul_invstd.mutable_data<T>(ctx.GetPlace());
    EigenArrayMap<T> x_sub_mean_mul_invstd_arr(
        x_sub_mean_mul_invstd.mutable_data<T>(ctx.GetPlace()), sample_size,
        NxC);
    x_sub_mean_mul_invstd_arr = (x_arr - mean_tile_data) * inv_var_tile_data;

    if (!use_global_stats) {
      if (dX) {
        dX->mutable_data<T>(ctx.GetPlace());
        set_constant(ctx.template device_context<DeviceContext>(), dX,
                     static_cast<T>(0));
        EigenArrayMap<T> dx_arr(dX->mutable_data<T>(ctx.GetPlace()),
                                sample_size, NxC);

        if (ddX) {
          dx_arr +=
              x_sub_mean_mul_invstd_arr * inv_var_tile_data *
              inv_var_tile_data / sample_size *
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
        // dscale = inv_var * (dy - np.mean(dy, axis=(h,w) - (x-mean) *
        //          inv_var.pow(2) * np.mean(dy * (x-mean), axis=(h,w)))) * ddx
        dScale->mutable_data<T>(ctx.GetPlace());
        set_constant(ctx.template device_context<DeviceContext>(), dScale,
                     static_cast<T>(0));
        EigenVectorArrayMap<T> dscale_arr(
            dScale->mutable_data<T>(ctx.GetPlace()), C);
        if (ddX) {
          Tensor first_grad;
          first_grad.Resize({sample_size, NxC});
          first_grad.mutable_data<T>(ctx.GetPlace());
          set_constant(ctx.template device_context<DeviceContext>(),
                       &first_grad, static_cast<T>(0));
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
        // ddy = (x - mean) * inv_var * ddscale + ddbias +
        //       scale * inv_var * (ddx - (x - mean) * inv_var.pow(2) *
        //       np.mean(ddx * (x - mean), axis=(h,w)))
        ddY->mutable_data<T>(ctx.GetPlace());
        set_constant(ctx.template device_context<DeviceContext>(), ddY,
                     static_cast<T>(0));
        EigenArrayMap<T> ddy_arr(ddY->mutable_data<T>(ctx.GetPlace()),
                                 sample_size, NxC);
        if (ddX) {
          ddy_arr +=
              scale_tile_data * inv_var_tile_data *
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
    } else {
      if (dX && ddScale) {
        ConstEigenVectorArrayMap<T> ddscale_arr(ddScale->data<T>(), C);

        Tensor ddscale_tile;
        ddscale_tile.Resize({sample_size, NxC});
        ddscale_tile.mutable_data<T>(ctx.GetPlace());
        EigenArrayMap<T> ddscale_tile_data(
            ddscale_tile.mutable_data<T>(ctx.GetPlace()), sample_size, NxC);
        ddscale_tile_data = ddscale_arr.transpose().replicate(sample_size, N);

        dX->mutable_data<T>(ctx.GetPlace());
        set_constant(ctx.template device_context<DeviceContext>(), dX,
                     static_cast<T>(0));
        EigenArrayMap<T> dx_arr(dX->mutable_data<T>(ctx.GetPlace()),
                                sample_size, NxC);

        dx_arr += dy_arr * inv_var_tile_data * ddscale_tile_data;
      }
      if (dScale && ddX) {
        dScale->mutable_data<T>(ctx.GetPlace());
        set_constant(ctx.template device_context<DeviceContext>(), dScale,
                     static_cast<T>(0));
        EigenVectorArrayMap<T> dscale_arr(
            dScale->mutable_data<T>(ctx.GetPlace()), C);

        Tensor dscale_tmp;
        dscale_tmp.Resize({NxC});
        dscale_tmp.mutable_data<T>(ctx.GetPlace());
        set_constant(ctx.template device_context<DeviceContext>(), &dscale_tmp,
                     static_cast<T>(0));
        EigenVectorArrayMap<T> dscale_tmp_data(
            dscale_tmp.mutable_data<T>(ctx.GetPlace()), NxC);

        dscale_tmp_data =
            (ddx_arr * dy_arr * inv_var_tile_data).colwise().sum();

        for (int nc = 0; nc < NxC; ++nc) {
          int c = nc % C;
          dscale_arr(c) += dscale_tmp_data(nc);
        }
      }
      if (ddY) {
        ddY->mutable_data<T>(ctx.GetPlace());
        set_constant(ctx.template device_context<DeviceContext>(), ddY,
                     static_cast<T>(0));
        EigenArrayMap<T> ddy_arr(ddY->mutable_data<T>(ctx.GetPlace()),
                                 sample_size, NxC);

        if (ddX) {
          ddy_arr += ddx_arr * scale_tile_data * inv_var_tile_data;
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

          ddy_arr +=
              x_sub_mean_mul_invstd_arr * ddscale_tile_data * ddbias_tile_data;
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(instance_norm, ops::InstanceNormOp, ops::InstanceNormOpMaker,
                  ops::InstanceNormOpInferVarType, ops::InstanceNormGradMaker);
REGISTER_OPERATOR(instance_norm_grad, ops::InstanceNormGradOp,
                  ops::InstanceNormDoubleGradMaker);
REGISTER_OPERATOR(instance_norm_grad_grad, ops::InstanceNormDoubleGradOp);

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
