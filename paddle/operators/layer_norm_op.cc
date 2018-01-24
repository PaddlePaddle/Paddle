/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/layer_norm_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using DataLayout = framework::DataLayout;

template <typename T>
using EigenMatrixMapRowMajor = Eigen::Map<
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
template <typename T>
using ConstEigenMatrixMapRowMajor = Eigen::Map<
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

class LayerNormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "");
    PADDLE_ENFORCE(ctx->HasInput("Scale"), "");
    PADDLE_ENFORCE(ctx->HasInput("Bias"), "");
    PADDLE_ENFORCE(ctx->HasOutput("Y"), "");

    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Scale").size(), 1UL);
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Scale")[0], 1);
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Bias").size(), 1UL);
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Bias")[0], 1);

    ctx->SetOutputDim("Y", ctx->GetInputDim("X"));
    ctx->SetOutputDim("Mean", {ctx->GetInputDim("X")[0]});
    ctx->SetOutputDim("Variance", {ctx->GetInputDim("X")[0]});

    ctx->ShareLoD("X", "Y");
  }
};

class LayerNormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  LayerNormOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The input tensor");
    AddInput("Scale",
             "Scale is a 1-dimensional tensor of size 1 "
             "that is applied to the output");
    AddInput("Bias",
             "Bias is a 1-dimensional tensor of size 1 "
             "that is applied to the output");
    AddOutput("Y", "result after normalization");
    AddOutput("Mean", "Mean of the current mini batch.");
    AddOutput("Variance", "Variance of the current mini batch.");

    AddAttr<float>("epsilon", "")
        .SetDefault(1e-5)
        .AddCustomChecker([](const float &epsilon) {
          PADDLE_ENFORCE(epsilon >= 0.0f && epsilon <= 0.001f,
                         "'epsilon' should be between 0.0 and 0.001.");
        });
    AddAttr<std::vector<int>>("axis",
                              "(vector<int> default:{1, 1, 1}), the "
                              "axis to normalize.")
        .SetDefault({1, 2, 3});  // todo(zcd) : who to set axis

    AddComment(R"DOC(
Layer Normalization.

Layer Norm has been implemented as discussed in the paper:
https://arxiv.org/abs/1607.06450
...
)DOC");
  }
};

template <typename T>
class LayerNormKernel<platform::CPUDeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const float epsilon = ctx.Attr<float>("epsilon");
    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *bias = ctx.Input<Tensor>("Bias");
    const auto *x = ctx.Input<Tensor>("X");
    const auto &x_dims = x->dims();

    const int N = x_dims[0];
    const int sample_size = x->numel() / N;

    auto scale_data = scale->data<T>()[0];
    auto bias_data = bias->data<T>()[0];

    auto *output = ctx.Output<Tensor>("Y");
    auto *mean = ctx.Output<Tensor>("Mean");
    auto *var = ctx.Output<Tensor>("Variance");
    output->mutable_data<T>(ctx.GetPlace());
    mean->mutable_data<T>(ctx.GetPlace());
    var->mutable_data<T>(ctx.GetPlace());

    int left = N, right = sample_size;
    auto input_map = ConstEigenMatrixMapRowMajor<T>(x->data<T>(), left, right);
    auto mean_map = EigenMatrixMapRowMajor<T>(mean->data<T>(), left, 1);
    auto var_map = EigenMatrixMapRowMajor<T>(var->data<T>(), left, 1);
    auto output_map = EigenMatrixMapRowMajor<T>(output->data<T>(), left, right);

    auto squre = [](T ele) { return ele * ele; };
    auto add_epslion = [epsilon](T ele) { return ele + epsilon; };

    mean_map = input_map.rowwise().mean();
    var_map = (input_map - mean_map.replicate(1, right))
                  .unaryExpr(squre)
                  .rowwise()
                  .mean()
                  .unaryExpr(add_epslion);

    auto scale_inv_std = [scale_data](T ele) {
      return std::sqrt(1 / ele) * scale_data;
    };
    auto sub_bias = [bias_data](T ele) { return bias_data - ele; };

    output_map = (var_map.unaryExpr(scale_inv_std).replicate(1, right))
                     .cwiseProduct(input_map) +
                 var_map.unaryExpr(scale_inv_std)
                     .cwiseProduct(mean_map)
                     .unaryExpr(sub_bias)
                     .replicate(1, right);
  }
};

class LayerNormGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    // check input
    PADDLE_ENFORCE(ctx->HasInput("X"));
    PADDLE_ENFORCE(ctx->HasInput("Scale"), "");
    PADDLE_ENFORCE(ctx->HasInput("Mean"), "");
    PADDLE_ENFORCE(ctx->HasInput("Variance"), "");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Y")), "");

    const auto x_dims = ctx->GetInputDim("X");

    // check output
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    }
    if (ctx->HasOutput(framework::GradVarName("Scale"))) {
      ctx->SetOutputDim(framework::GradVarName("Scale"), {1});
    }
    if (ctx->HasOutput(framework::GradVarName("Bias"))) {
      ctx->SetOutputDim(framework::GradVarName("Bias"), {1});
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    const auto *var = ctx.InputVar(framework::GradVarName("Y"));
    if (var == nullptr) {
      PADDLE_THROW("can't find Y@GRAD");
    }
    const Tensor *t = nullptr;
    if (var->IsType<Tensor>()) {
      t = &var->Get<Tensor>();
    } else if (var->IsType<LoDTensor>()) {
      t = &var->Get<LoDTensor>();
    }
    if (t == nullptr) {
      PADDLE_THROW("can't find Y@GRAD");
    }
    return framework::OpKernelType(framework::ToDataType(t->type()),
                                   ctx.GetPlace());
  }
};

template <typename T>
class LayerNormGradKernel<platform::CPUDeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *x = ctx.Input<Tensor>("X");
    const auto *mean = ctx.Input<Tensor>("Mean");
    const auto *var = ctx.Input<Tensor>("Variance");
    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));

    const auto &x_dims = x->dims();
    const int N = x_dims[0];
    const int sample_size = x->numel() / N;
    int left = N, right = sample_size;

    auto scale_data = scale->data<T>()[0];

    // init output
    auto *d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *d_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto *d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    auto x_map = ConstEigenMatrixMapRowMajor<T>(x->data<T>(), left, right);
    auto d_y_map = ConstEigenMatrixMapRowMajor<T>(d_y->data<T>(), left, right);
    auto mean_map = ConstEigenMatrixMapRowMajor<T>(mean->data<T>(), left, 1);
    auto var_map = ConstEigenMatrixMapRowMajor<T>(var->data<T>(), left, 1);

    if (d_bias) {
      d_bias->mutable_data<T>(ctx.GetPlace());
      d_bias->data<T>()[0] = d_y_map.sum();
    }
    if (d_scale) {
      d_scale->mutable_data<T>(ctx.GetPlace());
      auto inv_std = [](T ele) { return std::sqrt(1 / ele); };
      d_scale->data<T>()[0] =
          ((x_map - mean_map.replicate(1, right))
               .cwiseProduct(var_map.unaryExpr(inv_std).replicate(1, right))
               .cwiseProduct(d_y_map))
              .sum();  // also can use `y` to get d_scale_map
    }

    if (d_x) {
      d_x->mutable_data<T>(ctx.GetPlace());
      auto d_x_map = EigenMatrixMapRowMajor<T>(d_x->data<T>(), left, right);
      auto triple_product = [](T ele) { return ele * ele; };
      auto neg_inv_std = [](T ele) { return -std::sqrt(1 / ele); };
      auto inv_std_scale_func = [scale_data](T ele) {
        return std::sqrt(1 / ele) * scale_data;
      };
      auto neg_inv_std_scale_func = [scale_data](T ele) {
        return -std::sqrt(1 / ele) * scale_data;
      };
      // dy_dx
      auto dx_end = var_map.unaryExpr(inv_std_scale_func)
                        .replicate(1, right)
                        .cwiseProduct(d_y_map);
      // dy_dmean_dx
      auto dmean_end = var_map.unaryExpr(neg_inv_std_scale_func)
                           .replicate(1, right)
                           .cwiseProduct(d_y_map)
                           .rowwise()
                           .sum();
      auto dx_mean = (T(1.0) / right) * dmean_end.replicate(1, right);
      // dy_var_dx
      auto dvar_end_0 = (x_map - mean_map.replicate(1, right))
                            .cwiseProduct(d_y_map)
                            .rowwise()
                            .sum();
      auto dvar_end = var_map.unaryExpr(neg_inv_std)
                          .unaryExpr(triple_product)
                          .cwiseProduct(dvar_end_0);
      auto dx_var = (T(1.0) / right) *
                    (x_map - mean_map.replicate(1, right))
                        .cwiseProduct(dvar_end.replicate(1, right));

      // d_x = (1. / N) * scale * inv_var * (N * d_y - np.sum(d_y, axis=0)
      //   - (X - mean) * inv_var * inv_var * np.sum(d_y * (X - mean), axis=0))

      d_x_map = dx_end + dx_mean + dx_var;
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(layer_norm, ops::LayerNormOp, ops::LayerNormOpMaker,
            layer_norm_grad, ops::LayerNormGradOp);
REGISTER_OP_CPU_KERNEL(
    layer_norm,
    ops::LayerNormKernel<paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(
    layer_norm_grad,
    ops::LayerNormGradKernel<paddle::platform::CPUDeviceContext, float>);
