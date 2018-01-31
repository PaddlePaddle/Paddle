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
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of LayerNormOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Y"),
                   "Output(Y) of LayerNormOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Mean"),
                   "Output(Mean) of LayerNormOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Variance"),
                   "Output(Variance) of LayerNormOp should not be null.");

    auto x_dim = ctx->GetInputDim("X");
    auto begin_norm_axis = ctx->Attrs().Get<int>("begin_norm_axis");
    PADDLE_ENFORCE_LT(begin_norm_axis, x_dim.size(),
                      "'begin_norm_axis' must be less than the rank of X.");

    auto matrix_dim = framework::flatten_to_2d(x_dim, begin_norm_axis);
    int left = static_cast<int>(matrix_dim[0]);
    int right = static_cast<int>(matrix_dim[1]);
    if (ctx->HasInput("Scale")) {
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("Scale").size(), 1UL);
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("Scale")[0], right);
    }
    if (ctx->HasInput("Bias")) {
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("Bias").size(), 1UL);
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("Bias")[0], right);
    }

    ctx->SetOutputDim("Y", ctx->GetInputDim("X"));
    ctx->SetOutputDim("Mean", {left});
    ctx->SetOutputDim("Variance", {left});
    ctx->ShareLoD("X", "Y");
  }
};

class LayerNormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  LayerNormOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "(LoDTensor) The input tensor.");
    AddInput("Scale",
             "(Tensor, optional) Scale is a 1-dimensional tensor of size "
             "H(`begin_norm_axis` splits the tensor(`X`) to a matrix [N,H])."
             "It is applied to the output.")
        .AsDispensable();
    AddInput("Bias",
             "(Tensor, optional) Bias is a 1-dimensional tensor of size "
             "H(`begin_norm_axis` splits the tensor(`X`) to a matrix [N,H])."
             "It is applied to the output.")
        .AsDispensable();
    AddOutput("Y", "(LoDTensor) Result after normalization.");
    AddOutput("Mean", "(Tensor) Mean of the current mini batch.")
        .AsIntermediate();
    AddOutput("Variance", "(Tensor) Variance of the current mini batch.")
        .AsIntermediate();

    AddAttr<float>("epsilon",
                   "(float, default 1e-5) Constant for "
                   "numerical stability")
        .SetDefault(1e-5)
        .AddCustomChecker([](const float &epsilon) {
          PADDLE_ENFORCE(epsilon >= 0.0f && epsilon <= 0.001f,
                         "'epsilon' should be between 0.0 and 0.001.");
        });
    AddAttr<int>("begin_norm_axis",
                 "(int default:1), the "
                 "axis of `begin_norm_axis ... Rank(X) - 1` will be "
                 "normalized. `begin_norm_axis` splits the tensor(`X`) to a "
                 "matrix [N,H].")
        .SetDefault(1)
        .AddCustomChecker([](const int &begin_norm_axis) {
          PADDLE_ENFORCE_GT(begin_norm_axis, 0,
                            "'begin_norm_axis' should be greater than zero.");
        });

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
    const auto begin_norm_axis = ctx.Attr<int>("begin_norm_axis");

    auto *output = ctx.Output<Tensor>("Y");
    auto *mean = ctx.Output<Tensor>("Mean");
    auto *var = ctx.Output<Tensor>("Variance");
    output->mutable_data<T>(ctx.GetPlace());
    mean->mutable_data<T>(ctx.GetPlace());
    var->mutable_data<T>(ctx.GetPlace());

    auto matrix_dim = framework::flatten_to_2d(x_dims, begin_norm_axis);
    int left = static_cast<int>(matrix_dim[0]);
    int right = static_cast<int>(matrix_dim[1]);

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

    auto inv_std_func = [](T ele) { return std::sqrt(1 / ele); };
    // TODO(zcd): Some thinking about output_map, is it appropriate that
    // `output_map` and `input_map` point to the same memory.
    auto inv_std = var_map.unaryExpr(inv_std_func);
    if (scale && bias) {
      auto scale_map =
          ConstEigenMatrixMapRowMajor<T>(scale->data<T>(), 1, right);
      auto bias_map = ConstEigenMatrixMapRowMajor<T>(bias->data<T>(), 1, right);
      output_map = (input_map - mean_map.replicate(1, right))
                       .cwiseProduct(inv_std.replicate(1, right))
                       .cwiseProduct(scale_map.replicate(left, 1)) +
                   bias_map.replicate(left, 1);
    } else if (scale) {
      auto scale_map =
          ConstEigenMatrixMapRowMajor<T>(scale->data<T>(), 1, right);
      output_map = (input_map - mean_map.replicate(1, right))
                       .cwiseProduct(inv_std.replicate(1, right))
                       .cwiseProduct(scale_map.replicate(left, 1));
    } else if (bias) {
      auto bias_map = ConstEigenMatrixMapRowMajor<T>(bias->data<T>(), 1, right);
      output_map = (input_map - mean_map.replicate(1, right))
                       .cwiseProduct(inv_std.replicate(1, right)) +
                   bias_map.replicate(left, 1);
    } else {
      output_map = (input_map - mean_map.replicate(1, right))
                       .cwiseProduct(inv_std.replicate(1, right));
    }
  }
};

class LayerNormGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    // check input
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of LayerNormOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Scale"),
                   "Input(Scale) of LayerNormOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Mean"),
                   "Input(Mean) of LayerNormOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Variance"),
                   "Input(Variance) of LayerNormOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Y")),
                   "Input(Y@GRAD) of LayerNormOp should not be null.");

    // check output
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    }
    if (ctx->HasOutput(framework::GradVarName("Scale"))) {
      ctx->SetOutputDim(framework::GradVarName("Scale"),
                        ctx->GetInputDim("Scale"));
    }
    if (ctx->HasOutput(framework::GradVarName("Bias"))) {
      ctx->SetOutputDim(framework::GradVarName("Bias"),
                        ctx->GetInputDim("Bias"));
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

    const auto begin_norm_axis = ctx.Attr<int>("begin_norm_axis");
    auto matrix_dim = framework::flatten_to_2d(x_dims, begin_norm_axis);
    int left = static_cast<int>(matrix_dim[0]);
    int right = static_cast<int>(matrix_dim[1]);

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
      auto d_bias_map = EigenMatrixMapRowMajor<T>(d_bias->data<T>(), 1, right);
      d_bias_map = d_y_map.colwise().sum();
    }
    if (d_scale) {
      d_scale->mutable_data<T>(ctx.GetPlace());
      auto d_scale_map =
          EigenMatrixMapRowMajor<T>(d_scale->data<T>(), 1, right);
      auto inv_std_func = [](T ele) { return std::sqrt(1 / ele); };
      // There are two equation to compute d_scale. One uses "Y" and the other
      // does not use "Y"
      d_scale_map =
          ((x_map - mean_map.replicate(1, right))
               .cwiseProduct(
                   var_map.unaryExpr(inv_std_func).replicate(1, right))
               .cwiseProduct(d_y_map))
              .colwise()
              .sum();
    }

    if (d_x) {
      d_x->mutable_data<T>(ctx.GetPlace());
      auto d_x_map = EigenMatrixMapRowMajor<T>(d_x->data<T>(), left, right);
      auto triple_product_func = [](T ele) { return ele * ele * ele; };
      auto inv_std_func = [](T ele) { return std::sqrt(1 / ele); };

      auto inv_std_map = var_map.unaryExpr(inv_std_func).eval();
      // TODO(zcd): these code can be refined
      if (d_scale) {
        auto scale_map =
            ConstEigenMatrixMapRowMajor<T>(scale->data<T>(), 1, right);
        // dy_dx
        auto dx_end =
            inv_std_map.replicate(1, right).cwiseProduct(d_y_map).cwiseProduct(
                scale_map.replicate(left, 1));

        // dy_dmean_dx
        auto dx_mean =
            (T(-1.0) / right) * dx_end.rowwise().sum().replicate(1, right);

        // dy_var_dx
        auto dvar_end_part = (x_map - mean_map.replicate(1, right))
                                 .cwiseProduct(scale_map.replicate(left, 1))
                                 .cwiseProduct(d_y_map)
                                 .rowwise()
                                 .sum();
        auto dvar_end = inv_std_map.unaryExpr(triple_product_func)
                            .cwiseProduct(dvar_end_part)
                            .replicate(1, right);
        auto dx_var =
            (T(-1.0) / right) *
            (x_map - mean_map.replicate(1, right)).cwiseProduct(dvar_end);

        d_x_map = dx_end + dx_mean + dx_var;
      } else {
        // dy_dx
        auto dx_end = inv_std_map.replicate(1, right).cwiseProduct(d_y_map);

        // dy_dmean_dx
        auto dx_mean =
            (T(-1.0) / right) * dx_end.rowwise().sum().replicate(1, right);

        // dy_var_dx
        auto dvar_end_part = (x_map - mean_map.replicate(1, right))
                                 .cwiseProduct(d_y_map)
                                 .rowwise()
                                 .sum();
        auto dvar_end = inv_std_map.unaryExpr(triple_product_func)
                            .cwiseProduct(dvar_end_part)
                            .replicate(1, right);
        auto dx_var =
            (T(-1.0) / right) *
            (x_map - mean_map.replicate(1, right)).cwiseProduct(dvar_end);

        d_x_map = dx_end + dx_mean + dx_var;
      }
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
