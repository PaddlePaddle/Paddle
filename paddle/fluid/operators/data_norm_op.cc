/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/data_norm_op.h"
#include <memory>
#include <string>
#include "paddle/fluid/framework/data_layout.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using DataLayout = framework::DataLayout;

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

class DataNormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "");
    PADDLE_ENFORCE(ctx->HasInput("BatchSize"), "");
    PADDLE_ENFORCE(ctx->HasInput("BatchSum"), "");
    PADDLE_ENFORCE(ctx->HasInput("BatchSquareSum"), "");
    PADDLE_ENFORCE(ctx->HasOutput("Means"), "");
    PADDLE_ENFORCE(ctx->HasOutput("Scales"), "");
    PADDLE_ENFORCE(ctx->HasOutput("Y"), "");

    const auto x_dims = ctx->GetInputDim("X");
    const DataLayout data_layout = framework::StringToDataLayout(
        ctx->Attrs().Get<std::string>("data_layout"));

    PADDLE_ENFORCE(x_dims.size() >= 2 && x_dims.size() <= 5,
                   "Input X must have 2 to 5 dimensions.");

    const int64_t C =
        (data_layout == DataLayout::kNCHW ? x_dims[1]
                                          : x_dims[x_dims.size() - 1]);

    PADDLE_ENFORCE_EQ(ctx->GetInputDim("BatchSize").size(), 1UL);
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("BatchSum").size(), 1UL);
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("BatchSquareSum").size(), 1UL);
    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("BatchSize")[0], C);
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("BatchSum")[0], C);
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("BatchSquareSum")[0], C);
    }

    ctx->SetOutputDim("Y", x_dims);
    ctx->SetOutputDim("Means", {C});
    ctx->SetOutputDim("Scales", {C});
    ctx->ShareLoD("X", "Y");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    // By default, the type of the scale, bias, mean,
    // and var tensors should both be float. (For float or float16 input tensor)
    // or double (For double input tensor).
    auto dn_param_type = framework::proto::VarType::FP32;
    if (input_data_type == framework::proto::VarType::FP64) {
      dn_param_type = framework::proto::VarType::FP64;
    }
    PADDLE_ENFORCE_EQ(dn_param_type,
                      OperatorWithKernel::IndicateVarDataType(ctx, "BatchSize"),
                      "BatchSize input should be of float type");
    PADDLE_ENFORCE_EQ(dn_param_type,
                      OperatorWithKernel::IndicateVarDataType(ctx, "BatchSum"),
                      "BatchSum input should be of float type");
    PADDLE_ENFORCE_EQ(dn_param_type, OperatorWithKernel::IndicateVarDataType(
                                         ctx, "BatchSquareSum"),
                      "BatchSquareSum input should be of float type");

    // TODO(pzelazko-intel): enable MKLDNN layout when it's ready
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
#ifdef PADDLE_WITH_MKLDNN
    if (library == framework::LibraryType::kPlain &&
        platform::CanMKLDNNBeUsed(ctx)) {
      library = framework::LibraryType::kMKLDNN;
      layout = framework::DataLayout::kMKLDNN;
    }
#endif

    return framework::OpKernelType(input_data_type, ctx.GetPlace(), layout,
                                   library);
  }
};

class DataNormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    // AddAttr<bool>("is_test", "").SetDefault(false);
    AddAttr<float>("epsilon", "")
        .SetDefault(1e-4)
        .AddCustomChecker([](const float &epsilon) {
          PADDLE_ENFORCE(epsilon >= 0.0f && epsilon <= 0.001f,
                         "'epsilon' should be between 0.0 and 0.001.");
        });
    AddAttr<int>("slot_dim",
                 "(int, default -1) Dimension of one slot if set, "
                 "when the input is concated by slot-wise embeddings")
        .SetDefault(-1);
    AddAttr<float>(
        "summary_decay_rate",
        "(float, default 0.9999999) The decay rate when update the summary")
        .SetDefault(0.9999999);
    AddAttr<std::string>("data_layout", "").SetDefault("NCHW");
    AddAttr<bool>("sync_stats", "(bool, default false) only used in multi-GPU")
        .SetDefault(false);
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false);
    AddInput("X", "The input tensor");
    AddInput("BatchSize",
             "BatchSize is a 1-dimensional tensor of size C "
             "that is applied to the output");
    AddInput("BatchSum",
             "BatchSum is a 1-dimensional tensor of size C "
             "that is applied to the output");
    AddInput("BatchSquareSum",
             "The global BatchSquareSum (for training) or "
             "estimated BatchSquareSum (for testing)");
    AddOutput("Y", "result after normalization");
    AddOutput("Means",
              "Mean of the history data batch, "
              "will apply to output when training")
        .AsIntermediate();
    AddOutput("Scales",
              "Scales of the history data batch, "
              "will apply to output when training")
        .AsIntermediate();
    AddComment(R"DOC(
Data Normalization.

Can be used as a normalizer function for data
The required data format for this layer is one of the following:
1. NHWC `[batch, in_height, in_width, in_channels]`
2. NCHW `[batch, in_channels, in_height, in_width]`

)DOC");
  }
};

template <typename T>
class DataNormKernel<platform::CPUDeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    // const bool is_test = ctx.Attr<bool>("is_test");
    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);

    const auto *x = ctx.Input<Tensor>("X");
    const auto &x_dims = x->dims();
    PADDLE_ENFORCE(x_dims.size() == 2, "The Input dim size should be 2");
    const int N = x_dims[0];
    const int C =
        (data_layout == DataLayout::kNCHW ? x_dims[1]
                                          : x_dims[x_dims.size() - 1]);
    auto *y = ctx.Output<Tensor>("Y");
    auto *mean_out = ctx.Output<Tensor>("Means");
    auto *scales = ctx.Output<Tensor>("Scales");

    // alloc memory
    T *y_data = y->mutable_data<T>(ctx.GetPlace());

    Eigen::Array<T, Eigen::Dynamic, 1> inv_std(C);
    ConstEigenVectorArrayMap<T> b_size_arr(
        ctx.Input<Tensor>("BatchSize")->data<T>(), C);
    ConstEigenVectorArrayMap<T> b_sum_arr(
        ctx.Input<Tensor>("BatchSum")->data<T>(), C);
    ConstEigenVectorArrayMap<T> b_square_sum_arr(
        ctx.Input<Tensor>("BatchSquareSum")->data<T>(), C);
    EigenVectorArrayMap<T> means_arr(mean_out->mutable_data<T>(ctx.GetPlace()),
                                     C);
    EigenVectorArrayMap<T> scales_arr(scales->mutable_data<T>(ctx.GetPlace()),
                                      C);
    means_arr = b_sum_arr / b_size_arr;
    scales_arr = (b_size_arr / b_square_sum_arr).sqrt();

    const T *means_data = mean_out->data<T>();
    const T *x_data = x->data<T>();
    const T *scales_data = scales->data<T>();
    const int slot_dim = ctx.Attr<int>("slot_dim");
    T min_precision = 1e-7f;
    switch (data_layout) {
      case DataLayout::kNCHW:  // It's two dimensions, so make no difference
      case DataLayout::kNHWC: {
        // if slot_dim is set and batch size is larger than zero, we choose
        // to check if show number is zero, if so, skip normalization.
        if (slot_dim > 0 && N > 0) {
          const int item_size = x->numel() / N;
          // location of show number in one embedding
          int offset = 0;
          for (int k = 0; k < N; ++k) {
            for (int i = 0; i < item_size; i += slot_dim) {
              if (x_data[offset + i] > -min_precision &&
                  x_data[offset + i] < min_precision) {
                // show = 0
                memset(y_data + offset + i, 0, sizeof(T) * slot_dim);
              } else {
                for (int j = i; j < i + slot_dim; ++j) {
                  y_data[offset + j] =
                      (x_data[offset + j] - means_data[j]) * scales_data[j];
                }
              }
            }

            offset += item_size;
          }
        } else {
          EigenArrayMap<T>(y_data, C, N) =
              (ConstEigenArrayMap<T>(x->data<T>(), C, N).colwise() - means_arr)
                  .colwise() *
              scales_arr;
        }
        break;
      }
      default:
        PADDLE_THROW("Unknown storage order: %d", data_layout);
    }
  }
};

class DataNormGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    // check input
    PADDLE_ENFORCE(ctx->HasInput("X"));
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Y")), "");
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("BatchSize"), true,
        platform::errors::NotFound(
            "Output(BatchSize) of DataNormGradOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("BatchSum"), true,
        platform::errors::NotFound(
            "Output(BatchSum) of DataNormGradOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("BatchSquareSum"), true,
        platform::errors::NotFound(
            "Output(BatchSquareSum) of DataNormGradOp should not be null."));
    PADDLE_ENFORCE(ctx->HasInput("Means"), "");
    PADDLE_ENFORCE(ctx->HasInput("Scales"), "");

    // check output
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("BatchSize")), "");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("BatchSum")), "");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("BatchSquareSum")),
                   "");

    const auto x_dims = ctx->GetInputDim("X");
    const DataLayout data_layout = framework::StringToDataLayout(
        ctx->Attrs().Get<std::string>("data_layout"));
    const int C =
        (data_layout == DataLayout::kNCHW ? x_dims[1]
                                          : x_dims[x_dims.size() - 1]);

    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    }
    ctx->SetOutputDim(framework::GradVarName("BatchSize"), {C});
    ctx->SetOutputDim(framework::GradVarName("BatchSum"), {C});
    ctx->SetOutputDim(framework::GradVarName("BatchSquareSum"), {C});
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

    // TODO(pzelazko-intel): enable MKLDNN layout when it's ready
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;

#ifdef PADDLE_WITH_MKLDNN
    if (library == framework::LibraryType::kPlain &&
        platform::CanMKLDNNBeUsed(ctx)) {
      library = framework::LibraryType::kMKLDNN;
      layout = framework::DataLayout::kMKLDNN;
    }
#endif

    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace(),
        layout, library);
  }
};

template <typename T>
class DataNormGradKernel<platform::CPUDeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *x = ctx.Input<Tensor>("X");
    const auto *d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const auto *scales = ctx.Input<Tensor>("Scales");
    const auto *means = ctx.Input<Tensor>("Means");

    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);

    // Get the size for each dimension.
    // NCHW [batch_size, in_channels, in_height, in_width]
    const auto &x_dims = x->dims();
    PADDLE_ENFORCE(x_dims.size() == 2, "The Input dim size should be 2");
    const int N = x_dims[0];
    const int C =
        (data_layout == DataLayout::kNCHW ? x_dims[1]
                                          : x_dims[x_dims.size() - 1]);

    // init output
    Tensor *d_x = nullptr;
    if (ctx.HasOutput(framework::GradVarName("X"))) {
      d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    }
    auto *d_batch_size =
        ctx.Output<Tensor>(framework::GradVarName("BatchSize"));
    auto *d_batch_sum = ctx.Output<Tensor>(framework::GradVarName("BatchSum"));
    auto *d_batch_square_sum =
        ctx.Output<Tensor>(framework::GradVarName("BatchSquareSum"));

    T *d_batch_size_data = d_batch_size->mutable_data<T>(ctx.GetPlace());
    T *d_batch_sum_data = d_batch_sum->mutable_data<T>(ctx.GetPlace());
    T *d_batch_square_sum_data =
        d_batch_square_sum->mutable_data<T>(ctx.GetPlace());
    EigenVectorArrayMap<T> d_batch_size_arr(d_batch_size_data, C);
    EigenVectorArrayMap<T> d_batch_sum_arr(d_batch_sum_data, C);
    EigenVectorArrayMap<T> d_batch_square_sum_arr(d_batch_square_sum_data, C);

    d_batch_size_arr.setZero();
    d_batch_sum_arr.setZero();
    d_batch_square_sum_arr.setZero();
    const T *x_data = x->data<T>();
    const T *means_data = means->data<T>();

    const float epsilon = ctx.Attr<float>("epsilon");
    T min_precision = 1e-7f;
    const int slot_dim = ctx.Attr<int>("slot_dim");
    switch (data_layout) {  // it's two dimensions, make no difference
      case DataLayout::kNCHW:
      case DataLayout::kNHWC: {
        ConstEigenVectorArrayMap<T> scales_arr(scales->data<T>(), C);
        ConstEigenVectorArrayMap<T> means_arr(means->data<T>(), C);
        ConstEigenArrayMap<T> x_arr(x->data<T>(), C, N);
        ConstEigenArrayMap<T> d_y_arr(d_y->data<T>(), C, N);
        if (d_x != nullptr) {
          EigenArrayMap<T> d_x_arr(d_x->mutable_data<T>(ctx.GetPlace()), C, N);
          d_x_arr.setZero();
          for (int nc = 0; nc < N; ++nc) {
            d_x_arr.col(nc) = d_y_arr.col(nc) * scales_arr;
          }
        }

        if (slot_dim > 0 && N > 0) {
          // if slot_dim is set and batch size is larger than zero, we choose
          // to check if show number is zero, if so, skip update statistics.
          int offset = 0;
          const int item_size = x->numel() / N;
          for (int k = 0; k < N; ++k) {
            for (int i = 0; i < item_size; i += slot_dim) {
              if (!(x_data[offset + i] > -min_precision &&
                    x_data[offset + i] < min_precision)) {
                // show != 0
                for (int j = i; j < i + slot_dim; ++j) {
                  d_batch_size_data[j] += 1;
                  d_batch_sum_data[j] += x_data[offset + j];
                  d_batch_square_sum_data[j] +=
                      (x_data[offset + j] - means_data[j]) *
                      (x_data[offset + j] - means_data[j]);
                }
              }
            }
            offset += item_size;
          }

          for (int i = 0; i < item_size; i += slot_dim) {
            for (int j = i; j < i + slot_dim; ++j) {
              if (d_batch_size_data[j] >= 1) {
                d_batch_sum_data[j] /= d_batch_size_data[j];
                d_batch_square_sum_data[j] =
                    d_batch_square_sum_data[j] / d_batch_size_data[j] +
                    d_batch_size_data[j] * epsilon;
                d_batch_size_data[j] = 1;
              }
            }
          }
        } else {
          // calculate data sum and squre sum
          Eigen::Array<T, Eigen::Dynamic, 1> sample_sum(C);
          Eigen::Array<T, Eigen::Dynamic, 1> sample_square_sum(C);
          // calculate data sample sum and square sum
          sample_sum.setZero();
          sample_square_sum.setZero();
          for (int nc = 0; nc < N; ++nc) {
            sample_sum += x_arr.col(nc);
            sample_square_sum += (x_arr.col(nc) - means_arr).square();
          }
          // calculate gradient
          d_batch_size_arr.setConstant(N);
          d_batch_sum_arr = sample_sum;
          d_batch_square_sum_arr =
              sample_square_sum + d_batch_size_arr * epsilon;
        }
        break;
      }
      default:
        PADDLE_THROW("Unknown storage order: %s", data_layout_str);
    }
  }
};

template <typename T>
class DataNormGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  std::unique_ptr<T> Apply() const override {
    auto *op = new T();
    op->SetType("data_norm_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));

    op->SetOutput("BatchSize", this->Input("BatchSize"));
    op->SetOutput("BatchSum", this->Input("BatchSum"));
    op->SetOutput("BatchSquareSum", this->Input("BatchSquareSum"));
    op->SetInput("Scales", this->Output("Scales"));
    op->SetInput("Means", this->Output("Means"));

    op->SetAttrMap(this->Attrs());

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("BatchSize"),
                  this->InputGrad("BatchSize"));
    op->SetOutput(framework::GradVarName("BatchSum"),
                  this->InputGrad("BatchSum"));
    op->SetOutput(framework::GradVarName("BatchSquareSum"),
                  this->InputGrad("BatchSquareSum"));

    return std::unique_ptr<T>(op);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(data_norm, ops::DataNormOp, ops::DataNormOpMaker,
                  ops::DataNormGradMaker<paddle::framework::OpDesc>,
                  ops::DataNormGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(data_norm_grad, ops::DataNormGradOp);

REGISTER_OP_CPU_KERNEL(
    data_norm, ops::DataNormKernel<paddle::platform::CPUDeviceContext, float>,
    ops::DataNormKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    data_norm_grad,
    ops::DataNormGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::DataNormGradKernel<paddle::platform::CPUDeviceContext, double>);
