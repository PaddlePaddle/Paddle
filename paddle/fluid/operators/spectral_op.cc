// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/spectral_op.h"
#include "paddle/fluid/operators/spectral_helper.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

// FFTC2C
class FFTC2COpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), the input tensor of fft_c2c op.");
    AddOutput("Out", "(Tensor), the output tensor of fft_c2c op.");
    AddAttr<std::vector<int64_t>>("axes",
                                  "std::vector<int64_t>, the fft axes.");
    AddAttr<std::string>("normalization",
                         "fft_norm_type, the fft normalization type.");
    AddAttr<bool>("forward", "bool, the fft direction.");
    AddComment(R"DOC(
      Compute complex to complex FFT.
    )DOC");
  }
};

class FFTC2COp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "fft_c2c");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "fft_c2c");
    const auto axes = ctx->Attrs().Get<std::vector<int64_t>>("axes");
    const auto x_dim = ctx->GetInputDim("X");
    for (size_t i = 0; i < axes.size(); i++) {
      PADDLE_ENFORCE_GT(x_dim[axes[i]], 0,
                        platform::errors::InvalidArgument(
                            "Invalid fft n-point (%d).", x_dim[axes[i]]));
    }
    ctx->ShareDim("X", /*->*/ "Out");  // only for c2c
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    const auto in_dtype = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    const auto kernel_dtype = framework::ToRealType(in_dtype);
    return framework::OpKernelType(kernel_dtype, ctx.GetPlace());
  }
};

template <typename T>
class FFTC2CGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("fft_c2c_grad");
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

class FFTC2CGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    const auto out_grad_name = framework::GradVarName("Out");
    OP_INOUT_CHECK(ctx->HasInput(out_grad_name), "Input", out_grad_name,
                   "fft_c2c_grad");
    const auto x_grad_name = framework::GradVarName("X");
    OP_INOUT_CHECK(ctx->HasOutput(x_grad_name), "Output", x_grad_name,
                   "fft_c2c_grad");

    ctx->SetOutputDim(x_grad_name, ctx->GetInputDim(out_grad_name));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    const auto in_dtype = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    const auto kernel_dtype = framework::ToRealType(in_dtype);
    return framework::OpKernelType(kernel_dtype, ctx.GetPlace());
  }
};

// FFTR2C
class FFTR2COpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), the input tensor of fft_r2c op.");
    AddOutput("Out", "(Tensor), the output tensor of fft_r2c op.");
    AddAttr<std::vector<int64_t>>("axes",
                                  "std::vector<int64_t>, the fft axes.");
    AddAttr<std::string>("normalization",
                         "fft_norm_type, the fft normalization type.");
    AddAttr<bool>("forward", "bool, the fft direction.");
    AddAttr<bool>("onesided", "bool, perform onesided fft.");
    AddComment(R"DOC(
      Compute real to complex FFT.
    )DOC");
  }
};

class FFTR2COp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "fft_r2c");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "fft_r2c");
    const auto axes = ctx->Attrs().Get<std::vector<int64_t>>("axes");
    const auto x_dim = ctx->GetInputDim("X");
    for (size_t i = 0; i < axes.size() - 1L; i++) {
      PADDLE_ENFORCE_GT(x_dim[axes[i]], 0,
                        platform::errors::InvalidArgument(
                            "Invalid fft n-point (%d).", x_dim[axes[i]]));
    }

    const bool onesided = ctx->Attrs().Get<bool>("onesided");
    if (!onesided) {
      ctx->ShareDim("X", /*->*/ "Out");
    } else {
      framework::DDim out_dim(ctx->GetInputDim("X"));
      const int64_t last_fft_axis = axes.back();
      const int64_t last_fft_dim_size = out_dim.at(last_fft_axis);
      out_dim.at(last_fft_axis) = last_fft_dim_size / 2 + 1;
      ctx->SetOutputDim("Out", out_dim);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    const auto in_dtype = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(in_dtype, ctx.GetPlace());
  }
};

template <typename T>
class FFTR2CGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("fft_r2c_grad");
    grad_op->SetInput("X", this->Input("X"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

class FFTR2CGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    const auto out_grad_name = framework::GradVarName("Out");
    OP_INOUT_CHECK(ctx->HasInput(out_grad_name), "Input", out_grad_name,
                   "fft_r2c_grad");
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "fft_r2c_grad");

    const auto x_grad_name = framework::GradVarName("X");
    OP_INOUT_CHECK(ctx->HasOutput(x_grad_name), "Output", x_grad_name,
                   "fft_r2c_grad");

    ctx->ShareDim("X", /*->*/ x_grad_name);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    const auto in_dtype = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    const auto kernel_dtype = framework::ToRealType(in_dtype);
    return framework::OpKernelType(kernel_dtype, ctx.GetPlace());
  }
};

// FFTC2R
class FFTC2ROpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), the input tensor of fft_c2r op.");
    AddOutput("Out", "(Tensor), the output tensor of fft_c2r op.");
    AddAttr<std::vector<int64_t>>("axes",
                                  "std::vector<int64_t>, the fft axes.");
    AddAttr<std::string>("normalization",
                         "fft_norm_type, the fft normalization type.");
    AddAttr<bool>("forward", "bool, the fft direction.");
    AddAttr<int64_t>(
        "last_dim_size", "int",
        "Length of the transformed "
        "axis of the output. For n output points, last_dim_size//2 + 1 input"
        " points are necessary. If the input is longer than this,"
        " it is cropped. If it is shorter than this, it is padded"
        " with zeros. If last_dim_size is not given, it is taken to be 2*(m-1)"
        " where m is the length of the input along the axis "
        "specified by axis.")
        .SetDefault(0L);
    AddComment(R"DOC(
      Compute complex to complex FFT.
    )DOC");
  }
};

class FFTC2ROp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "fft_c2r");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "fft_c2r");

    const auto axes = ctx->Attrs().Get<std::vector<int64_t>>("axes");
    const auto x_dim = ctx->GetInputDim("X");
    for (size_t i = 0; i < axes.size() - 1L; i++) {
      PADDLE_ENFORCE_GT(x_dim[axes[i]], 0,
                        platform::errors::InvalidArgument(
                            "Invalid fft n-point (%d).", x_dim[axes[i]]));
    }

    const int64_t last_dim_size = ctx->Attrs().Get<int64_t>("last_dim_size");
    framework::DDim out_dim(ctx->GetInputDim("X"));
    const int64_t last_fft_axis = axes.back();
    if (last_dim_size == 0) {
      const int64_t last_fft_dim_size = out_dim.at(last_fft_axis);
      const int64_t fft_n_point = (last_fft_dim_size - 1) * 2;
      PADDLE_ENFORCE_GT(fft_n_point, 0,
                        platform::errors::InvalidArgument(
                            "Invalid fft n-point (%d).", fft_n_point));
      out_dim.at(last_fft_axis) = fft_n_point;
    } else {
      PADDLE_ENFORCE_GT(last_dim_size, 0,
                        platform::errors::InvalidArgument(
                            "Invalid fft n-point (%d).", last_dim_size));
      out_dim.at(last_fft_axis) = last_dim_size;
    }
    ctx->SetOutputDim("Out", out_dim);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    const auto in_dtype = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    const auto kernel_dtype = framework::ToRealType(in_dtype);
    return framework::OpKernelType(kernel_dtype, ctx.GetPlace());
  }
};

template <typename T>
class FFTC2RGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("fft_c2r_grad");
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

class FFTC2RGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    const auto out_grad_name = framework::GradVarName("Out");
    OP_INOUT_CHECK(ctx->HasInput(out_grad_name), "Input", out_grad_name,
                   "fft_c2r_grad");

    const auto x_grad_name = framework::GradVarName("X");
    OP_INOUT_CHECK(ctx->HasOutput(x_grad_name), "Output", x_grad_name,
                   "fft_c2r_grad");

    const auto axes = ctx->Attrs().Get<std::vector<int64_t>>("axes");

    const auto out_grad_dim = ctx->GetInputDim(out_grad_name);
    framework::DDim x_grad_dim(out_grad_dim);
    const int64_t last_fft_axis = axes.back();
    const int64_t last_fft_dim_size = x_grad_dim.at(last_fft_axis);
    x_grad_dim.at(last_fft_axis) = last_fft_dim_size / 2 + 1;
    ctx->SetOutputDim(x_grad_name, x_grad_dim);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    const auto in_dtype = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    return framework::OpKernelType(in_dtype, ctx.GetPlace());
  }
};

// common functions
FFTNormMode get_norm_from_string(const std::string& norm, bool forward) {
  if (norm.empty() || norm == "backward") {
    return forward ? FFTNormMode::none : FFTNormMode::by_n;
  }

  if (norm == "forward") {
    return forward ? FFTNormMode::by_n : FFTNormMode::none;
  }

  if (norm == "ortho") {
    return FFTNormMode::by_sqrt_n;
  }

  PADDLE_THROW(platform::errors::InvalidArgument(
      "FFT norm string must be 'forward' or 'backward' or 'ortho', "
      "received %s",
      norm));
}

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(fft_c2c, ops::FFTC2COp, ops::FFTC2COpMaker,
                  ops::FFTC2CGradOpMaker<paddle::framework::OpDesc>,
                  ops::FFTC2CGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(
    fft_c2c, ops::FFTC2CKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FFTC2CKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OPERATOR(fft_c2c_grad, ops::FFTC2CGradOp);
REGISTER_OP_CPU_KERNEL(
    fft_c2c_grad,
    ops::FFTC2CGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FFTC2CGradKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OPERATOR(fft_r2c, ops::FFTR2COp, ops::FFTR2COpMaker,
                  ops::FFTR2CGradOpMaker<paddle::framework::OpDesc>,
                  ops::FFTR2CGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(
    fft_r2c, ops::FFTR2CKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FFTR2CKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OPERATOR(fft_r2c_grad, ops::FFTR2CGradOp);
REGISTER_OP_CPU_KERNEL(
    fft_r2c_grad,
    ops::FFTR2CGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FFTR2CGradKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OPERATOR(fft_c2r, ops::FFTC2ROp, ops::FFTC2ROpMaker,
                  ops::FFTC2RGradOpMaker<paddle::framework::OpDesc>,
                  ops::FFTC2RGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(
    fft_c2r, ops::FFTC2RKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FFTC2RKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OPERATOR(fft_c2r_grad, ops::FFTC2RGradOp);
REGISTER_OP_CPU_KERNEL(
    fft_c2r_grad,
    ops::FFTC2RGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FFTC2RGradKernel<paddle::platform::CPUDeviceContext, double>);
