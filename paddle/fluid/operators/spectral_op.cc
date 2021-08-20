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

#include <algorithm>
#include <string>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/platform/complex.h"

#if defined(PADDLE_WITH_ONEMKL)
// #include "mkl_dfti.h"
// #include "mkl_service.h"
#elif defined(PADDLE_WITH_POCKETFFT)
#include "extern_pocketfft/pocketfft_hdronly.h"
#endif

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

//////////////// C2C
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
      // add doc here
    )DOC");
  }
};

class FFTC2COp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      platform::errors::InvalidArgument(
                          "Input(%s) of FFTC2COp should not be null.", "X"));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      platform::errors::InvalidArgument(
                          "Output(%s) of FFTC2COp should not be null.", "Out"));

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
    PADDLE_ENFORCE_EQ(
        ctx->HasInput(framework::GradVarName("Out")), true,
        platform::errors::InvalidArgument(
            "Input(%s) of FFTC2CGradOp should not be null.", "DOut"));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput(framework::GradVarName("X")), true,
        platform::errors::InvalidArgument(
            "Output(%s) of FFTC2CGradOp should not be null.", "DX"));
    auto x_grad_name = framework::GradVarName("X");
    ctx->SetOutputDim(x_grad_name, ctx->GetInputDim("X"));
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

///////////////// R2C
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
      // add doc here
    )DOC");
  }
};

class FFTR2COp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      platform::errors::InvalidArgument(
                          "Input(%s) of FFTC2ROp should not be null.", "X"));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      platform::errors::InvalidArgument(
                          "Output(%s) of FFTC2ROp should not be null.", "Out"));
    const auto axes = ctx->Attrs().Get<std::vector<int64_t>>("axes");
    const bool onesided = ctx->Attrs().Get<bool>("onesided");
    if (!onesided) {
      ctx->ShareDim("X", /*->*/ "Out");  //
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
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

class FFTR2CGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput(framework::GradVarName("Out")), true,
        platform::errors::InvalidArgument(
            "Input(%s) of FFTR2CGradOp should not be null.", "DOut"));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput(framework::GradVarName("X")), true,
        platform::errors::InvalidArgument(
            "Output(%s) of FFTR2CGradOp should not be null.", "DX"));
    auto x_grad_name = framework::GradVarName("X");
    auto out_grad_name = framework::GradVarName("Out");
    const bool onesided = ctx->Attrs().Get<bool>("onesided");
    const auto axes = ctx->Attrs().Get<std::vector<int64_t>>("axes");
    if (!onesided) {
      ctx->ShareDim(out_grad_name, /*->*/ x_grad_name);  //
    } else {
      const auto out_grad_dim = ctx->GetInputDim(out_grad_name);
      framework::DDim x_grad_dim(out_grad_dim);
      const int64_t last_fft_axis = axes.back();
      const int64_t last_fft_dim_size = x_grad_dim.at(last_fft_axis);
      x_grad_dim.at(last_fft_axis) = (last_fft_dim_size - 1) * 2;
      ctx->SetOutputDim(x_grad_name, x_grad_dim);
    }
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

//////////////// C2R
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
    AddComment(R"DOC(
      // add doc here
    )DOC");
  }
};

class FFTC2ROp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      platform::errors::InvalidArgument(
                          "Input(%s) of FFTC2ROp should not be null.", "X"));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      platform::errors::InvalidArgument(
                          "Output(%s) of FFTC2ROp should not be null.", "Out"));
    const auto axes = ctx->Attrs().Get<std::vector<int64_t>>("axes");

    framework::DDim out_dim(ctx->GetInputDim("X"));
    const int64_t last_fft_axis = axes.back();
    const int64_t last_fft_dim_size = out_dim.at(last_fft_axis);
    out_dim.at(last_fft_axis) = (last_fft_dim_size - 1) * 2;
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
    PADDLE_ENFORCE_EQ(
        ctx->HasInput(framework::GradVarName("Out")), true,
        platform::errors::InvalidArgument(
            "Input(%s) of FFTC2RGradOp should not be null.", "DOut"));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput(framework::GradVarName("X")), true,
        platform::errors::InvalidArgument(
            "Output(%s) of FFTC2RGradOp should not be null.", "DX"));
    auto x_grad_name = framework::GradVarName("X");
    auto out_grad_name = framework::GradVarName("Out");
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

//////////////// common
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
      "Fft norm string must be forward or backward or ortho"));
}

template <typename T>
T compute_factor(int64_t size, FFTNormMode normalization) {
  constexpr auto one = static_cast<T>(1);
  switch (normalization) {
    case FFTNormMode::none:
      return one;
    case FFTNormMode::by_n:
      return one / static_cast<T>(size);
    case FFTNormMode::by_sqrt_n:
      return one / std::sqrt(static_cast<T>(size));
  }
  PADDLE_THROW("Unsupported normalization type");
}

////////////////// Functors
#if defined(PADDLE_WITH_ONEMKL)
template <typename T>
struct FFTC2CFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& ctx, const Tensor* x,
                  Tensor* out, const std::vector<int64_t>& axes,
                  FFTNormMode normalization, bool forward) {}
};

template <typename T>
struct FFTR2CFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& ctx, const Tensor* x,
                  Tensor* out, const std::vector<int64_t>& axes,
                  FFTNormMode normalization, bool forward, bool onesided) {}
};

template <typename T>
struct FFTC2RFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& ctx, const Tensor* x,
                  Tensor* out, const std::vector<int64_t>& axes,
                  FFTNormMode normalization, bool forward) {}
};

#elif defined(PADDLE_WITH_POCKETFFT)
template <typename T>
struct FFTC2CFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& ctx, const Tensor* x,
                  Tensor* out, const std::vector<int64_t>& axes,
                  FFTNormMode normalization, bool forward) {
    using R = typename T::value_type;
    using C = std::complex<R>;

    const auto& input_dim = x->dims();
    const std::vector<size_t> in_sizes =
        framework::vectorize<size_t>(input_dim);
    std::vector<int64_t> in_strides =
        framework::vectorize<int64_t>(framework::stride(input_dim));
    const int64_t data_size = sizeof(C);
    std::transform(in_strides.begin(), in_strides.end(), in_strides.begin(),
                   [](int64_t s) { return s * data_size; });

    const auto* in_data = reinterpret_cast<const C*>(x->data<T>());
    auto* out_data = reinterpret_cast<C*>(out->data<T>());
    // well, we have to use std::vector<size_t> here
    std::vector<size_t> axes_(axes.size());
    std::copy(axes.begin(), axes.end(), axes_.begin());
    // compuet facet
    int64_t signal_numel = 1;
    for (auto i : axes) {
      signal_numel *= in_sizes[i];
    }
    R factor = compute_factor<R>(signal_numel, normalization);
    pocketfft::c2c(in_sizes, in_strides, in_strides, axes_, forward, in_data,
                   out_data, factor);
  }
};

template <typename T>
struct FFTR2CFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& ctx, const Tensor* x,
                  Tensor* out, const std::vector<int64_t>& axes,
                  FFTNormMode normalization, bool forward, bool onesided) {
    using R = typename T::value_type;
    using C = std::complex<R>;

    const auto& input_dim = x->dims();
    const std::vector<size_t> in_sizes =
        framework::vectorize<size_t>(input_dim);
    std::vector<int64_t> in_strides =
        framework::vectorize<int64_t>(framework::stride(input_dim));
    {
      const int64_t data_size = sizeof(R);
      std::transform(in_strides.begin(), in_strides.end(), in_strides.begin(),
                     [](int64_t s) { return s * data_size; });
    }

    const auto& output_dim = out->dims();
    const std::vector<size_t> out_sizes =
        framework::vectorize<size_t>(output_dim);
    std::vector<int64_t> out_strides =
        framework::vectorize<int64_t>(framework::stride(output_dim));
    {
      const int64_t data_size = sizeof(C);
      std::transform(out_strides.begin(), out_strides.end(),
                     out_strides.begin(),
                     [](int64_t s) { return s * data_size; });
    }

    const auto* in_data = x->data<R>();
    auto* out_data = reinterpret_cast<C*>(out->data<T>());
    // well, we have to use std::vector<size_t> here
    std::vector<size_t> axes_(axes.size());
    std::copy(axes.begin(), axes.end(), axes_.begin());
    // compuet facet
    int64_t signal_numel = 1;
    for (auto i : axes) {
      signal_numel *= in_sizes[i];
    }
    R factor = compute_factor<R>(signal_numel, normalization);
    pocketfft::r2c(in_sizes, in_strides, out_strides, axes_, forward, in_data,
                   out_data, factor);
  }
};

template <typename T>
struct FFTC2RFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& ctx, const Tensor* x,
                  Tensor* out, const std::vector<int64_t>& axes,
                  FFTNormMode normalization, bool forward) {
    using R = typename T::value_type;
    using C = std::complex<R>;

    const auto& input_dim = x->dims();
    const std::vector<size_t> in_sizes =
        framework::vectorize<size_t>(input_dim);
    std::vector<int64_t> in_strides =
        framework::vectorize<int64_t>(framework::stride(input_dim));
    {
      const int64_t data_size = sizeof(C);
      std::transform(in_strides.begin(), in_strides.end(), in_strides.begin(),
                     [](int64_t s) { return s * data_size; });
    }

    const auto& output_dim = out->dims();
    const std::vector<size_t> out_sizes =
        framework::vectorize<size_t>(output_dim);
    std::vector<int64_t> out_strides =
        framework::vectorize<int64_t>(framework::stride(output_dim));
    {
      const int64_t data_size = sizeof(R);
      std::transform(out_strides.begin(), out_strides.end(),
                     out_strides.begin(),
                     [](int64_t s) { return s * data_size; });
    }

    const auto* in_data = reinterpret_cast<const C*>(x->data<T>());
    auto* out_data = out->data<R>();
    // well, we have to use std::vector<size_t> here
    std::vector<size_t> axes_(axes.size());
    std::copy(axes.begin(), axes.end(), axes_.begin());
    // compuet facet
    int64_t signal_numel = 1;
    for (auto i : axes) {
      signal_numel *= out_sizes[i];
    }
    R factor = compute_factor<R>(signal_numel, normalization);
    pocketfft::c2r(out_sizes, in_strides, out_strides, axes_, forward, in_data,
                   out_data, factor);
  }
};

#endif
// mkl fft for all cases
void exec_fft(const Tensor* x, Tensor* out, const std::vector<int64_t>& out_dim,
              int64_t normalization, bool forward) {
  // construct the descriptor

  // compute
}  // namespace anonymous

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
