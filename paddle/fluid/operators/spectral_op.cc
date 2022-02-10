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
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/operators/transpose_op.h"
#include "paddle/fluid/platform/complex.h"

#if defined(PADDLE_WITH_ONEMKL)
#include "paddle/fluid/platform/dynload/mklrt.h"
#elif defined(PADDLE_WITH_POCKETFFT)
#include "extern_pocketfft/pocketfft_hdronly.h"
#endif

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/for_range.h"

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

// FFT Functors
#if defined(PADDLE_WITH_ONEMKL)

#define MKL_DFTI_CHECK(expr)                                       \
  do {                                                             \
    MKL_LONG status = (expr);                                      \
    if (!platform::dynload::DftiErrorClass(status, DFTI_NO_ERROR)) \
      PADDLE_THROW(platform::errors::External(                     \
          platform::dynload::DftiErrorMessage(status)));           \
  } while (0);

namespace {

struct DftiDescriptorDeleter {
  void operator()(DFTI_DESCRIPTOR_HANDLE handle) {
    if (handle != nullptr) {
      MKL_DFTI_CHECK(platform::dynload::DftiFreeDescriptor(&handle));
    }
  }
};

// A RAII wrapper for MKL_DESCRIPTOR*
class DftiDescriptor {
 public:
  void init(DFTI_CONFIG_VALUE precision, DFTI_CONFIG_VALUE signal_type,
            MKL_LONG signal_ndim, MKL_LONG* sizes) {
    PADDLE_ENFORCE_EQ(desc_.get(), nullptr,
                      platform::errors::AlreadyExists(
                          "DftiDescriptor has already been initialized."));

    DFTI_DESCRIPTOR* raw_desc;
    MKL_DFTI_CHECK(platform::dynload::DftiCreateDescriptorX(
        &raw_desc, precision, signal_type, signal_ndim, sizes));
    desc_.reset(raw_desc);
  }

  DFTI_DESCRIPTOR* get() const {
    DFTI_DESCRIPTOR* raw_desc = desc_.get();
    PADDLE_ENFORCE_NOT_NULL(raw_desc,
                            platform::errors::PreconditionNotMet(
                                "DFTI DESCRIPTOR has not been initialized."));
    return raw_desc;
  }

 private:
  std::unique_ptr<DFTI_DESCRIPTOR, DftiDescriptorDeleter> desc_;
};

DftiDescriptor _plan_mkl_fft(const framework::proto::VarType::Type& in_dtype,
                             const framework::proto::VarType::Type& out_dtype,
                             const framework::DDim& in_strides,
                             const framework::DDim& out_strides,
                             const std::vector<int>& signal_sizes,
                             FFTNormMode normalization, bool forward) {
  const DFTI_CONFIG_VALUE precision = [&] {
    switch (in_dtype) {
      case framework::proto::VarType::FP32:
        return DFTI_SINGLE;
      case framework::proto::VarType::COMPLEX64:
        return DFTI_SINGLE;
      case framework::proto::VarType::FP64:
        return DFTI_DOUBLE;
      case framework::proto::VarType::COMPLEX128:
        return DFTI_DOUBLE;
      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Invalid input datatype (%s), input data type should be FP32, "
            "FP64, COMPLEX64 or COMPLEX128.",
            framework::DataTypeToString(in_dtype)));
    }
  }();

  // C2C, R2C, C2R
  const FFTTransformType fft_type = GetFFTTransformType(in_dtype, out_dtype);
  const DFTI_CONFIG_VALUE domain =
      (fft_type == FFTTransformType::C2C) ? DFTI_COMPLEX : DFTI_REAL;

  DftiDescriptor descriptor;
  std::vector<MKL_LONG> fft_sizes(signal_sizes.cbegin(), signal_sizes.cend());
  const MKL_LONG signal_ndim = fft_sizes.size() - 1;
  descriptor.init(precision, domain, signal_ndim, fft_sizes.data() + 1);

  // placement inplace or not inplace
  MKL_DFTI_CHECK(platform::dynload::DftiSetValue(
      descriptor.get(), DFTI_PLACEMENT, DFTI_NOT_INPLACE));

  // number of transformations
  const MKL_LONG batch_size = fft_sizes[0];
  MKL_DFTI_CHECK(platform::dynload::DftiSetValue(
      descriptor.get(), DFTI_NUMBER_OF_TRANSFORMS, batch_size));

  // input & output distance
  const MKL_LONG idist = in_strides[0];
  const MKL_LONG odist = out_strides[0];
  MKL_DFTI_CHECK(platform::dynload::DftiSetValue(descriptor.get(),
                                                 DFTI_INPUT_DISTANCE, idist));
  MKL_DFTI_CHECK(platform::dynload::DftiSetValue(descriptor.get(),
                                                 DFTI_OUTPUT_DISTANCE, odist));

  // input & output stride
  std::vector<MKL_LONG> mkl_in_stride(1 + signal_ndim, 0);
  std::vector<MKL_LONG> mkl_out_stride(1 + signal_ndim, 0);
  for (MKL_LONG i = 1; i <= signal_ndim; i++) {
    mkl_in_stride[i] = in_strides[i];
    mkl_out_stride[i] = out_strides[i];
  }
  MKL_DFTI_CHECK(platform::dynload::DftiSetValue(
      descriptor.get(), DFTI_INPUT_STRIDES, mkl_in_stride.data()));
  MKL_DFTI_CHECK(platform::dynload::DftiSetValue(
      descriptor.get(), DFTI_OUTPUT_STRIDES, mkl_out_stride.data()));

  // conjugate even storage
  if (!(fft_type == FFTTransformType::C2C)) {
    MKL_DFTI_CHECK(platform::dynload::DftiSetValue(
        descriptor.get(), DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX));
  }

  MKL_LONG signal_numel =
      std::accumulate(fft_sizes.cbegin() + 1, fft_sizes.cend(), 1UL,
                      std::multiplies<MKL_LONG>());
  if (normalization != FFTNormMode::none) {
    const double scale =
        ((normalization == FFTNormMode::by_sqrt_n)
             ? 1.0 / std::sqrt(static_cast<double>(signal_numel))
             : 1.0 / static_cast<double>(signal_numel));
    const auto scale_direction = [&]() {
      if (fft_type == FFTTransformType::R2C ||
          (fft_type == FFTTransformType::C2C && forward)) {
        return DFTI_FORWARD_SCALE;
      } else {
        // (fft_type == FFTTransformType::C2R ||
        //          (fft_type == FFTTransformType::C2C && !forward))
        return DFTI_BACKWARD_SCALE;
      }
    }();
    MKL_DFTI_CHECK(platform::dynload::DftiSetValue(descriptor.get(),
                                                   scale_direction, scale));
  }

  // commit the descriptor
  MKL_DFTI_CHECK(platform::dynload::DftiCommitDescriptor(descriptor.get()));
  return descriptor;
}

// Execute a general fft operation (can be c2c, onesided r2c or onesided c2r)
template <typename DeviceContext, typename Ti, typename To>
void exec_fft(const DeviceContext& ctx, const Tensor* x, Tensor* out,
              const std::vector<int64_t>& axes, FFTNormMode normalization,
              bool forward) {
  const framework::DDim& in_sizes = x->dims();
  const int ndim = in_sizes.size();
  const int signal_ndim = axes.size();
  const int batch_ndim = ndim - signal_ndim;
  const framework::DDim& out_sizes = out->dims();

  // make a dim permutation
  std::vector<int> dim_permute(ndim);
  std::iota(dim_permute.begin(), dim_permute.end(), 0);
  std::vector<bool> is_transformed_dim(ndim, false);
  for (const auto& d : axes) {
    is_transformed_dim[d] = true;
  }
  const auto batch_end =
      std::partition(dim_permute.begin(), dim_permute.end(),
                     [&](size_t axis) { return !is_transformed_dim[axis]; });
  std::copy(axes.cbegin(), axes.cend(), batch_end);

  // transpose input according to that permutation
  framework::DDim transposed_input_shape = in_sizes.transpose(dim_permute);
  std::vector<int64_t> transposed_input_shape_ =
      framework::vectorize(transposed_input_shape);
  framework::Tensor transposed_input;
  transposed_input.Resize(transposed_input_shape);
  const auto place = ctx.GetPlace();
  transposed_input.mutable_data<Ti>(place);
  TransCompute<platform::CPUDeviceContext, Ti>(ndim, ctx, *x, &transposed_input,
                                               dim_permute);

  // make an collapsed input: collapse batch axes for input
  const int batch_size = std::accumulate(
      transposed_input_shape.Get(), transposed_input_shape.Get() + batch_ndim,
      1L, std::multiplies<int64_t>());
  std::vector<int> collapsed_input_shape_(1 + signal_ndim);
  collapsed_input_shape_[0] = batch_size;
  std::copy(transposed_input_shape_.begin() + batch_ndim,
            transposed_input_shape_.end(), collapsed_input_shape_.begin() + 1);
  const framework::DDim collapsed_input_shape =
      framework::make_ddim(collapsed_input_shape_);
  transposed_input.Resize(collapsed_input_shape);
  framework::Tensor& collapsed_input = transposed_input;

  // make a collapsed output
  std::vector<int> collapsed_output_shape_(1 + signal_ndim);
  collapsed_output_shape_[0] = batch_size;
  for (int i = 0; i < signal_ndim; i++) {
    collapsed_output_shape_[1 + i] = out_sizes[axes[i]];
  }
  const framework::DDim collapsed_output_shape =
      framework::make_ddim(collapsed_output_shape_);
  framework::Tensor collapsed_output;
  collapsed_output.Resize(collapsed_output_shape);
  collapsed_output.mutable_data(place, out->type());

  // signal sizes
  std::vector<int> signal_sizes(1 + signal_ndim);
  signal_sizes[0] = batch_size;
  for (int i = 0; i < signal_ndim; i++) {
    signal_sizes[1 + i] =
        std::max(collapsed_input_shape[1 + i], collapsed_output_shape[1 + i]);
  }

  // input & output stride
  const framework::DDim input_stride = framework::stride(collapsed_input_shape);
  const framework::DDim output_stride =
      framework::stride(collapsed_output_shape);

  // make a DFTI_DESCRIPTOR
  DftiDescriptor desc =
      _plan_mkl_fft(x->type(), out->type(), input_stride, output_stride,
                    signal_sizes, normalization, forward);

  const FFTTransformType fft_type = GetFFTTransformType(x->type(), out->type());
  if (fft_type == FFTTransformType::C2R && forward) {
    framework::Tensor collapsed_input_conj(collapsed_input.type());
    collapsed_input_conj.mutable_data<Ti>(collapsed_input.dims(),
                                          ctx.GetPlace());
    // conjugate the input
    platform::ForRange<DeviceContext> for_range(ctx, collapsed_input.numel());
    math::ConjFunctor<Ti> functor(collapsed_input.data<Ti>(),
                                  collapsed_input.numel(),
                                  collapsed_input_conj.data<Ti>());
    for_range(functor);
    MKL_DFTI_CHECK(platform::dynload::DftiComputeBackward(
        desc.get(), collapsed_input_conj.data(), collapsed_output.data()));
  } else if (fft_type == FFTTransformType::R2C && !forward) {
    framework::Tensor collapsed_output_conj(collapsed_output.type());
    collapsed_output_conj.mutable_data<To>(collapsed_output.dims(),
                                           ctx.GetPlace());
    MKL_DFTI_CHECK(platform::dynload::DftiComputeForward(
        desc.get(), collapsed_input.data(), collapsed_output_conj.data()));
    // conjugate the output
    platform::ForRange<DeviceContext> for_range(ctx, collapsed_output.numel());
    math::ConjFunctor<To> functor(collapsed_output_conj.data<To>(),
                                  collapsed_output.numel(),
                                  collapsed_output.data<To>());
    for_range(functor);
  } else {
    if (forward) {
      MKL_DFTI_CHECK(platform::dynload::DftiComputeForward(
          desc.get(), collapsed_input.data(), collapsed_output.data()));
    } else {
      MKL_DFTI_CHECK(platform::dynload::DftiComputeBackward(
          desc.get(), collapsed_input.data(), collapsed_output.data()));
    }
  }

  // resize for the collapsed output
  framework::DDim transposed_output_shape = out_sizes.transpose(dim_permute);
  collapsed_output.Resize(transposed_output_shape);
  framework::Tensor& transposed_output = collapsed_output;

  // reverse the transposition
  std::vector<int> reverse_dim_permute(ndim);
  for (int i = 0; i < ndim; i++) {
    reverse_dim_permute[dim_permute[i]] = i;
  }
  TransCompute<platform::CPUDeviceContext, To>(ndim, ctx, transposed_output,
                                               out, reverse_dim_permute);
}
}  // anonymous namespace

template <typename Ti, typename To>
struct FFTC2CFunctor<platform::CPUDeviceContext, Ti, To> {
  void operator()(const platform::CPUDeviceContext& ctx, const Tensor* x,
                  Tensor* out, const std::vector<int64_t>& axes,
                  FFTNormMode normalization, bool forward) {
    exec_fft<platform::CPUDeviceContext, Ti, To>(ctx, x, out, axes,
                                                 normalization, forward);
  }
};

template <typename Ti, typename To>
struct FFTR2CFunctor<platform::CPUDeviceContext, Ti, To> {
  void operator()(const platform::CPUDeviceContext& ctx, const Tensor* x,
                  Tensor* out, const std::vector<int64_t>& axes,
                  FFTNormMode normalization, bool forward) {
    exec_fft<platform::CPUDeviceContext, Ti, To>(ctx, x, out, axes,
                                                 normalization, forward);
  }
};

template <typename Ti, typename To>
struct FFTC2RFunctor<platform::CPUDeviceContext, Ti, To> {
  void operator()(const platform::CPUDeviceContext& ctx, const Tensor* x,
                  Tensor* out, const std::vector<int64_t>& axes,
                  FFTNormMode normalization, bool forward) {
    if (axes.size() > 1) {
      const std::vector<int64_t> c2c_dims(axes.begin(), axes.end() - 1);
      Tensor temp;
      temp.mutable_data<Ti>(x->dims(), ctx.GetPlace());

      FFTC2CFunctor<platform::CPUDeviceContext, Ti, Ti> c2c_functor;
      c2c_functor(ctx, x, &temp, c2c_dims, normalization, forward);

      const std::vector<int64_t> new_axes{axes.back()};
      exec_fft<platform::CPUDeviceContext, Ti, To>(ctx, &temp, out, new_axes,
                                                   normalization, forward);
    } else {
      exec_fft<platform::CPUDeviceContext, Ti, To>(ctx, x, out, axes,
                                                   normalization, forward);
    }
  }
};

#elif defined(PADDLE_WITH_POCKETFFT)

namespace {
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
  PADDLE_THROW(
      platform::errors::InvalidArgument("Unsupported normalization type"));
}
}  // anonymous namespace

template <typename Ti, typename To>
struct FFTC2CFunctor<platform::CPUDeviceContext, Ti, To> {
  void operator()(const platform::CPUDeviceContext& ctx, const Tensor* x,
                  Tensor* out, const std::vector<int64_t>& axes,
                  FFTNormMode normalization, bool forward) {
    using R = typename Ti::value_type;
    using C = std::complex<R>;

    const auto& input_dim = x->dims();
    const std::vector<size_t> in_sizes =
        framework::vectorize<size_t>(input_dim);
    std::vector<std::ptrdiff_t> in_strides =
        framework::vectorize<std::ptrdiff_t>(framework::stride(input_dim));
    const int64_t data_size = sizeof(C);
    std::transform(in_strides.begin(), in_strides.end(), in_strides.begin(),
                   [&](std::ptrdiff_t s) { return s * data_size; });

    const auto* in_data = reinterpret_cast<const C*>(x->data<Ti>());
    auto* out_data = reinterpret_cast<C*>(out->data<To>());
    // pocketfft requires std::vector<size_t>
    std::vector<size_t> axes_(axes.size());
    std::copy(axes.begin(), axes.end(), axes_.begin());
    // compuet factor
    int64_t signal_numel = 1;
    for (auto i : axes) {
      signal_numel *= in_sizes[i];
    }
    R factor = compute_factor<R>(signal_numel, normalization);
    pocketfft::c2c(in_sizes, in_strides, in_strides, axes_, forward, in_data,
                   out_data, factor);
  }
};

template <typename Ti, typename To>
struct FFTR2CFunctor<platform::CPUDeviceContext, Ti, To> {
  void operator()(const platform::CPUDeviceContext& ctx, const Tensor* x,
                  Tensor* out, const std::vector<int64_t>& axes,
                  FFTNormMode normalization, bool forward) {
    using R = Ti;
    using C = std::complex<R>;

    const auto& input_dim = x->dims();
    const std::vector<size_t> in_sizes =
        framework::vectorize<size_t>(input_dim);
    std::vector<std::ptrdiff_t> in_strides =
        framework::vectorize<std::ptrdiff_t>(framework::stride(input_dim));
    {
      const int64_t data_size = sizeof(R);
      std::transform(in_strides.begin(), in_strides.end(), in_strides.begin(),
                     [&](std::ptrdiff_t s) { return s * data_size; });
    }

    const auto& output_dim = out->dims();
    const std::vector<size_t> out_sizes =
        framework::vectorize<size_t>(output_dim);
    std::vector<std::ptrdiff_t> out_strides =
        framework::vectorize<std::ptrdiff_t>(framework::stride(output_dim));
    {
      const int64_t data_size = sizeof(C);
      std::transform(out_strides.begin(), out_strides.end(),
                     out_strides.begin(),
                     [&](std::ptrdiff_t s) { return s * data_size; });
    }

    const auto* in_data = x->data<R>();
    auto* out_data = reinterpret_cast<C*>(out->data<To>());
    // pocketfft requires std::vector<size_t>
    std::vector<size_t> axes_(axes.size());
    std::copy(axes.begin(), axes.end(), axes_.begin());
    // compuet normalization factor
    int64_t signal_numel = 1;
    for (auto i : axes) {
      signal_numel *= in_sizes[i];
    }
    R factor = compute_factor<R>(signal_numel, normalization);
    pocketfft::r2c(in_sizes, in_strides, out_strides, axes_, forward, in_data,
                   out_data, factor);
  }
};

template <typename Ti, typename To>
struct FFTC2RFunctor<platform::CPUDeviceContext, Ti, To> {
  void operator()(const platform::CPUDeviceContext& ctx, const Tensor* x,
                  Tensor* out, const std::vector<int64_t>& axes,
                  FFTNormMode normalization, bool forward) {
    using R = To;
    using C = std::complex<R>;

    const auto& input_dim = x->dims();
    const std::vector<size_t> in_sizes =
        framework::vectorize<size_t>(input_dim);
    std::vector<std::ptrdiff_t> in_strides =
        framework::vectorize<std::ptrdiff_t>(framework::stride(input_dim));
    {
      const int64_t data_size = sizeof(C);
      std::transform(in_strides.begin(), in_strides.end(), in_strides.begin(),
                     [&](std::ptrdiff_t s) { return s * data_size; });
    }

    const auto& output_dim = out->dims();
    const std::vector<size_t> out_sizes =
        framework::vectorize<size_t>(output_dim);
    std::vector<std::ptrdiff_t> out_strides =
        framework::vectorize<std::ptrdiff_t>(framework::stride(output_dim));
    {
      const int64_t data_size = sizeof(R);
      std::transform(out_strides.begin(), out_strides.end(),
                     out_strides.begin(),
                     [&](std::ptrdiff_t s) { return s * data_size; });
    }

    const auto* in_data = reinterpret_cast<const C*>(x->data<Ti>());
    auto* out_data = out->data<R>();
    // pocketfft requires std::vector<size_t>
    std::vector<size_t> axes_(axes.size());
    std::copy(axes.begin(), axes.end(), axes_.begin());
    // compuet normalization factor
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
