/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/eigen/eigen_function.h"
#include "paddle/fluid/platform/complex.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

enum class FFTNormMode : int64_t {
  none,       // No normalization
  by_sqrt_n,  // Divide by sqrt(signal_size)
  by_n,       // Divide by signal_size
};

FFTNormMode get_norm_from_string(const std::string& norm, bool forward);

// Enum representing the FFT type
enum class FFTTransformType : int64_t {
  C2C = 0,  // Complex-to-complex
  R2C,      // Real-to-complex
  C2R,      // Complex-to-real
};

// Create transform type enum from bools representing if input and output are
// complex
inline FFTTransformType GetFFTTransformType(
    framework::proto::VarType::Type input_dtype,
    framework::proto::VarType::Type output_dtype) {
  auto complex_input = framework::IsComplexType(input_dtype);
  auto complex_output = framework::IsComplexType(output_dtype);
  if (complex_input && complex_output) {
    return FFTTransformType::C2C;
  } else if (complex_input && !complex_output) {
    return FFTTransformType::C2R;
  } else if (!complex_input && complex_output) {
    return FFTTransformType::R2C;
  }
  PADDLE_THROW(
      platform::errors::InvalidArgument("Real to real FFTs are not supported"));
}

// Returns true if the transform type has complex input
inline bool has_complex_input(FFTTransformType type) {
  switch (type) {
    case FFTTransformType::C2C:
    case FFTTransformType::C2R:
      return true;

    case FFTTransformType::R2C:
      return false;
  }
  PADDLE_THROW(platform::errors::InvalidArgument("Unknown FFTTransformType"));
}

// Returns true if the transform type has complex output
inline bool has_complex_output(FFTTransformType type) {
  switch (type) {
    case FFTTransformType::C2C:
    case FFTTransformType::R2C:
      return true;

    case FFTTransformType::C2R:
      return false;
  }
  PADDLE_THROW(platform::errors::InvalidArgument("Unknown FFTTransformType"));
}

template <typename DeviceContext, typename Ti, typename To>
struct FFTC2CFunctor {
  void operator()(const DeviceContext& ctx, const Tensor* X, Tensor* out,
                  const std::vector<int64_t>& axes, FFTNormMode normalization,
                  bool forward);
};

template <typename DeviceContext, typename Ti, typename To>
struct FFTR2CFunctor {
  void operator()(const DeviceContext& ctx, const Tensor* X, Tensor* out,
                  const std::vector<int64_t>& axes, FFTNormMode normalization,
                  bool forward, bool onesided);
};

template <typename DeviceContext, typename Ti, typename To>
struct FFTC2RFunctor {
  void operator()(const DeviceContext& ctx, const Tensor* X, Tensor* out,
                  const std::vector<int64_t>& axes, FFTNormMode normalization,
                  bool forward);
};

template <typename DeviceContext, typename T>
class FFTC2CKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using C = paddle::platform::complex<T>;
    auto& dev_ctx = ctx.device_context<DeviceContext>();

    auto axes = ctx.Attr<std::vector<int64_t>>("axes");
    const std::string& norm_str = ctx.Attr<std::string>("normalization");
    const bool forward = ctx.Attr<bool>("forward");
    const auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Output<Tensor>("Out");

    y->mutable_data<C>(ctx.GetPlace());
    auto normalization = get_norm_from_string(norm_str, forward);

    FFTC2CFunctor<DeviceContext, C, C> fft_c2c_func;
    fft_c2c_func(dev_ctx, x, y, axes, normalization, forward);
  }
};

template <typename DeviceContext, typename T>
class FFTC2CGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using C = paddle::platform::complex<T>;
    auto& dev_ctx = ctx.device_context<DeviceContext>();

    auto axes = ctx.Attr<std::vector<int64_t>>("axes");
    const std::string& norm_str = ctx.Attr<std::string>("normalization");
    const bool forward = ctx.Attr<bool>("forward");
    const auto* dy = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));

    dx->mutable_data<C>(ctx.GetPlace());
    auto normalization = get_norm_from_string(norm_str, forward);

    FFTC2CFunctor<DeviceContext, C, C> fft_c2c_func;
    fft_c2c_func(dev_ctx, dy, dx, axes, normalization, !forward);
  }
};

template <typename DeviceContext, typename T>
class FFTR2CKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using C = paddle::platform::complex<T>;
    auto& dev_ctx = ctx.device_context<DeviceContext>();

    auto axes = ctx.Attr<std::vector<int64_t>>("axes");
    const std::string& norm_str = ctx.Attr<std::string>("normalization");
    const bool forward = ctx.Attr<bool>("forward");
    const bool onesided = ctx.Attr<bool>("onesided");
    const auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Output<Tensor>("Out");

    y->mutable_data<C>(ctx.GetPlace());
    auto normalization = get_norm_from_string(norm_str, forward);

    FFTR2CFunctor<DeviceContext, T, C> fft_r2c_func;
    fft_r2c_func(dev_ctx, x, y, axes, normalization, forward, onesided);
  }
};

template <typename DeviceContext, typename T>
class FFTR2CGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using C = paddle::platform::complex<T>;
    auto& dev_ctx = ctx.device_context<DeviceContext>();

    auto axes = ctx.Attr<std::vector<int64_t>>("axes");
    const std::string& norm_str = ctx.Attr<std::string>("normalization");
    const bool forward = ctx.Attr<bool>("forward");

    const auto* dy = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));

    dx->mutable_data<C>(ctx.GetPlace());
    auto normalization = get_norm_from_string(norm_str, forward);

    FFTC2RFunctor<DeviceContext, C, T> fft_c2r_func;
    fft_c2r_func(dev_ctx, dy, dx, axes, normalization, forward);
  }
};

template <typename DeviceContext, typename T>
class FFTC2RKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using C = paddle::platform::complex<T>;
    auto& dev_ctx = ctx.device_context<DeviceContext>();

    auto axes = ctx.Attr<std::vector<int64_t>>("axes");
    const std::string& norm_str = ctx.Attr<std::string>("normalization");
    const bool forward = ctx.Attr<bool>("forward");
    const auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Output<Tensor>("Out");

    y->mutable_data<T>(ctx.GetPlace());
    auto normalization = get_norm_from_string(norm_str, forward);

    FFTC2RFunctor<DeviceContext, C, T> fft_c2r_func;
    fft_c2r_func(dev_ctx, x, y, axes, normalization, forward);
  }
};

template <typename DeviceContext, typename T>
class FFTC2RGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using C = paddle::platform::complex<T>;
    auto& dev_ctx = ctx.device_context<DeviceContext>();

    auto axes = ctx.Attr<std::vector<int64_t>>("axes");
    const std::string& norm_str = ctx.Attr<std::string>("normalization");
    const bool forward = ctx.Attr<bool>("forward");
    const bool onesided = true;
    const auto* dy = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));

    dx->mutable_data<C>(ctx.GetPlace());
    auto normalization = get_norm_from_string(norm_str, forward);

    FFTR2CFunctor<DeviceContext, T, C> fft_r2c_func;
    fft_r2c_func(dev_ctx, dy, dx, axes, normalization, forward, onesided);
  }
};
}  // namespace operators
}  // namespace paddle
