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
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

enum class FFTNormMode : int64_t {
  none,       // No normalization
  by_sqrt_n,  // Divide by sqrt(signal_size)
  by_n,       // Divide by signal_size
};

FFTNormMode get_norm_from_string(const std::string& norm, bool forward);

template <typename DeviceContext, typename T>
struct FFTC2CFunctor {
  void operator()(const DeviceContext& ctx, const Tensor* X, Tensor* out,
                  const std::vector<int64_t>& axes, FFTNormMode normalization,
                  bool forward);
};

template <typename DeviceContext, typename T>
struct FFTR2CFunctor {
  void operator()(const DeviceContext& ctx, const Tensor* X, Tensor* out,
                  const std::vector<int64_t>& axes, FFTNormMode normalization,
                  bool forward, bool onesided);
};

template <typename DeviceContext, typename T>
struct FFTC2RFunctor {
  void operator()(const DeviceContext& ctx, const Tensor* X, Tensor* out,
                  const std::vector<int64_t>& axes, FFTNormMode normalization,
                  bool forward, bool onesided);
};

template <typename DeviceContext, typename T>
class FFTC2CKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using U = paddle::platform::complex<T>;
    auto& dev_ctx = ctx.device_context<DeviceContext>();

    auto axes = ctx.Attr<std::vector<int64_t>>("axes");
    const std::string& norm_str = ctx.Attr<std::string>("normalization");
    const bool forward = ctx.Attr<bool>("forward");
    const auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Output<Tensor>("Out");

    y->mutable_data<U>(ctx.GetPlace());
    auto normalization = get_norm_from_string(norm_str, forward);

    FFTC2CFunctor<DeviceContext, U> fft_c2c_func;
    fft_c2c_func(dev_ctx, x, y, axes, normalization, forward);
  }
};

template <typename DeviceContext, typename T>
class FFTC2CGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using U = paddle::platform::complex<T>;
    auto& dev_ctx = ctx.device_context<DeviceContext>();

    auto axes = ctx.Attr<std::vector<int64_t>>("axes");
    const std::string& norm_str = ctx.Attr<std::string>("normalization");
    const bool forward = ctx.Attr<bool>("forward");
    const auto* dy = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));

    dx->mutable_data<U>(ctx.GetPlace());
    auto normalization = get_norm_from_string(norm_str, forward);

    FFTC2CFunctor<DeviceContext, U> fft_c2c_func;
    fft_c2c_func(dev_ctx, dy, dx, axes, normalization, forward);
  }
};

template <typename DeviceContext, typename T>
class FFTR2CKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using U = paddle::platform::complex<T>;
    auto& dev_ctx = ctx.device_context<DeviceContext>();

    auto axes = ctx.Attr<std::vector<int64_t>>("axes");
    const std::string& norm_str = ctx.Attr<std::string>("normalization");
    const bool forward = ctx.Attr<bool>("forward");
    const bool onesided = ctx.Attr<bool>("onesided");
    const auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Output<Tensor>("Out");

    y->mutable_data<U>(ctx.GetPlace());
    auto normalization = get_norm_from_string(norm_str, forward);

    FFTR2CFunctor<DeviceContext, U> fft_r2c_func;
    fft_r2c_func(dev_ctx, x, y, axes, normalization, forward, onesided);
  }
};

template <typename DeviceContext, typename T>
class FFTR2CGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using U = paddle::platform::complex<T>;
    auto& dev_ctx = ctx.device_context<DeviceContext>();

    auto axes = ctx.Attr<std::vector<int64_t>>("axes");
    const std::string& norm_str = ctx.Attr<std::string>("normalization");
    const bool forward = ctx.Attr<bool>("forward");

    const auto* dy = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));

    dx->mutable_data<U>(ctx.GetPlace());
    auto normalization = get_norm_from_string(norm_str, forward);

    FFTC2RFunctor<DeviceContext, U> fft_c2r_func;
    fft_c2r_func(dev_ctx, dy, dx, axes, normalization, forward);
  }
};

template <typename DeviceContext, typename T>
class FFTC2RKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using U = paddle::platform::complex<T>;
    auto& dev_ctx = ctx.device_context<DeviceContext>();

    auto axes = ctx.Attr<std::vector<int64_t>>("axes");
    const std::string& norm_str = ctx.Attr<std::string>("normalization");
    const bool forward = ctx.Attr<bool>("forward");
    const auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Output<Tensor>("Out");

    y->mutable_data<U>(ctx.GetPlace());
    auto normalization = get_norm_from_string(norm_str, forward);

    FFTC2RFunctor<DeviceContext, U> fft_c2r_func;
    fft_c2r_func(dev_ctx, x, y, axes, normalization, forward);
  }
};

template <typename DeviceContext, typename T>
class FFTC2RGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using U = paddle::platform::complex<T>;
    auto& dev_ctx = ctx.device_context<DeviceContext>();

    auto axes = ctx.Attr<std::vector<int64_t>>("axes");
    const std::string& norm_str = ctx.Attr<std::string>("normalization");
    const bool forward = ctx.Attr<bool>("forward");
    const bool onesided = true;
    const auto* dy = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));

    dx->mutable_data<U>(ctx.GetPlace());
    auto normalization = get_norm_from_string(norm_str, forward);

    FFTR2CFunctor<DeviceContext, U> fft_r2c_func;
    fft_r2c_func(dev_ctx, dy, dx, axes, normalization, forward, onesided);
  }
};
}  // namespace operators
}  // namespace paddle
