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
#include "paddle/fluid/framework/data_type_transform.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/eigen/eigen_function.h"
#include "paddle/fluid/operators/math/padding.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/fluid/platform/for_range.h"
#if defined(__NVCC__) || defined(__HIPCC__)
#include "thrust/device_vector.h"
#endif

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

// template <typename DeviceContext, typename T>
// void fill_tensor(const Tensor* in, int64_t axes, int64_t begin, int64_t end,
// T value) {
//   framework::DDim shape = in->dims();
//   eigen_in = framework::EigenVector<T>::Flatten(*in);
//   eigin_in.setZero();

//   framework::Tensor* out;
//   paddle::operators::Slice<DeviceContext, T, D>(ctx, in, out, begin, end,
//   axes);

//   auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
//   EigenConstant<std::decay_t<decltype(place)>, T,
//   out->dims().size()>::Eval(place, out, value);
// }

template <typename T>
struct FFTFillConjGradFunctor {
  T* input_;
  const size_t axis_;
  const int64_t* strides_;
  const size_t double_length_;

  FFTFillConjGradFunctor(T* input, size_t axis, const int64_t* strides,
                         size_t double_length)
      : input_(input),
        axis_(axis),
        strides_(strides),
        double_length_(double_length) {}

  HOSTDEVICE void operator()(size_t index) {
    size_t offtset = index;  // back
    size_t index_i;
    for (size_t i = 0; i <= axis_; i++) {
      index_i = offtset / strides_[i];
      offtset %= strides_[i];
    }

    if ((0 < index_i) && (index_i < double_length_ + 1)) {
      input_[index] *= static_cast<T>(2);
    }
  }
};

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

    const auto axes = ctx.Attr<std::vector<int64_t>>("axes");
    const std::string& norm_str = ctx.Attr<std::string>("normalization");
    const bool forward = ctx.Attr<bool>("forward");
    const bool onesided = ctx.Attr<bool>("onesided");

    const auto* dy = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    dx->mutable_data<T>(ctx.GetPlace());
    framework::Tensor complex_dx;
    complex_dx.mutable_data<C>(dx->dims(), ctx.GetPlace());

    auto normalization = get_norm_from_string(norm_str, forward);
    VLOG(5) << "[FFT][R2C][GRAD]"
            << "Exec FFTR2CGradKernel(onesided=" << onesided << ")";
    FFTC2CFunctor<DeviceContext, C, C> fft_c2c_func;

    if (!onesided) {
      fft_c2c_func(dev_ctx, dy, dx, axes, normalization, !forward);
    } else {
      framework::Tensor full_dy;
      full_dy.mutable_data<C>(dx->dims(), ctx.GetPlace());

      auto last_dim = axes.back();
      VLOG(5) << "last dim";
      auto start = dy->dims().at(last_dim);
      auto end = full_dy.dims().at(last_dim);
      auto zero_length = static_cast<int>(end - start);
      auto rank = dy->dims().size();

      std::vector<int> pads(rank * 2, 0);
      pads[last_dim * 2 + 1] = zero_length;

      paddle::operators::math::PaddingFunctor<DeviceContext, C>(
          rank, ctx, pads, static_cast<C>(0), *dy, &full_dy);
      fft_c2c_func(dev_ctx, &full_dy, &complex_dx, axes, normalization,
                   !forward);
    }
    framework::TransComplexToReal(dx->type(), complex_dx.type(), complex_dx,
                                  dx);
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

    C* pdx = dx->mutable_data<C>(ctx.GetPlace());
    auto normalization = get_norm_from_string(norm_str, forward);

    FFTR2CFunctor<DeviceContext, T, C> fft_r2c_func;
    fft_r2c_func(dev_ctx, dy, dx, axes, normalization, !forward, onesided);

    const int64_t double_length =
        dy->dims()[axes.back()] - dx->dims()[axes.back()];
    const framework::DDim strides = framework::stride(dx->dims());

#if defined(__NVCC__) || defined(__HIPCC__)
    const thrust::device_vector<int64_t> strides_g(
        framework::vectorize(strides));
    const int64_t* pstrides = thrust::raw_pointer_cast(strides_g.data());
#else
    const int64_t* pstrides = strides.Get();
#endif

    FFTFillConjGradFunctor<C> func(pdx, axes.back(), pstrides, double_length);
    size_t limit = dx->numel();
    platform::ForRange<DeviceContext> for_range(dev_ctx, limit);
    for_range(func);
  }
};

}  // namespace operators
}  // namespace paddle
