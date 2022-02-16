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
#define NOMINMAX  // to use std::min std::max correctly on windows
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/data_type_transform.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/conj_op.h"
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
                  bool forward);
};

template <typename DeviceContext, typename Ti, typename To>
struct FFTC2RFunctor {
  void operator()(const DeviceContext& ctx, const Tensor* X, Tensor* out,
                  const std::vector<int64_t>& axes, FFTNormMode normalization,
                  bool forward);
};

// Giving a linear destination index and strides of tensor, get_idx return the
// corresponding linear position of source tensor.
// The linear index is the position of flatten tensor.
// Giving a linear destination index and strides of tensor, get_idx return the
// corresponding linear position of source tensor.
// The linear index is the position of flatten tensor.
HOSTDEVICE inline int64_t get_src_idx(const int64_t dst_idx,
                                      const int64_t* dst_strides,
                                      const int64_t* dst_shape,
                                      const int64_t* src_strides,
                                      const bool* is_fft_axis, const bool conj,
                                      const int64_t rank) {
  int64_t src_idx = 0;
  int64_t quotient = dst_idx;
  int64_t remainder = 0;

  for (int64_t i = 0; i < rank; i++) {
    remainder = quotient % dst_strides[i];
    quotient = quotient / dst_strides[i];
    if (conj && is_fft_axis[i]) {
      src_idx += ((dst_shape[i] - quotient) % dst_shape[i]) * src_strides[i];
    } else {
      src_idx += src_strides[i] * quotient;
    }
    quotient = remainder;
  }

  return src_idx;
}

HOSTDEVICE inline bool is_conj_part(const int64_t dst_idx,
                                    const int64_t* dst_strides,
                                    const int64_t last_axis,
                                    const int64_t last_axis_size) {
  int64_t quotient = dst_idx;
  int64_t remainder = 0;

  for (int64_t i = 0; i < last_axis + 1; i++) {
    remainder = quotient % dst_strides[i];
    quotient = quotient / dst_strides[i];

    if ((i == last_axis) && (quotient > last_axis_size - 1)) {
      return true;
    }

    quotient = remainder;
  }

  return false;
}

// FFTFillConjFunctor fill the destination tensor with source tensor and
// conjugate symmetry element of source tensor .
// Use framework::ForRange to iterate destination element with
// supporting different device
template <typename C>
struct FFTFillConjFunctor {
  FFTFillConjFunctor(const C* src_data, C* dst_data, const int64_t* src_strides,
                     const int64_t* dst_strides, const int64_t* dst_shape,
                     const bool* is_fft_axis, const int64_t last_axis,
                     const int64_t last_axis_size, const int64_t rank)
      : src_data_(src_data),
        dst_data_(dst_data),
        src_strides_(src_strides),
        dst_strides_(dst_strides),
        dst_shape_(dst_shape),
        is_fft_axis_(is_fft_axis),
        last_axis_(last_axis),
        last_axis_size_(last_axis_size),
        rank_(rank) {}
  HOSTDEVICE void operator()(int64_t dst_idx) {
    if (is_conj_part(dst_idx, dst_strides_, last_axis_, last_axis_size_)) {
      const auto conj_idx =
          get_src_idx(dst_idx, dst_strides_, dst_shape_, src_strides_,
                      is_fft_axis_, true, rank_);
      auto src_value = src_data_[conj_idx];
      auto conj_value = C(src_value.real, -src_value.imag);
      dst_data_[dst_idx] = conj_value;
    } else {
      const auto copy_idx =
          get_src_idx(dst_idx, dst_strides_, dst_shape_, src_strides_,
                      is_fft_axis_, false, rank_);
      dst_data_[dst_idx] = src_data_[copy_idx];
    }
  }

  const C* src_data_;
  C* dst_data_;
  const int64_t* src_strides_;
  const int64_t* dst_strides_;
  const int64_t* dst_shape_;
  const bool* is_fft_axis_;
  const int64_t last_axis_;
  const int64_t last_axis_size_;
  const int64_t rank_;
};

template <typename DeviceContext, typename C>
void fill_conj(const DeviceContext& ctx, const Tensor* src, Tensor* dst,
               const std::vector<int64_t>& axes) {
  std::vector<int64_t> src_strides_v =
      framework::vectorize<int64_t>(framework::stride(src->dims()));
  std::vector<int64_t> dst_strides_v =
      framework::vectorize<int64_t>(framework::stride(dst->dims()));
  std::vector<int64_t> dst_shape_v = framework::vectorize<int64_t>(dst->dims());
  const auto src_data = src->data<C>();
  auto dst_data = dst->data<C>();
  const auto last_axis = axes.back();
  const auto last_axis_size = dst->dims().at(last_axis) / 2 + 1;
  const int64_t rank = dst->dims().size();
  auto _is_fft_axis = std::make_unique<bool[]>(rank);
  for (const auto i : axes) {
    _is_fft_axis[i] = true;
  }

#if defined(__NVCC__) || defined(__HIPCC__)
  const thrust::device_vector<int64_t> src_strides_g(src_strides_v);
  const auto src_strides = thrust::raw_pointer_cast(src_strides_g.data());
  const thrust::device_vector<int64_t> dst_strides_g(dst_strides_v);
  const auto dst_strides = thrust::raw_pointer_cast(dst_strides_g.data());
  const thrust::device_vector<int64_t> dst_shape_g(dst_shape_v);
  const auto dst_shape = thrust::raw_pointer_cast(dst_shape_g.data());
  const thrust::device_vector<bool> is_fft_axis_g(_is_fft_axis.get(),
                                                  _is_fft_axis.get() + rank);
  const auto p_is_fft_axis = thrust::raw_pointer_cast(is_fft_axis_g.data());
#else
  const auto src_strides = src_strides_v.data();
  const auto dst_strides = dst_strides_v.data();
  const auto dst_shape = dst_shape_v.data();
  const auto p_is_fft_axis = _is_fft_axis.get();
#endif
  platform::ForRange<DeviceContext> for_range(ctx, dst->numel());
  FFTFillConjFunctor<C> fill_conj_functor(src_data, dst_data, src_strides,
                                          dst_strides, dst_shape, p_is_fft_axis,
                                          last_axis, last_axis_size, rank);
  for_range(fill_conj_functor);
}

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

    if (onesided) {
      fft_r2c_func(dev_ctx, x, y, axes, normalization, forward);
    } else {
      framework::DDim onesided_dims(y->dims());
      const int64_t onesided_last_axis_size = y->dims().at(axes.back()) / 2 + 1;
      onesided_dims.at(axes.back()) = onesided_last_axis_size;
      framework::Tensor onesided_out;
      onesided_out.mutable_data<C>(onesided_dims, ctx.GetPlace());
      fft_r2c_func(dev_ctx, x, &onesided_out, axes, normalization, forward);
      fill_conj<DeviceContext, C>(dev_ctx, &onesided_out, y, axes);
    }
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
    FFTC2CFunctor<DeviceContext, C, C> fft_c2c_func;

    if (!onesided) {
      fft_c2c_func(dev_ctx, dy, &complex_dx, axes, normalization, !forward);
    } else {
      framework::Tensor full_dy;
      full_dy.mutable_data<C>(dx->dims(), ctx.GetPlace());
      auto zero_length = static_cast<int>(full_dy.dims().at(axes.back()) -
                                          dy->dims().at(axes.back()));
      auto rank = dy->dims().size();

      std::vector<int> pads(rank * 2, 0);
      pads[axes.back() * 2 + 1] = zero_length;

      paddle::operators::math::PaddingFunctor<DeviceContext, C>(
          rank, ctx, pads, static_cast<C>(0), *dy, &full_dy);
      fft_c2c_func(dev_ctx, &full_dy, &complex_dx, axes, normalization,
                   !forward);
    }
    framework::TransComplexToReal(
        framework::TransToProtoVarType(dx->dtype()),
        framework::TransToProtoVarType(complex_dx.dtype()), complex_dx, dx);
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
    const auto* dy = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));

    C* pdx = dx->mutable_data<C>(ctx.GetPlace());
    auto normalization = get_norm_from_string(norm_str, forward);

    FFTR2CFunctor<DeviceContext, T, C> fft_r2c_func;
    fft_r2c_func(dev_ctx, dy, dx, axes, normalization, !forward);

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
