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
#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
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
#include "paddle/fluid/operators/transpose_op.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/padding.h"
#if defined(__NVCC__) || defined(__HIPCC__)
#include "thrust/device_vector.h"
#endif
#if defined(PADDLE_WITH_ONEMKL)
#include "paddle/phi/backends/dynload/mklrt.h"
#elif defined(PADDLE_WITH_POCKETFFT)
#include "extern_pocketfft/pocketfft_hdronly.h"
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

// FFT Functors
#if defined(PADDLE_WITH_ONEMKL)

#define MKL_DFTI_CHECK(expr)                                                   \
  do {                                                                         \
    MKL_LONG status = (expr);                                                  \
    if (!phi::dynload::DftiErrorClass(status, DFTI_NO_ERROR))                  \
      PADDLE_THROW(                                                            \
          platform::errors::External(phi::dynload::DftiErrorMessage(status))); \
  } while (0);

struct DftiDescriptorDeleter {
  void operator()(DFTI_DESCRIPTOR_HANDLE handle) {
    if (handle != nullptr) {
      MKL_DFTI_CHECK(phi::dynload::DftiFreeDescriptor(&handle));
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
    MKL_DFTI_CHECK(phi::dynload::DftiCreateDescriptorX(
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

static DftiDescriptor _plan_mkl_fft(
    const framework::proto::VarType::Type& in_dtype,
    const framework::proto::VarType::Type& out_dtype,
    const framework::DDim& in_strides, const framework::DDim& out_strides,
    const std::vector<int>& signal_sizes, FFTNormMode normalization,
    bool forward) {
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
  MKL_DFTI_CHECK(phi::dynload::DftiSetValue(descriptor.get(), DFTI_PLACEMENT,
                                            DFTI_NOT_INPLACE));

  // number of transformations
  const MKL_LONG batch_size = fft_sizes[0];
  MKL_DFTI_CHECK(phi::dynload::DftiSetValue(
      descriptor.get(), DFTI_NUMBER_OF_TRANSFORMS, batch_size));

  // input & output distance
  const MKL_LONG idist = in_strides[0];
  const MKL_LONG odist = out_strides[0];
  MKL_DFTI_CHECK(
      phi::dynload::DftiSetValue(descriptor.get(), DFTI_INPUT_DISTANCE, idist));
  MKL_DFTI_CHECK(phi::dynload::DftiSetValue(descriptor.get(),
                                            DFTI_OUTPUT_DISTANCE, odist));

  // input & output stride
  std::vector<MKL_LONG> mkl_in_stride(1 + signal_ndim, 0);
  std::vector<MKL_LONG> mkl_out_stride(1 + signal_ndim, 0);
  for (MKL_LONG i = 1; i <= signal_ndim; i++) {
    mkl_in_stride[i] = in_strides[i];
    mkl_out_stride[i] = out_strides[i];
  }
  MKL_DFTI_CHECK(phi::dynload::DftiSetValue(
      descriptor.get(), DFTI_INPUT_STRIDES, mkl_in_stride.data()));
  MKL_DFTI_CHECK(phi::dynload::DftiSetValue(
      descriptor.get(), DFTI_OUTPUT_STRIDES, mkl_out_stride.data()));

  // conjugate even storage
  if (!(fft_type == FFTTransformType::C2C)) {
    MKL_DFTI_CHECK(phi::dynload::DftiSetValue(
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
    MKL_DFTI_CHECK(
        phi::dynload::DftiSetValue(descriptor.get(), scale_direction, scale));
  }

  // commit the descriptor
  MKL_DFTI_CHECK(phi::dynload::DftiCommitDescriptor(descriptor.get()));
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
      phi::vectorize(transposed_input_shape);
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
      phi::make_ddim(collapsed_input_shape_);
  transposed_input.Resize(collapsed_input_shape);
  framework::Tensor& collapsed_input = transposed_input;

  // make a collapsed output
  std::vector<int> collapsed_output_shape_(1 + signal_ndim);
  collapsed_output_shape_[0] = batch_size;
  for (int i = 0; i < signal_ndim; i++) {
    collapsed_output_shape_[1 + i] = out_sizes[axes[i]];
  }
  const framework::DDim collapsed_output_shape =
      phi::make_ddim(collapsed_output_shape_);
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
  const framework::DDim input_stride = phi::stride(collapsed_input_shape);
  const framework::DDim output_stride = phi::stride(collapsed_output_shape);

  // make a DFTI_DESCRIPTOR
  DftiDescriptor desc =
      _plan_mkl_fft(framework::TransToProtoVarType(x->dtype()),
                    framework::TransToProtoVarType(out->dtype()), input_stride,
                    output_stride, signal_sizes, normalization, forward);

  const FFTTransformType fft_type =
      GetFFTTransformType(framework::TransToProtoVarType(x->dtype()),
                          framework::TransToProtoVarType(out->type()));
  if (fft_type == FFTTransformType::C2R && forward) {
    framework::Tensor collapsed_input_conj(collapsed_input.dtype());
    collapsed_input_conj.mutable_data<Ti>(collapsed_input.dims(),
                                          ctx.GetPlace());
    // conjugate the input
    platform::ForRange<DeviceContext> for_range(ctx, collapsed_input.numel());
    phi::funcs::ConjFunctor<Ti> functor(collapsed_input.data<Ti>(),
                                        collapsed_input.numel(),
                                        collapsed_input_conj.data<Ti>());
    for_range(functor);
    MKL_DFTI_CHECK(phi::dynload::DftiComputeBackward(
        desc.get(), collapsed_input_conj.data(), collapsed_output.data()));
  } else if (fft_type == FFTTransformType::R2C && !forward) {
    framework::Tensor collapsed_output_conj(collapsed_output.dtype());
    collapsed_output_conj.mutable_data<To>(collapsed_output.dims(),
                                           ctx.GetPlace());
    MKL_DFTI_CHECK(phi::dynload::DftiComputeForward(
        desc.get(), collapsed_input.data(), collapsed_output_conj.data()));
    // conjugate the output
    platform::ForRange<DeviceContext> for_range(ctx, collapsed_output.numel());
    phi::funcs::ConjFunctor<To> functor(collapsed_output_conj.data<To>(),
                                        collapsed_output.numel(),
                                        collapsed_output.data<To>());
    for_range(functor);
  } else {
    if (forward) {
      MKL_DFTI_CHECK(phi::dynload::DftiComputeForward(
          desc.get(), collapsed_input.data(), collapsed_output.data()));
    } else {
      MKL_DFTI_CHECK(phi::dynload::DftiComputeBackward(
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

template <typename Ti, typename To>
struct FFTC2CFunctor<platform::CPUDeviceContext, Ti, To> {
  void operator()(const platform::CPUDeviceContext& ctx, const Tensor* x,
                  Tensor* out, const std::vector<int64_t>& axes,
                  FFTNormMode normalization, bool forward) {
    using R = typename Ti::value_type;
    using C = std::complex<R>;

    const auto& input_dim = x->dims();
    const std::vector<size_t> in_sizes = phi::vectorize<size_t>(input_dim);
    std::vector<std::ptrdiff_t> in_strides =
        phi::vectorize<std::ptrdiff_t>(phi::stride(input_dim));
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
    const std::vector<size_t> in_sizes = phi::vectorize<size_t>(input_dim);
    std::vector<std::ptrdiff_t> in_strides =
        phi::vectorize<std::ptrdiff_t>(phi::stride(input_dim));
    {
      const int64_t data_size = sizeof(R);
      std::transform(in_strides.begin(), in_strides.end(), in_strides.begin(),
                     [&](std::ptrdiff_t s) { return s * data_size; });
    }

    const auto& output_dim = out->dims();
    const std::vector<size_t> out_sizes = phi::vectorize<size_t>(output_dim);
    std::vector<std::ptrdiff_t> out_strides =
        phi::vectorize<std::ptrdiff_t>(phi::stride(output_dim));
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
    const std::vector<size_t> in_sizes = phi::vectorize<size_t>(input_dim);
    std::vector<std::ptrdiff_t> in_strides =
        phi::vectorize<std::ptrdiff_t>(phi::stride(input_dim));
    {
      const int64_t data_size = sizeof(C);
      std::transform(in_strides.begin(), in_strides.end(), in_strides.begin(),
                     [&](std::ptrdiff_t s) { return s * data_size; });
    }

    const auto& output_dim = out->dims();
    const std::vector<size_t> out_sizes = phi::vectorize<size_t>(output_dim);
    std::vector<std::ptrdiff_t> out_strides =
        phi::vectorize<std::ptrdiff_t>(phi::stride(output_dim));
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
      phi::vectorize<int64_t>(phi::stride(src->dims()));
  std::vector<int64_t> dst_strides_v =
      phi::vectorize<int64_t>(phi::stride(dst->dims()));
  std::vector<int64_t> dst_shape_v = phi::vectorize<int64_t>(dst->dims());
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

      phi::funcs::PaddingFunctor<DeviceContext, C>(
          rank, ctx.template device_context<DeviceContext>(), pads,
          static_cast<C>(0), *dy, &full_dy);
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
    const framework::DDim strides = phi::stride(dx->dims());

#if defined(__NVCC__) || defined(__HIPCC__)
    const thrust::device_vector<int64_t> strides_g(phi::vectorize(strides));
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
