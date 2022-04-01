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

#pragma once

#include "paddle/fluid/operators/spectral_op.h"

#if defined(PADDLE_WITH_ONEMKL)
#include "paddle/phi/backends/dynload/mklrt.h"
#elif defined(PADDLE_WITH_POCKETFFT)
#include "extern_pocketfft/pocketfft_hdronly.h"
#endif

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

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

}  // namespace operators
}  // namespace paddle
