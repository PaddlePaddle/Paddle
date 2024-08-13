// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/funcs/fft.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"
#if defined(PADDLE_WITH_ONEMKL)
#include "paddle/phi/kernels/funcs/mkl_fft_utils.h"
#elif defined(PADDLE_WITH_POCKETFFT)
#define POCKETFFT_CACHE_SIZE 16
#include "extern_pocketfft/pocketfft_hdronly.h"
#endif

namespace phi::funcs {
#if defined(PADDLE_WITH_ONEMKL)

}  // namespace phi::funcs
namespace phi::funcs::detail {
// Execute a general fft operation (can be c2c, onesided r2c or onesided c2r)
template <typename Ti, typename To>
void exec_fft(const phi::CPUContext& ctx,
              const DenseTensor& x,
              DenseTensor* out,
              const std::vector<int64_t>& axes,
              FFTNormMode normalization,
              bool forward) {
  const phi::DDim& in_sizes = x.dims();
  const int ndim = in_sizes.size();
  const int signal_ndim = axes.size();
  const int batch_ndim = ndim - signal_ndim;
  const phi::DDim& out_sizes = out->dims();

  // make a dim permutation
  std::vector<int> dim_permute(ndim);
  std::iota(dim_permute.begin(), dim_permute.end(), 0);
  std::vector<bool> is_transformed_dim(ndim, false);
  for (const auto& d : axes) {
    is_transformed_dim[d] = true;
  }
  const auto batch_end =
      std::partition(dim_permute.begin(), dim_permute.end(), [&](size_t axis) {
        return !is_transformed_dim[axis];
      });
  std::copy(axes.cbegin(), axes.cend(), batch_end);

  // transpose input according to the permutation
  DenseTensor transposed_input =
      Transpose<Ti, phi::CPUContext>(ctx, x, dim_permute);
  const phi::DDim& transposed_input_shape = transposed_input.dims();

  // batch size
  int64_t batch_size = 1L;
  for (int i = 0; i < batch_ndim; i++) {
    batch_size *= transposed_input_shape[i];
  }

  // make an collapsed input: collapse batch axes for input
  std::vector<int64_t> collapsed_input_shape_;
  collapsed_input_shape_.reserve(1 + signal_ndim);
  collapsed_input_shape_.emplace_back(batch_size);
  for (int i = 0; i < signal_ndim; i++) {
    collapsed_input_shape_.push_back(in_sizes[axes[i]]);
  }
  phi::DDim collapsed_input_shape = common::make_ddim(collapsed_input_shape_);
  transposed_input.Resize(collapsed_input_shape);
  DenseTensor& collapsed_input = transposed_input;

  // make a collapsed output
  phi::DDim transposed_output_shape = out_sizes.transpose(dim_permute);
  std::vector<int64_t> collapsed_output_shape_;
  collapsed_output_shape_.reserve(1 + signal_ndim);
  collapsed_output_shape_.emplace_back(batch_size);
  for (int i = 0; i < signal_ndim; i++) {
    collapsed_output_shape_.push_back(out_sizes[axes[i]]);
  }
  phi::DDim collapsed_output_shape = common::make_ddim(collapsed_output_shape_);
  DenseTensor collapsed_output;
  collapsed_output.Resize(collapsed_output_shape);
  ctx.Alloc<To>(&collapsed_output);

  // make a DFTI_DESCRIPTOR
  std::vector<int64_t> signal_sizes(1 + signal_ndim);
  signal_sizes[0] = batch_size;
  for (int i = 0; i < signal_ndim; i++) {
    signal_sizes[1 + i] =
        std::max(collapsed_input_shape[1 + i], collapsed_output_shape[1 + i]);
  }
  const phi::DDim input_stride = common::stride(collapsed_input_shape);
  const phi::DDim output_stride = common::stride(collapsed_output_shape);

  DftiDescriptor desc = plan_mkl_fft(x.dtype(),
                                     out->dtype(),
                                     input_stride,
                                     output_stride,
                                     signal_sizes,
                                     normalization,
                                     forward);
  // execute the transform
  const FFTTransformType fft_type = GetFFTTransformType(x.dtype(), out->type());
  if (fft_type == FFTTransformType::C2R && forward) {
    ConjKernel<Ti, phi::CPUContext>(ctx, collapsed_input, &collapsed_input);
    MKL_DFTI_CHECK(phi::dynload::DftiComputeBackward(
        desc.get(), collapsed_input.data(), collapsed_output.data()));
  } else if (fft_type == FFTTransformType::R2C && !forward) {
    MKL_DFTI_CHECK(phi::dynload::DftiComputeForward(
        desc.get(), collapsed_input.data(), collapsed_output.data()));
    ConjKernel<To, phi::CPUContext>(ctx, collapsed_output, &collapsed_output);
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
  collapsed_output.Resize(transposed_output_shape);
  phi::DenseTensor& transposed_output = collapsed_output;

  // reverse the transposition
  std::vector<int> reverse_dim_permute(ndim);
  for (int i = 0; i < ndim; i++) {
    reverse_dim_permute[dim_permute[i]] = i;
  }
  TransposeKernel<To, phi::CPUContext>(
      ctx, transposed_output, reverse_dim_permute, out);
}
}  // namespace phi::funcs::detail
namespace phi::funcs {

template <typename Ti, typename To>
struct FFTC2CFunctor<phi::CPUContext, Ti, To> {
  void operator()(const phi::CPUContext& ctx,
                  const DenseTensor& x,
                  DenseTensor* out,
                  const std::vector<int64_t>& axes,
                  FFTNormMode normalization,
                  bool forward) {
    detail::exec_fft<Ti, To>(ctx, x, out, axes, normalization, forward);
  }
};

template <typename Ti, typename To>
struct FFTR2CFunctor<phi::CPUContext, Ti, To> {
  void operator()(const phi::CPUContext& ctx,
                  const DenseTensor& x,
                  DenseTensor* out,
                  const std::vector<int64_t>& axes,
                  FFTNormMode normalization,
                  bool forward) {
    detail::exec_fft<Ti, To>(ctx, x, out, axes, normalization, forward);
  }
};

template <typename Ti, typename To>
struct FFTC2RFunctor<phi::CPUContext, Ti, To> {
  void operator()(const phi::CPUContext& ctx,
                  const DenseTensor& x,
                  DenseTensor* out,
                  const std::vector<int64_t>& axes,
                  FFTNormMode normalization,
                  bool forward) {
    if (axes.size() > 1) {
      DenseTensor c2c_result = EmptyLike<Ti, phi::CPUContext>(ctx, x);

      const std::vector<int64_t> c2c_dims(axes.begin(), axes.end() - 1);
      FFTC2CFunctor<phi::CPUContext, Ti, Ti> c2c_functor;
      c2c_functor(ctx, x, &c2c_result, c2c_dims, normalization, forward);

      const std::vector<int64_t> new_axes{axes.back()};
      detail::exec_fft<Ti, To>(
          ctx, c2c_result, out, new_axes, normalization, forward);
    } else {
      detail::exec_fft<Ti, To>(ctx, x, out, axes, normalization, forward);
    }
  }
};

#elif defined(PADDLE_WITH_POCKETFFT)
}  // namespace phi::funcs
namespace phi::funcs::detail {
template <typename T>
static T compute_factor(size_t size, FFTNormMode normalization) {
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
      common::errors::InvalidArgument("Unsupported normalization type"));
}
}  // namespace phi::funcs::detail
namespace phi::funcs {

template <typename Ti, typename To>
struct FFTC2CFunctor<phi::CPUContext, Ti, To> {
  void operator()(const phi::CPUContext& ctx UNUSED,
                  const DenseTensor& x,
                  DenseTensor* out,
                  const std::vector<int64_t>& axes,
                  FFTNormMode normalization,
                  bool forward) {
    using R = typename Ti::value_type;
    using C = std::complex<R>;

    const auto& input_dim = x.dims();
    const std::vector<size_t> in_sizes = common::vectorize<size_t>(input_dim);
    std::vector<std::ptrdiff_t> in_strides =
        common::vectorize<std::ptrdiff_t>(common::stride(input_dim));
    const int64_t data_size = sizeof(C);
    std::transform(in_strides.begin(),
                   in_strides.end(),
                   in_strides.begin(),
                   [&](std::ptrdiff_t s) { return s * data_size; });

    const auto* in_data = reinterpret_cast<const C*>(x.data<Ti>());
    auto* out_data = reinterpret_cast<C*>(out->data<To>());
    // pocketfft requires std::vector<size_t>
    std::vector<size_t> axes_(axes.size());
    std::copy(axes.begin(), axes.end(), axes_.begin());
    // compute factor
    size_t signal_numel = 1;
    for (const auto axis : axes) {
      signal_numel *= in_sizes[axis];
    }
    R factor = detail::compute_factor<R>(signal_numel, normalization);
    pocketfft::c2c(in_sizes,
                   in_strides,
                   in_strides,
                   axes_,
                   forward,
                   in_data,
                   out_data,
                   factor);
  }
};

template <typename Ti, typename To>
struct FFTR2CFunctor<phi::CPUContext, Ti, To> {
  void operator()(const phi::CPUContext& ctx UNUSED,
                  const DenseTensor& x,
                  DenseTensor* out,
                  const std::vector<int64_t>& axes,
                  FFTNormMode normalization,
                  bool forward) {
    using R = Ti;
    using C = std::complex<R>;

    const auto& input_dim = x.dims();
    const std::vector<size_t> in_sizes = common::vectorize<size_t>(input_dim);
    std::vector<std::ptrdiff_t> in_strides =
        common::vectorize<std::ptrdiff_t>(common::stride(input_dim));
    {
      const int64_t data_size = sizeof(R);
      std::transform(in_strides.begin(),
                     in_strides.end(),
                     in_strides.begin(),
                     [&](std::ptrdiff_t s) { return s * data_size; });
    }

    const auto& output_dim = out->dims();
    const std::vector<size_t> out_sizes = common::vectorize<size_t>(output_dim);
    std::vector<std::ptrdiff_t> out_strides =
        common::vectorize<std::ptrdiff_t>(common::stride(output_dim));
    {
      const int64_t data_size = sizeof(C);
      std::transform(out_strides.begin(),
                     out_strides.end(),
                     out_strides.begin(),
                     [&](std::ptrdiff_t s) { return s * data_size; });
    }

    const auto* in_data = x.data<R>();
    auto* out_data = reinterpret_cast<C*>(out->data<To>());
    // pocketfft requires std::vector<size_t>
    std::vector<size_t> axes_(axes.size());
    std::copy(axes.begin(), axes.end(), axes_.begin());
    // compute normalization factor
    size_t signal_numel = 1;
    for (const auto axis : axes) {
      signal_numel *= in_sizes[axis];
    }
    R factor = detail::compute_factor<R>(signal_numel, normalization);
    pocketfft::r2c(in_sizes,
                   in_strides,
                   out_strides,
                   axes_,
                   forward,
                   in_data,
                   out_data,
                   factor);
  }
};

template <typename Ti, typename To>
struct FFTC2RFunctor<phi::CPUContext, Ti, To> {
  void operator()(const phi::CPUContext& ctx UNUSED,
                  const DenseTensor& x,
                  DenseTensor* out,
                  const std::vector<int64_t>& axes,
                  FFTNormMode normalization,
                  bool forward) {
    using R = To;
    using C = std::complex<R>;

    const auto& input_dim = x.dims();
    const std::vector<size_t> in_sizes = common::vectorize<size_t>(input_dim);
    std::vector<std::ptrdiff_t> in_strides =
        common::vectorize<std::ptrdiff_t>(common::stride(input_dim));
    {
      const int64_t data_size = sizeof(C);
      std::transform(in_strides.begin(),
                     in_strides.end(),
                     in_strides.begin(),
                     [&](std::ptrdiff_t s) { return s * data_size; });
    }

    const auto& output_dim = out->dims();
    const std::vector<size_t> out_sizes = common::vectorize<size_t>(output_dim);
    std::vector<std::ptrdiff_t> out_strides =
        common::vectorize<std::ptrdiff_t>(common::stride(output_dim));
    {
      const int64_t data_size = sizeof(R);
      std::transform(out_strides.begin(),
                     out_strides.end(),
                     out_strides.begin(),
                     [&](std::ptrdiff_t s) { return s * data_size; });
    }

    const auto* in_data = reinterpret_cast<const C*>(x.data<Ti>());
    auto* out_data = out->data<R>();
    // pocketfft requires std::vector<size_t>
    std::vector<size_t> axes_(axes.size());
    std::copy(axes.begin(), axes.end(), axes_.begin());
    // compute normalization factor
    size_t signal_numel = 1;
    for (const auto axis : axes) {
      signal_numel *= out_sizes[axis];
    }
    R factor = detail::compute_factor<R>(signal_numel, normalization);
    pocketfft::c2r(out_sizes,
                   in_strides,
                   out_strides,
                   axes_,
                   forward,
                   in_data,
                   out_data,
                   factor);
  }
};
#endif

using complex64_t = phi::dtype::complex<float>;
using complex128_t = phi::dtype::complex<double>;
template struct FFTC2CFunctor<phi::CPUContext, complex64_t, complex64_t>;
template struct FFTC2CFunctor<phi::CPUContext, complex128_t, complex128_t>;
template struct FFTC2RFunctor<phi::CPUContext, complex64_t, float>;
template struct FFTC2RFunctor<phi::CPUContext, complex128_t, double>;
template struct FFTR2CFunctor<phi::CPUContext, float, complex64_t>;
template struct FFTR2CFunctor<phi::CPUContext, double, complex128_t>;
}  // namespace phi::funcs
