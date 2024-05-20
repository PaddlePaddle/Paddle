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

#include <cmath>

#include "paddle/phi/kernels/funcs/fft.h"
#include "paddle/phi/kernels/funcs/fft_cache.h"

#include "paddle/common/ddim.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/kernels/assign_kernel.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/scale_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {
namespace funcs {
namespace detail {

// Use the optimized path to perform single R2C or C2R if transformation dim is
// supported by cuFFT
static bool use_optimized_fft_path(const std::vector<int64_t>& axes) {
  // For performance reason, when axes starts with (0, 1), do not use the
  // optimized path.
  if (axes.size() > kMaxFFTNdim ||
      (axes.size() >= 2 && axes[0] == 0 && axes[1] == 1)) {
    return false;
  } else {
    return true;
  }
}

static double fft_normalization_scale(FFTNormMode normalization,
                                      const std::vector<int64_t>& sizes,
                                      const std::vector<int64_t>& dims) {
  // auto norm = static_cast<fft_norm_mode>(normalization);
  if (normalization == FFTNormMode::none) {
    return static_cast<double>(1.0);
  }

  int64_t signal_numel = 1;
  for (auto dim : dims) {
    signal_numel *= sizes[dim];
  }
  const double scale_denom = (normalization == FFTNormMode::by_sqrt_n)
                                 ? std::sqrt(signal_numel)
                                 : static_cast<double>(signal_numel);
  return static_cast<double>(1.0 / scale_denom);
}

template <typename T>
void exec_normalization(const phi::GPUContext& ctx,
                        const DenseTensor& in,
                        DenseTensor* out,
                        FFTNormMode normalization,
                        const std::vector<int64_t>& sizes,
                        const std::vector<int64_t>& axes) {
  const double scale = fft_normalization_scale(normalization, sizes, axes);
  if (scale != 1.0) {
    ScaleKernel<T, phi::GPUContext>(ctx, in, scale, 0, true, out);
  } else {
    AssignKernel<phi::GPUContext>(ctx, in, out);
  }
}

bool has_large_prime_factor(int64_t n) {
  constexpr int64_t first_large_prime = 11;
  const std::array<int64_t, 4> prime_radices{{2, 3, 5, 7}};
  for (auto prime : prime_radices) {
    if (n < first_large_prime) {
      return false;
    }
    while (n % prime == 0) {
      n /= prime;
    }
  }
  return n != 1;
}

#if defined(PADDLE_WITH_CUDA)
inline bool use_cache(const int64_t* signal_size) {
  bool using_cache = true;
  int cufft_version;
  phi::dynload::cufftGetVersion(&cufft_version);
  if (10300 <= cufft_version && cufft_version <= 10400) {
    using_cache = std::none_of(
        signal_size + 1, signal_size + kMaxDataNdim, [](int64_t dim_size) {
          return has_large_prime_factor(dim_size);
        });
  }
  return using_cache;
}
#elif defined(PADDLE_WITH_HIP)
inline bool use_cache(const int64_t* signal_size) { return true; }
#endif

// up to 3d unnormalized fft transform (c2r, r2c, c2c)
template <typename Ti, typename To>
void exec_fft(const phi::GPUContext& ctx,
              const DenseTensor& x,
              DenseTensor* out,
              const std::vector<int64_t>& axes,
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
      Transpose<Ti, phi::GPUContext>(ctx, x, dim_permute);
  const phi::DDim transposed_input_shape = transposed_input.dims();

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

  FFTConfigKey key =
      create_fft_configkey(collapsed_input, collapsed_output, signal_ndim);
  int64_t device_id = ctx.GetPlace().GetDeviceId();
  FFTConfig* config = nullptr;
  std::unique_ptr<FFTConfig> config_ = nullptr;
  bool using_cache = use_cache(key.sizes_);

  if (using_cache) {
    FFTConfigCache& plan_cache = get_fft_plan_cache(device_id);
    std::unique_lock<std::mutex> guard(plan_cache.mutex, std::defer_lock);
    guard.lock();
    config = &(plan_cache.lookup(key));
  } else {
    config_ = std::make_unique<FFTConfig>(key);
    config = config_.get();
  }

  const int64_t workspace_size = static_cast<int64_t>(config->workspace_size());
  DenseTensor workspace_tensor = Empty<uint8_t>(ctx, {workspace_size});

  // prepare cufft for execution
#if defined(PADDLE_WITH_CUDA)
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cufftSetStream(config->plan(), ctx.stream()));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cufftSetWorkArea(config->plan(), workspace_tensor.data()));
#elif defined(PADDLE_WITH_HIP)
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::hipfftSetStream(config->plan(), ctx.stream()));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::hipfftSetWorkArea(config->plan(), workspace_tensor.data()));
#endif

  // execution of fft plan
  const FFTTransformType fft_type = config->transform_type();
  if (fft_type == FFTTransformType::C2R && forward) {
    ConjKernel<Ti, phi::GPUContext>(ctx, collapsed_input, &collapsed_input);
    exec_plan(*config, collapsed_input.data(), collapsed_output.data(), false);
  } else if (fft_type == FFTTransformType::R2C && !forward) {
    exec_plan(*config, collapsed_input.data(), collapsed_output.data(), true);
    ConjKernel<To, phi::GPUContext>(ctx, collapsed_output, &collapsed_output);
  } else {
    exec_plan(
        *config, collapsed_input.data(), collapsed_output.data(), forward);
  }

  // resize for the collapsed output
  collapsed_output.Resize(transposed_output_shape);
  phi::DenseTensor& transposed_output = collapsed_output;

  // reverse the transposition
  std::vector<int> reverse_dim_permute(ndim);
  for (int i = 0; i < ndim; i++) {
    reverse_dim_permute[dim_permute[i]] = i;
  }
  TransposeKernel<To, phi::GPUContext>(
      ctx, transposed_output, reverse_dim_permute, out);
}
}  // namespace detail

template <typename Ti, typename To>
struct FFTC2CFunctor<phi::GPUContext, Ti, To> {
  void operator()(const phi::GPUContext& ctx,
                  const DenseTensor& x,
                  DenseTensor* out,
                  const std::vector<int64_t>& axes,
                  FFTNormMode normalization,
                  bool forward) {
    if (axes.empty()) {
      AssignKernel<phi::GPUContext>(ctx, x, out);
      return;
    }

    std::vector<int64_t> working_axes = axes;
    std::sort(working_axes.begin(), working_axes.end());
    std::vector<int64_t> first_dims;
    size_t max_dims;

    DenseTensor working_tensor = x;  // shallow copy
    while (true) {
      max_dims = std::min(static_cast<size_t>(detail::kMaxFFTNdim),
                          working_axes.size());
      first_dims.assign(working_axes.end() - max_dims, working_axes.end());

      detail::exec_fft<Ti, To>(ctx, working_tensor, out, first_dims, forward);
      working_axes.resize(working_axes.size() - max_dims);
      first_dims.clear();

      if (working_axes.empty()) {
        break;
      }

      if (working_tensor.IsSharedWith(x)) {
        working_tensor = std::move(*out);
        *out = EmptyLike<Ti>(ctx, x);
      } else {
        std::swap(*out, working_tensor);
      }
    }

    std::vector<int64_t> out_dims = common::vectorize(x.dims());
    detail::exec_normalization<To>(
        ctx, *out, out, normalization, out_dims, axes);
  }
};

template <typename Ti, typename To>
struct FFTC2RFunctor<phi::GPUContext, Ti, To> {
  void operator()(const phi::GPUContext& ctx,
                  const DenseTensor& x,
                  DenseTensor* out,
                  const std::vector<int64_t>& axes,
                  FFTNormMode normalization,
                  bool forward) {
    std::vector<int64_t> out_dims = common::vectorize(out->dims());

    if (detail::use_optimized_fft_path(axes)) {
      DenseTensor x_copy = Assign(ctx, x);
      detail::exec_fft<Ti, To>(ctx, x_copy, out, axes, forward);
    } else {
      DenseTensor c2c_result = EmptyLike<Ti, phi::GPUContext>(ctx, x);
      FFTC2CFunctor<phi::GPUContext, Ti, Ti> c2c_functor;
      c2c_functor(ctx,
                  x,
                  &c2c_result,
                  {axes.begin(), axes.end() - 1},
                  FFTNormMode::none,
                  forward);
      detail::exec_fft<Ti, To>(ctx, c2c_result, out, {axes.back()}, forward);
    }
    detail::exec_normalization<To>(
        ctx, *out, out, normalization, out_dims, axes);
  }
};

template <typename Ti, typename To>
struct FFTR2CFunctor<phi::GPUContext, Ti, To> {
  void operator()(const phi::GPUContext& ctx,
                  const DenseTensor& x,
                  DenseTensor* out,
                  const std::vector<int64_t>& axes,
                  FFTNormMode normalization,
                  bool forward) {
    if (detail::use_optimized_fft_path(axes)) {
      detail::exec_fft<Ti, To>(ctx, x, out, axes, forward);
    } else {
      DenseTensor r2c_result = EmptyLike<To, phi::GPUContext>(ctx, *out);
      detail::exec_fft<Ti, To>(ctx, x, &r2c_result, {axes.back()}, forward);

      FFTC2CFunctor<phi::GPUContext, To, To> fft_c2c_func;
      fft_c2c_func(ctx,
                   r2c_result,
                   out,
                   {axes.begin(), axes.end() - 1},
                   FFTNormMode::none,
                   forward);
    }

    const auto in_dims = common::vectorize(x.dims());
    detail::exec_normalization<To>(
        ctx, *out, out, normalization, in_dims, axes);
  }
};

using complex64_t = phi::dtype::complex<float>;
using complex128_t = phi::dtype::complex<double>;
template struct FFTC2CFunctor<phi::GPUContext, complex64_t, complex64_t>;
template struct FFTC2CFunctor<phi::GPUContext, complex128_t, complex128_t>;
template struct FFTC2RFunctor<phi::GPUContext, complex64_t, float>;
template struct FFTC2RFunctor<phi::GPUContext, complex128_t, double>;
template struct FFTR2CFunctor<phi::GPUContext, float, complex64_t>;
template struct FFTR2CFunctor<phi::GPUContext, double, complex128_t>;

}  // namespace funcs
}  // namespace phi
