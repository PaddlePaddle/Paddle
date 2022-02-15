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
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/operators/conj_op.h"
#include "paddle/fluid/operators/math/complex_functors.h"
#include "paddle/fluid/operators/spectral_helper.h"
#include "paddle/fluid/operators/spectral_op.h"
#include "paddle/fluid/operators/transpose_op.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {

namespace {

// Calculates the normalization constant
double fft_normalization_scale(FFTNormMode normalization,
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

template <typename DeviceContext, typename T>
void exec_normalization(const DeviceContext& ctx, const Tensor* in, Tensor* out,
                        FFTNormMode normalization,
                        const std::vector<int64_t>& sizes,
                        const std::vector<int64_t>& axes) {
  double scale = fft_normalization_scale(normalization, sizes, axes);
  if (scale != 1.0) {
    auto eigen_out = framework::EigenVector<T>::Flatten(*out);
    auto eigen_in = framework::EigenVector<T>::Flatten(*in);
    auto dev = ctx.eigen_device();
    EigenScale<Eigen::GpuDevice, T>::Eval(*dev, eigen_out, eigen_in,
                                          static_cast<T>(scale),
                                          static_cast<T>(0), false);
  } else {
    framework::TensorCopy(*in, ctx.GetPlace(), out);
  }
}

#if defined(PADDLE_WITH_CUDA)
FFTConfigKey create_fft_configkey(const framework::Tensor& input,
                                  const framework::Tensor& output,
                                  int signal_ndim) {
  // Create the transform plan (either from cache or locally)
  const auto value_type =
      framework::IsComplexType(framework::TransToProtoVarType(input.dtype()))
          ? framework::ToRealType(framework::TransToProtoVarType(input.dtype()))
          : framework::TransToProtoVarType(input.dtype());
  auto fft_type =
      GetFFTTransformType(framework::TransToProtoVarType(input.dtype()),
                          framework::TransToProtoVarType(output.dtype()));
  // signal sizes
  std::vector<int64_t> signal_size(signal_ndim + 1);

  signal_size[0] = input.dims()[0];
  for (int64_t i = 1; i <= signal_ndim; ++i) {
    auto in_size = input.dims()[i];
    auto out_size = output.dims()[i];
    signal_size[i] = std::max(in_size, out_size);
  }
  FFTConfigKey key(framework::vectorize(input.dims()),
                   framework::vectorize(output.dims()), signal_size, fft_type,
                   value_type);
  return key;
}

// Execute a pre-planned transform
static void exec_cufft_plan_raw(const FFTConfig& config, void* in_data,
                                void* out_data, bool forward) {
  auto& plan = config.plan();

  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cufftXtExec(
      plan, in_data, out_data, forward ? CUFFT_FORWARD : CUFFT_INVERSE));
}

template <typename DeviceContext, typename Ti, typename To>
void exec_cufft_plan(const DeviceContext& ctx, const FFTConfig& config,
                     framework::Tensor* input, framework::Tensor* output,
                     bool forward) {
  // execute transform plan
  auto fft_type = config.transform_type();
  if (fft_type == FFTTransformType::C2R && forward) {
    forward = false;
    framework::Tensor input_conj(input->type());
    input_conj.mutable_data<Ti>(input->dims(), ctx.GetPlace());
    platform::ForRange<DeviceContext> for_range(ctx, input->numel());
    math::ConjFunctor<Ti> functor(input->data<Ti>(), input->numel(),
                                  input_conj.data<Ti>());
    for_range(functor);
    exec_cufft_plan_raw(config, input_conj.data(), output->data(), forward);
  } else if (fft_type == FFTTransformType::R2C && !forward) {
    forward = true;
    framework::Tensor out_conj(output->type());
    out_conj.mutable_data<To>(output->dims(), ctx.GetPlace());
    exec_cufft_plan_raw(config, input->data(), out_conj.data(), forward);

    platform::ForRange<DeviceContext> for_range(ctx, output->numel());
    math::ConjFunctor<To> functor(out_conj.data<To>(), output->numel(),
                                  output->data<To>());
    for_range(functor);
  } else {
    exec_cufft_plan_raw(config, input->data(), output->data(), forward);
  }
}

#elif defined(PADDLE_WITH_HIP)

FFTConfigKey create_fft_configkey(const framework::Tensor& input,
                                  const framework::Tensor& output,
                                  int signal_ndim) {
  // Create the transform plan (either from cache or locally)
  const auto value_type =
      framework::IsComplexType(framework::TransToProtoVarType(input.dtype()))
          ? framework::ToRealType(framework::TransToProtoVarType(input.dtype()))
          : framework::TransToProtoVarType(input.dtype());
  auto fft_type =
      GetFFTTransformType(framework::TransToProtoVarType(input.dtype()),
                          framework::TransToProtoVarType(output.type()));
  // signal sizes
  std::vector<int64_t> signal_size(signal_ndim + 1);

  signal_size[0] = input.dims()[0];
  for (int64_t i = 1; i <= signal_ndim; ++i) {
    auto in_size = input.dims()[i];
    auto out_size = output.dims()[i];
    signal_size[i] = std::max(in_size, out_size);
  }
  FFTConfigKey key(framework::vectorize(input.dims()),
                   framework::vectorize(output.dims()), signal_size, fft_type,
                   value_type);
  return key;
}

// Execute a pre-planned transform
static void exec_hipfft_plan_raw(const FFTConfig& config, void* in_data,
                                 void* out_data, bool forward) {
  auto& plan = config.plan();

  auto value_type = config.data_type();
  if (value_type == framework::proto::VarType::FP32) {
    switch (config.transform_type()) {
      case FFTTransformType::C2C: {
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::hipfftExecC2C(
            plan, static_cast<hipfftComplex*>(in_data),
            static_cast<hipfftComplex*>(out_data),
            forward ? HIPFFT_FORWARD : HIPFFT_BACKWARD));
        return;
      }
      case FFTTransformType::R2C: {
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::hipfftExecR2C(
            plan, static_cast<hipfftReal*>(in_data),
            static_cast<hipfftComplex*>(out_data)));
        return;
      }
      case FFTTransformType::C2R: {
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::hipfftExecC2R(
            plan, static_cast<hipfftComplex*>(in_data),
            static_cast<hipfftReal*>(out_data)));
        return;
      }
    }
  } else if (value_type == framework::proto::VarType::FP64) {
    switch (config.transform_type()) {
      case FFTTransformType::C2C: {
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::hipfftExecZ2Z(
            plan, static_cast<hipfftDoubleComplex*>(in_data),
            static_cast<hipfftDoubleComplex*>(out_data),
            forward ? HIPFFT_FORWARD : HIPFFT_BACKWARD));
        return;
      }
      case FFTTransformType::R2C: {
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::hipfftExecD2Z(
            plan, static_cast<hipfftDoubleReal*>(in_data),
            static_cast<hipfftDoubleComplex*>(out_data)));
        return;
      }
      case FFTTransformType::C2R: {
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::hipfftExecZ2D(
            plan, static_cast<hipfftDoubleComplex*>(in_data),
            static_cast<hipfftDoubleReal*>(out_data)));
        return;
      }
    }
  }
  PADDLE_THROW(platform::errors::InvalidArgument(
      "hipFFT only support transforms of type float32 and float64"));
}

template <typename DeviceContext, typename Ti, typename To>
void exec_hipfft_plan(const DeviceContext& ctx, const FFTConfig& config,
                      framework::Tensor* input, framework::Tensor* output,
                      bool forward) {
  auto fft_type = config.transform_type();
  if (fft_type == FFTTransformType::C2R && forward) {
    forward = false;
    framework::Tensor input_conj(input->type());
    input_conj.mutable_data<Ti>(input->dims(), ctx.GetPlace());
    platform::ForRange<DeviceContext> for_range(ctx, input->numel());
    math::ConjFunctor<Ti> functor(input->data<Ti>(), input->numel(),
                                  input_conj.data<Ti>());
    for_range(functor);
    exec_hipfft_plan_raw(config, input_conj.data(), output->data(), forward);
  } else if (fft_type == FFTTransformType::R2C && !forward) {
    forward = true;
    framework::Tensor out_conj(output->type());
    out_conj.mutable_data<To>(output->dims(), ctx.GetPlace());
    exec_hipfft_plan_raw(config, input->data(), out_conj.data(), forward);

    platform::ForRange<DeviceContext> for_range(ctx, output->numel());
    math::ConjFunctor<To> functor(out_conj.data<To>(), output->numel(),
                                  output->data<To>());
    for_range(functor);
  } else {
    exec_hipfft_plan_raw(config, input->data(), output->data(), forward);
  }
}

#endif

// Execute a general unnormalized fft operation (can be c2c, onesided r2c or
// onesided c2r)
template <typename DeviceContext, typename Ti, typename To>
void exec_fft(const DeviceContext& ctx, const Tensor* X, Tensor* out,
              const std::vector<int64_t>& dim, bool forward) {
  const auto x_dims = framework::vectorize(X->dims());
  const int64_t ndim = static_cast<int64_t>(X->dims().size());
  auto tensor_place = ctx.GetPlace();

  // make a dim permutation
  std::vector<int> dim_permute(ndim);
  std::iota(dim_permute.begin(), dim_permute.end(), int{0});
  std::vector<bool> is_transformed_dim(ndim);
  for (const auto& d : dim) {
    is_transformed_dim[d] = true;
  }
  auto batch_end =
      std::partition(dim_permute.begin(), dim_permute.end(),
                     [&](int64_t d) { return !is_transformed_dim[d]; });
  std::sort(dim_permute.begin(), batch_end);
  std::copy(dim.cbegin(), dim.cend(), batch_end);

  // transpose input according to dim permutation
  auto transposed_input_shape = X->dims().transpose(dim_permute);
  framework::Tensor transposed_input;
  transposed_input.Resize(transposed_input_shape);
  transposed_input.mutable_data<Ti>(tensor_place);
  TransCompute<DeviceContext, Ti>(ndim, ctx, *X, &transposed_input,
                                  dim_permute);

  // Reshape batch dimensions into a single dimension
  const int64_t signal_ndim = static_cast<int64_t>(dim.size());
  std::vector<int64_t> collapsed_input_shape(signal_ndim + 1);

  auto transposed_input_shape_ = framework::vectorize(transposed_input_shape);
  const int64_t batch_dims = ndim - signal_ndim;
  auto batch_size =
      std::accumulate(transposed_input_shape_.begin(),
                      transposed_input_shape_.begin() + batch_dims,
                      static_cast<int>(1), std::multiplies<int>());
  collapsed_input_shape[0] = batch_size;

  std::copy(transposed_input_shape_.begin() + batch_dims,
            transposed_input_shape_.end(), collapsed_input_shape.begin() + 1);

  framework::Tensor& collapsed_input = transposed_input;
  collapsed_input.Resize(framework::make_ddim(collapsed_input_shape));

  // make a collpased output
  const auto out_dims = framework::vectorize(out->dims());
  std::vector<int64_t> collapsed_output_shape(1 + signal_ndim);
  collapsed_output_shape[0] = batch_size;
  for (size_t i = 0; i < dim.size(); ++i) {
    collapsed_output_shape[i + 1] = out_dims[dim[i]];
  }
  framework::Tensor collapsed_output;
  collapsed_output.Resize(framework::make_ddim(collapsed_output_shape));
  collapsed_output.mutable_data<To>(tensor_place);

  FFTConfig* config = nullptr;

#if defined(PADDLE_WITH_CUDA)
  std::unique_ptr<FFTConfig> config_ = nullptr;
  // create plan
  FFTConfigKey key =
      create_fft_configkey(collapsed_input, collapsed_output, signal_ndim);
  bool using_cache = false;
#if !defined(CUFFT_VERSION) || (CUFFT_VERSION < 10200)
  using_cache = true;
#endif

  if (using_cache) {
    const int64_t device_id = static_cast<int64_t>(
        reinterpret_cast<const platform::CUDAPlace*>(&collapsed_input.place())
            ->GetDeviceId());
    FFTConfigCache& plan_cache = get_fft_plan_cache(device_id);
    std::unique_lock<std::mutex> guard(plan_cache.mutex, std::defer_lock);
    guard.lock();
    config = &(plan_cache.lookup(key));
  } else {
    config_ = std::make_unique<FFTConfig>(key);
    config = config_.get();
  }

  // prepare cufft for execution
  PADDLE_ENFORCE_GPU_SUCCESS(
      platform::dynload::cufftSetStream(config->plan(), ctx.stream()));
  framework::Tensor workspace_tensor;
  workspace_tensor.mutable_data<To>(tensor_place, config->workspace_size());
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cufftSetWorkArea(
      config->plan(), workspace_tensor.data<To>()));
  // execute transform plan
  exec_cufft_plan<DeviceContext, Ti, To>(ctx, *config, &collapsed_input,
                                         &collapsed_output, forward);

#elif defined(PADDLE_WITH_HIP)
  // create plan
  FFTConfigKey key =
      create_fft_configkey(collapsed_input, collapsed_output, signal_ndim);
  const int64_t device_id = static_cast<int64_t>(
      reinterpret_cast<const platform::CUDAPlace*>(&collapsed_input.place())
          ->GetDeviceId());
  FFTConfigCache& plan_cache = get_fft_plan_cache(device_id);
  std::unique_lock<std::mutex> guard(plan_cache.mutex, std::defer_lock);
  guard.lock();
  config = &(plan_cache.lookup(key));

  // prepare cufft for execution
  PADDLE_ENFORCE_GPU_SUCCESS(
      platform::dynload::hipfftSetStream(config->plan(), ctx.stream()));
  framework::Tensor workspace_tensor;
  workspace_tensor.mutable_data<To>(tensor_place, config->workspace_size());
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::hipfftSetWorkArea(
      config->plan(), workspace_tensor.data<To>()));
  // execute transform plan
  exec_hipfft_plan<DeviceContext, Ti, To>(ctx, *config, &collapsed_input,
                                          &collapsed_output, forward);
#endif

  // Inverting output by reshape and transpose to original batch and dimension
  auto transposed_out_shape = out->dims().transpose(dim_permute);

  collapsed_output.Resize(transposed_out_shape);
  auto& transposed_output = collapsed_output;

  std::vector<int> reverse_dim_permute(ndim);
  for (size_t i = 0; i < ndim; i++) {
    reverse_dim_permute[dim_permute[i]] = i;
  }

  TransCompute<DeviceContext, To>(ndim, ctx, transposed_output, out,
                                  reverse_dim_permute);
}

}  // anonymous namespace

// Use the optimized path to perform single R2C or C2R if transformation dim is
// supported by cuFFT
bool use_optimized_fft_path(const std::vector<int64_t>& axes) {
  // For performance reason, when axes starts with (0, 1), do not use the
  // optimized path.
  if (axes.size() > kMaxFFTNdim ||
      (axes.size() >= 2 && axes[0] == 0 && axes[1] == 1)) {
    return false;
  } else {
    return true;
  }
}

template <typename Ti, typename To>
struct FFTC2CFunctor<platform::CUDADeviceContext, Ti, To> {
  void operator()(const platform::CUDADeviceContext& ctx, const Tensor* X,
                  Tensor* out, const std::vector<int64_t>& axes,
                  FFTNormMode normalization, bool forward) {
    if (axes.empty()) {
      framework::TensorCopy(*X, ctx.GetPlace(), out);
      return;
    }

    framework::Tensor* p_out = out;
    std::vector<int64_t> out_dims = framework::vectorize(X->dims());
    std::vector<int64_t> working_axes(axes.begin(), axes.end());
    std::vector<int64_t> first_dims;
    size_t max_dims;
    framework::Tensor working_tensor;
    working_tensor.mutable_data<Ti>(X->dims(), ctx.GetPlace());
    framework::Tensor* p_working_tensor = &working_tensor;
    framework::TensorCopy(*X, ctx.GetPlace(), &working_tensor);

    while (true) {
      max_dims =
          std::min(static_cast<size_t>(kMaxFFTNdim), working_axes.size());
      first_dims.assign(working_axes.end() - max_dims, working_axes.end());

      exec_fft<platform::CUDADeviceContext, Ti, To>(ctx, p_working_tensor,
                                                    p_out, first_dims, forward);
      working_axes.resize(working_axes.size() - max_dims);
      first_dims.clear();

      if (working_axes.empty()) {
        break;
      }

      std::swap(p_out, p_working_tensor);
    }
    exec_normalization<platform::CUDADeviceContext, To>(
        ctx, p_out, out, normalization, out_dims, axes);
  }
};

template <typename Ti, typename To>
struct FFTC2RFunctor<platform::CUDADeviceContext, Ti, To> {
  void operator()(const platform::CUDADeviceContext& ctx, const Tensor* X,
                  Tensor* out, const std::vector<int64_t>& axes,
                  FFTNormMode normalization, bool forward) {
    std::vector<int64_t> in_dims = framework::vectorize(X->dims());
    std::vector<int64_t> out_dims = framework::vectorize(out->dims());

    if (use_optimized_fft_path(axes)) {
      framework::Tensor x_copy(X->type());
      x_copy.mutable_data<Ti>(X->dims(), ctx.GetPlace());
      framework::TensorCopy(*X, ctx.GetPlace(), &x_copy);
      exec_fft<platform::CUDADeviceContext, Ti, To>(ctx, &x_copy, out, axes,
                                                    forward);
    } else {
      framework::Tensor temp_tensor;
      temp_tensor.mutable_data<Ti>(X->dims(), ctx.GetPlace());
      const std::vector<int64_t> dims(axes.begin(), axes.end() - 1);

      FFTC2CFunctor<platform::CUDADeviceContext, Ti, Ti> c2c_functor;
      c2c_functor(ctx, X, &temp_tensor, dims, FFTNormMode::none, forward);

      exec_fft<platform::CUDADeviceContext, Ti, To>(ctx, &temp_tensor, out,
                                                    {axes.back()}, forward);
    }
    exec_normalization<platform::CUDADeviceContext, To>(
        ctx, out, out, normalization, out_dims, axes);
  }
};

// n dimension real to complex FFT use cufft lib
template <typename Ti, typename To>
struct FFTR2CFunctor<platform::CUDADeviceContext, Ti, To> {
  void operator()(const platform::CUDADeviceContext& ctx, const Tensor* X,
                  Tensor* out, const std::vector<int64_t>& axes,
                  FFTNormMode normalization, bool forward) {
    // Step1: R2C transform on the last dimension
    framework::Tensor* r2c_out = out;
    const std::vector<int64_t> last_dim{axes.back()};
    std::vector<int64_t> out_dims = framework::vectorize(out->dims());
    exec_fft<platform::CUDADeviceContext, Ti, To>(ctx, X, r2c_out, last_dim,
                                                  forward);

    // Step2: C2C transform on the remaining dimension
    framework::Tensor c2c_out;
    if (axes.size() > 1) {
      c2c_out.mutable_data<To>(out->dims(), ctx.GetPlace());
      std::vector<int64_t> remain_dim(axes.begin(), axes.end() - 1);
      FFTC2CFunctor<platform::CUDADeviceContext, To, To> fft_c2c_func;
      fft_c2c_func(ctx, r2c_out, &c2c_out, remain_dim, FFTNormMode::none,
                   forward);
    }

    const auto in_sizes = framework::vectorize(X->dims());
    framework::Tensor* norm_tensor = axes.size() > 1 ? &c2c_out : r2c_out;
    exec_normalization<platform::CUDADeviceContext, To>(
        ctx, norm_tensor, out, normalization, in_sizes, axes);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    fft_c2c, ops::FFTC2CKernel<paddle::platform::CUDADeviceContext, float>,
    ops::FFTC2CKernel<paddle::platform::CUDADeviceContext, double>);

REGISTER_OP_CUDA_KERNEL(
    fft_c2c_grad,
    ops::FFTC2CGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::FFTC2CGradKernel<paddle::platform::CUDADeviceContext, double>);

REGISTER_OP_CUDA_KERNEL(
    fft_c2r, ops::FFTC2RKernel<paddle::platform::CUDADeviceContext, float>,
    ops::FFTC2RKernel<paddle::platform::CUDADeviceContext, double>);

REGISTER_OP_CUDA_KERNEL(
    fft_c2r_grad,
    ops::FFTC2RGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::FFTC2RGradKernel<paddle::platform::CUDADeviceContext, double>);

REGISTER_OP_CUDA_KERNEL(
    fft_r2c, ops::FFTR2CKernel<paddle::platform::CUDADeviceContext, float>,
    ops::FFTR2CKernel<paddle::platform::CUDADeviceContext, double>);

REGISTER_OP_CUDA_KERNEL(
    fft_r2c_grad,
    ops::FFTR2CGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::FFTR2CGradKernel<paddle::platform::CUDADeviceContext, double>);
