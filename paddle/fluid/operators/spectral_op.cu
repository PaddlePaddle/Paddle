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

#include <cufft.h>
#include <cufftXt.h>

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
#include "paddle/fluid/operators/spectral_op.h"
#include "paddle/fluid/operators/transpose_op.h"
#include "paddle/fluid/platform/dynload/cufft.h"

namespace paddle {
namespace operators {

namespace {

using ScalarType = framework::proto::VarType::Type;
const int64_t kMaxCUFFTNdim = 3;
const int64_t kMaxDataNdim = kMaxCUFFTNdim + 1;

static inline std::string get_cufft_error_info(cufftResult error) {
  switch (error) {
    case CUFFT_SUCCESS:
      return "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN:
      return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED:
      return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE:
      return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE:
      return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR:
      return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED:
      return "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED:
      return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE:
      return "CUFFT_INVALID_SIZE";
    case CUFFT_UNALIGNED_DATA:
      return "CUFFT_UNALIGNED_DATA";
    case CUFFT_INCOMPLETE_PARAMETER_LIST:
      return "CUFFT_INCOMPLETE_PARAMETER_LIST";
    case CUFFT_INVALID_DEVICE:
      return "CUFFT_INVALID_DEVICE";
    case CUFFT_PARSE_ERROR:
      return "CUFFT_PARSE_ERROR";
    case CUFFT_NO_WORKSPACE:
      return "CUFFT_NO_WORKSPACE";
    case CUFFT_NOT_IMPLEMENTED:
      return "CUFFT_NOT_IMPLEMENTED";
#ifndef __HIPCC__
    case CUFFT_LICENSE_ERROR:
      return "CUFFT_LICENSE_ERROR";
#endif
    case CUFFT_NOT_SUPPORTED:
      return "CUFFT_NOT_SUPPORTED";
    default:
      std::ostringstream ss;
      ss << "unknown error " << error;
      return ss.str();
  }
}

static inline void CUFFT_CHECK(cufftResult error) {
  PADDLE_ENFORCE_CUDA_SUCCESS(error);
}

// This struct is used to easily compute hashes of the
// parameters. It will be the **key** to the plan cache.
struct PlanKey {
  // between 1 and kMaxCUFFTNdim, i.e., 1 <= signal_ndim <= 3
  int64_t signal_ndim_;
  // These include additional batch dimension as well.
  int64_t sizes_[kMaxDataNdim];
  int64_t input_shape_[kMaxDataNdim];
  int64_t output_shape_[kMaxDataNdim];
  FFTTransformType fft_type_;
  ScalarType value_type_;

  PlanKey() = default;

  PlanKey(const std::vector<int64_t>& in_shape,
          const std::vector<int64_t>& out_shape,
          const std::vector<int64_t>& signal_size, FFTTransformType fft_type,
          ScalarType value_type) {
    // Padding bits must be zeroed for hashing
    memset(this, 0, sizeof(*this));
    signal_ndim_ = signal_size.size() - 1;
    fft_type_ = fft_type;
    value_type_ = value_type;

    std::copy(signal_size.cbegin(), signal_size.cend(), sizes_);
    std::copy(in_shape.cbegin(), in_shape.cend(), input_shape_);
    std::copy(out_shape.cbegin(), out_shape.cend(), output_shape_);
  }
};

// An RAII encapsulation of cuFFTHandle
class CuFFTHandle {
  ::cufftHandle handle_;

 public:
  CuFFTHandle() { CUFFT_CHECK(platform::dynload::cufftCreate(&handle_)); }

  ::cufftHandle& get() { return handle_; }
  const ::cufftHandle& get() const { return handle_; }

  ~CuFFTHandle() {
// Not using fftDestroy() for rocFFT to work around double freeing of handles
#ifndef __HIPCC__
    CUFFT_CHECK(platform::dynload::cufftDestroy(handle_));
#endif
  }
};

#ifdef __HIPCC__
using plan_size_type = int;
#else
using plan_size_type = long long int;  // NOLINT
#endif

// This class contains all the information needed to execute a cuFFT plan:
//   1. the plan
//   2. the workspace size needed
class CuFFTConfig {
 public:
  // Only move semantics is enought for this class. Although we already use
  // unique_ptr for the plan, still remove copy constructor and assignment op so
  // we don't accidentally copy and take perf hit.
  CuFFTConfig(const CuFFTConfig&) = delete;
  CuFFTConfig& operator=(CuFFTConfig const&) = delete;

  explicit CuFFTConfig(const PlanKey& plan_key)
      : CuFFTConfig(
            std::vector<int64_t>(plan_key.sizes_,
                                 plan_key.sizes_ + plan_key.signal_ndim_ + 1),
            plan_key.signal_ndim_, plan_key.fft_type_, plan_key.value_type_) {}

  // sizes are full signal, including batch size and always two-sided
  CuFFTConfig(const std::vector<int64_t>& sizes, const int64_t signal_ndim,
              FFTTransformType fft_type, ScalarType dtype)
      : fft_type_(fft_type), value_type_(dtype) {
    // signal sizes (excluding batch dim)
    std::vector<plan_size_type> signal_sizes(sizes.begin() + 1, sizes.end());

    // input batch size
    const auto batch = static_cast<plan_size_type>(sizes[0]);
    // const int64_t signal_ndim = sizes.size() - 1;
    PADDLE_ENFORCE_EQ(signal_ndim, sizes.size() - 1,
                      platform::errors::InvalidArgument(
                          "The signal_ndim must be equal to sizes.size() - 1,"
                          "But signal_ndim is: [%d], sizes.size() - 1 is: [%d]",
                          signal_ndim, sizes.size() - 1));

#ifdef __HIPCC__
    hipfftType exec_type = [&] {
      if (dtype == framework::proto::VarType::FP32) {
        switch (fft_type) {
          case FFTTransformType::C2C:
            return HIPFFT_C2C;
          case FFTTransformType::R2C:
            return HIPFFT_R2C;
          case FFTTransformType::C2R:
            return HIPFFT_C2R;
        }
      } else if (dtype == framework::proto::VarType::FP64) {
        switch (fft_type) {
          case FFTTransformType::C2C:
            return HIPFFT_Z2Z;
          case FFTTransformType::R2C:
            return HIPFFT_D2Z;
          case FFTTransformType::C2R:
            return HIPFFT_Z2D;
        }
      }
      PADDLE_THROW(platform::errors::InvalidArgument(
          "hipFFT only support transforms of type float32 and float64"));
    }();
#else
    cudaDataType itype, otype, exec_type;
    const auto complex_input = has_complex_input(fft_type);
    const auto complex_output = has_complex_output(fft_type);
    if (dtype == framework::proto::VarType::FP32) {
      itype = complex_input ? CUDA_C_32F : CUDA_R_32F;
      otype = complex_output ? CUDA_C_32F : CUDA_R_32F;
      exec_type = CUDA_C_32F;
    } else if (dtype == framework::proto::VarType::FP64) {
      itype = complex_input ? CUDA_C_64F : CUDA_R_64F;
      otype = complex_output ? CUDA_C_64F : CUDA_R_64F;
      exec_type = CUDA_C_64F;
    } else if (dtype == framework::proto::VarType::FP16) {
      itype = complex_input ? CUDA_C_16F : CUDA_R_16F;
      otype = complex_output ? CUDA_C_16F : CUDA_R_16F;
      exec_type = CUDA_C_16F;
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "cuFFT only support transforms of type float16, float32 and "
          "float64"));
    }
#endif

    // disable auto allocation of workspace to use allocator from the framework
    CUFFT_CHECK(platform::dynload::cufftSetAutoAllocation(
        plan(), /* autoAllocate */ 0));

    size_t ws_size_t;

// make plan
#ifdef __HIPCC__
    CUFFT_CHECK(hipfftMakePlanMany(
        plan(), signal_ndim, signal_sizes.data(),
        /* inembed */ nullptr, /* base_istride */ 1, /* idist */ 1,
        /* onembed */ nullptr, /* base_ostride */ 1, /* odist */ 1, exec_type,
        batch, &ws_size_t));
#else

    CUFFT_CHECK(platform::dynload::cufftXtMakePlanMany(
        plan(), signal_ndim, signal_sizes.data(),
        /* inembed */ nullptr, /* base_istride */ 1, /* idist */ 1, itype,
        /* onembed */ nullptr, /* base_ostride */ 1, /* odist */ 1, otype,
        batch, &ws_size_t, exec_type));
#endif

    ws_size = ws_size_t;
  }

  const cufftHandle& plan() const { return plan_ptr.get(); }

  FFTTransformType transform_type() const { return fft_type_; }
  ScalarType data_type() const { return value_type_; }
  size_t workspace_size() const { return ws_size; }

 private:
  CuFFTHandle plan_ptr;
  size_t ws_size;
  FFTTransformType fft_type_;
  ScalarType value_type_;
};

// Execute a pre-planned transform
static void exec_cufft_plan(const CuFFTConfig& config, void* in_data,
                            void* out_data, bool forward) {
  auto& plan = config.plan();
#ifdef __HIPCC__
  auto value_type = config.data_type();
  if (value_type == framework::proto::VarType::FP32) {
    switch (config.transform_type()) {
      case FFTTransformType::C2C: {
        CUFFT_CHECK(hipfftExecC2C(plan, static_cast<hipfftComplex*>(in_data),
                                  static_cast<hipfftComplex*>(out_data),
                                  forward ? HIPFFT_FORWARD : HIPFFT_BACKWARD));
        return;
      }
      case FFTTransformType::R2C: {
        CUFFT_CHECK(hipfftExecR2C(plan, static_cast<hipfftReal*>(in_data),
                                  static_cast<hipfftComplex*>(out_data)));
        return;
      }
      case FFTTransformType::C2R: {
        CUFFT_CHECK(hipfftExecC2R(plan, static_cast<hipfftComplex*>(in_data),
                                  static_cast<hipfftReal*>(out_data)));
        return;
      }
    }
  } else if (value_type == framework::proto::VarType::FP64) {
    switch (config.transform_type()) {
      case FFTTransformType::C2C: {
        CUFFT_CHECK(hipfftExecZ2Z(plan,
                                  static_cast<hipfftDoubleComplex*>(in_data),
                                  static_cast<hipfftDoubleComplex*>(out_data),
                                  forward ? HIPFFT_FORWARD : HIPFFT_BACKWARD));
        return;
      }
      case FFTTransformType::R2C: {
        CUFFT_CHECK(hipfftExecD2Z(plan, static_cast<hipfftDoubleReal*>(in_data),
                                  static_cast<hipfftDoubleComplex*>(out_data)));
        return;
      }
      case FFTTransformType::C2R: {
        CUFFT_CHECK(hipfftExecZ2D(plan,
                                  static_cast<hipfftDoubleComplex*>(in_data),
                                  static_cast<hipfftDoubleReal*>(out_data)));
        return;
      }
    }
  }
  PADDLE_THROW(platform::errors::InvalidArgument(
      "hipFFT only support transforms of type float32 and float64"));
#else
  CUFFT_CHECK(platform::dynload::cufftXtExec(
      plan, in_data, out_data, forward ? CUFFT_FORWARD : CUFFT_INVERSE));
#endif
}

// Execute a general unnormalized fft operation (can be c2c, onesided r2c or
// onesided c2r)
template <typename DeviceContext, typename Ti, typename To>
void exec_fft(const DeviceContext& ctx, const Tensor* X, Tensor* out,
              const std::vector<int64_t>& dim, bool forward) {
  const auto x_dims = framework::vectorize(X->dims());
  const auto out_dims = framework::vectorize(out->dims());
  const int64_t ndim = static_cast<int64_t>(X->dims().size());
  const int64_t signal_ndim = static_cast<int64_t>(dim.size());
  const int64_t batch_dims = ndim - signal_ndim;
  auto tensor_place = ctx.GetPlace();

  // Transpose batch dimensions first, then with transforming dims
  std::vector<int> dim_permute(ndim);
  std::vector<int> reverse_dim_permute(ndim);
  std::vector<int64_t> trans_dims(ndim);
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

  for (size_t i = 0; i < ndim; i++) {
    trans_dims[i] = x_dims[dim_permute[i]];  // shape of input transpose
    reverse_dim_permute[dim_permute[i]] =
        static_cast<int>(i);  // reverse of dim permute
  }
  framework::Tensor input;
  input.Resize(framework::make_ddim(trans_dims));
  input.mutable_data<Ti>(tensor_place);
  /*
  auto in_ret = TransposeSimple<Ti>::run(ctx, *X, dim_permute, input);
  if (!in_ret) {
    TransCompute<DeviceContext, Ti>(ndim, ctx, *X, input, dim_permute);
  }
  */
  TransCompute<DeviceContext, Ti>(ndim, ctx, *X, &input, dim_permute);

  // Reshape batch dimensions into a single dimension
  std::vector<int64_t> batched_sizes(signal_ndim + 1);
  auto batch_size =
      std::accumulate(trans_dims.begin(), trans_dims.begin() + batch_dims,
                      static_cast<int>(1), std::multiplies<int>());
  batched_sizes[0] = batch_size;
  std::copy(trans_dims.begin() + batch_dims, trans_dims.end(),
            batched_sizes.begin() + 1);
  input.Resize(framework::make_ddim(batched_sizes));

  // Check the shape of transforming dims with input and output
  std::vector<int64_t> signal_size(signal_ndim + 1);
  signal_size[0] = batch_size;
  for (int64_t i = 0; i < signal_ndim; ++i) {
    auto in_size = input.dims()[i + 1];
    auto out_size = out_dims[dim[i]];
    signal_size[i + 1] = std::max(in_size, out_size);
    PADDLE_ENFORCE_EQ(
        (in_size == signal_size[i + 1] ||
         in_size == (signal_size[i + 1] / 2) + 1),
        true,
        platform::errors::InvalidArgument(
            "The dimension[%d] of Input size: [%d] must be equal or half to "
            "The dimension[%d] of Output size: [%d]",
            dim[i], in_size, dim[i], out_size));
    PADDLE_ENFORCE_EQ(
        (out_size == signal_size[i + 1] ||
         out_size == (signal_size[i + 1] / 2) + 1),
        true,
        platform::errors::InvalidArgument(
            "The dimension[%d] of Output size: [%d] must be equal or half to "
            "The dimension[%d] of Input size: [%d]",
            dim[i], out_size, dim[i], in_size));
  }

  std::vector<int64_t> reshape_out_sizes(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    reshape_out_sizes[i] = out_dims[dim_permute[i]];
  }
  std::vector<int64_t> batched_out_sizes(batched_sizes.begin(),
                                         batched_sizes.end());
  for (size_t i = 0; i < dim.size(); ++i) {
    batched_out_sizes[i + 1] = out_dims[dim[i]];
  }

  // output
  framework::Tensor output;
  output.Resize(framework::make_ddim(batched_out_sizes));
  output.mutable_data<To>(tensor_place);

  // Create the transform plan (either from cache or locally)
  const auto value_type = framework::IsComplexType(input.type())
                              ? framework::ToRealType(input.type())
                              : input.type();
  auto fft_type = GetFFTTransformType(input.type(), output.type());

  PlanKey Key(framework::vectorize(input.dims()),
              framework::vectorize(output.dims()), signal_size, fft_type,
              value_type);
  CuFFTConfig uncached_plan(Key);
  CuFFTConfig* config = &uncached_plan;
  auto& plan = config->plan();

  // prepare cufft for execution
  CUFFT_CHECK(platform::dynload::cufftSetStream(plan, ctx.stream()));
  framework::Tensor workspace_tensor;
  workspace_tensor.mutable_data<To>(tensor_place, config->workspace_size());
  CUFFT_CHECK(
      platform::dynload::cufftSetWorkArea(plan, workspace_tensor.data<To>()));

  // execute transform plan
  if (fft_type == FFTTransformType::C2R && forward) {
    forward = false;
    framework::Tensor input_conj(input.type());
    input_conj.mutable_data<Ti>(input.dims(), ctx.GetPlace());
    platform::ForRange<DeviceContext> for_range(ctx, input.numel());
    math::ConjFunctor<Ti> functor(input.data<Ti>(), input.numel(),
                                  input_conj.data<Ti>());
    for_range(functor);
    exec_cufft_plan(*config, input_conj.data<void>(), output.data<void>(),
                    forward);
  } else if (fft_type == FFTTransformType::R2C && !forward) {
    forward = true;
    framework::Tensor out_conj(output.type());
    out_conj.mutable_data<To>(output.dims(), ctx.GetPlace());
    exec_cufft_plan(*config, input.data<void>(), out_conj.data<void>(),
                    forward);

    platform::ForRange<DeviceContext> for_range(ctx, output.numel());
    math::ConjFunctor<To> functor(out_conj.data<To>(), output.numel(),
                                  output.data<To>());
    for_range(functor);
  } else {
    exec_cufft_plan(*config, input.data<void>(), output.data<void>(), forward);
  }

  // Inverting output by reshape and transpose to original batch and dimension
  output.Resize(framework::make_ddim(reshape_out_sizes));
  out->Resize(framework::make_ddim(out_dims));
  TransCompute<DeviceContext, To>(ndim, ctx, output, out, reverse_dim_permute);
}

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
}  // anonymous namespace

// Use the optimized path to perform single R2C or C2R if transformation dim is
// supported by cuFFT
bool use_optimized_cufft_path(const std::vector<int64_t>& axes) {
  // For performance reason, when axes starts with (0, 1), do not use the
  // optimized path.
  if (axes.size() > kMaxCUFFTNdim ||
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
          std::min(static_cast<size_t>(kMaxCUFFTNdim), working_axes.size());
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

    if (use_optimized_cufft_path(axes)) {
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
