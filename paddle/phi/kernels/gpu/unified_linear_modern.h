/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

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

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

// Forward declarations for PaddlePaddle types
namespace phi {
class GPUContext;
class DenseTensor;
enum class DataType : int32_t;
enum class ActivationType : int32_t;
}  // namespace phi

// CUDA forward declarations
using cublasLtHandle_t = void*;
using cublasLtMatmulDesc_t = void*;
using cublasLtMatrixLayout_t = void*;
using cublasLtEpilogueDesc_t = void*;
using cublasLtStatus_t = int;
using cublasHandle_t = void*;
using cublasStatus_t = int;
using cudaStream_t = void*;

// CUDA constants
constexpr int CUBLAS_STATUS_SUCCESS = 0;
constexpr int CUBLAS_STATUS_NOT_SUPPORTED = 1;
constexpr int CUBLAS_STATUS_NOT_INITIALIZED = 2;
constexpr int CUDA_R_32F = 0;
constexpr int CUDA_R_64F = 1;
constexpr int CUBLAS_COMPUTE_32F_FAST_16F = 2;
constexpr int CUBLAS_COMPUTE_64F = 3;
constexpr int CUBLAS_OP_N = 0;
constexpr int CUBLAS_OP_T = 1;
constexpr int CUBLASLT_EPILOGUE_DEFAULT = 0;
constexpr int CUBLASLT_EPILOGUE_RELU = 1;
constexpr int CUBLASLT_EPILOGUE_GELU = 2;

// CUDA function forward declarations
extern "C" {
cublasLtStatus_t cublasLtCreate(cublasLtHandle_t* handle);
cublasLtStatus_t cublasLtDestroy(cublasLtHandle_t handle);
cublasLtStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t* desc,
                                          int compute_type,
                                          int scale_type);
cublasLtStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t desc);
cublasLtStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t* desc,
                                            int type,
                                            int64_t rows,
                                            int64_t cols,
                                            int64_t ld);
cublasLtStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t desc);
cublasLtStatus_t cublasLtMatmul(cublasLtHandle_t lightHandle,
                                cublasLtMatmulDesc_t computeDesc,
                                const void* alpha,
                                const void* A,
                                cublasLtMatrixLayout_t Adesc,
                                const void* B,
                                cublasLtMatrixLayout_t Bdesc,
                                const void* beta,
                                const void* C,
                                cublasLtMatrixLayout_t Cdesc,
                                void* D,
                                cublasLtMatrixLayout_t Ddesc,
                                const void* algo,
                                void* workspace,
                                size_t workspaceSizeInBytes,
                                cudaStream_t stream);

cublasStatus_t cublasCreate_v2(cublasHandle_t* handle);
cublasStatus_t cublasDestroy_v2(cublasHandle_t handle);
cublasStatus_t cublasSgemm_v2(cublasHandle_t handle,
                              int transa,
                              int transb,
                              int m,
                              int n,
                              int k,
                              const float* alpha,
                              const float* A,
                              int lda,
                              const float* B,
                              int ldb,
                              const float* beta,
                              float* C,
                              int ldc);
cublasStatus_t cublasDgemm_v2(cublasHandle_t handle,
                              int transa,
                              int transb,
                              int m,
                              int n,
                              int k,
                              const double* alpha,
                              const double* A,
                              int lda,
                              const double* B,
                              int ldb,
                              const double* beta,
                              double* C,
                              int ldc);
}

// Modern type traits for C++17 compatibility
template <typename T>
using is_arithmetic_v = typename std::is_arithmetic<T>::type;

template <typename T, typename U>
using is_same_v = typename std::is_same<T, U>::type;

// C++17 compatible concepts
template <typename T>
constexpr bool NumericType =
    std::is_arithmetic<T>::value && !std::is_same<T, bool>::value;

template <typename T>
constexpr bool SupportedType =
    std::is_same<T, float>::value || std::is_same<T, double>::value;

namespace phi {

// Modern C++20 concepts for zero-overhead abstractions
template <typename T>
concept NumericType = std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

template <typename T>
concept SupportedType = std::same_as<T, float> || std::same_as<T, double> ||
    std::same_as<T, phi::float16> || std::same_as<T, phi::bfloat16>;

// Modern activation types with compile-time optimization
enum class ActivationType { NONE, RELU, GELU, TANH, SIGMOID };

// Zero-overhead error handling with C++17 compatibility
template <typename T>
struct Result {
  T value;
  std::string error;
  bool has_value;

  explicit constexpr Result(T val)
      : value{std::move(val)}, error{}, has_value{true} {}
  explicit constexpr Result(std::string err)
      : value{}, error{std::move(err)}, has_value{false} {}

  constexpr bool has_value() const { return has_value; }
  constexpr T& value() { return value; }
  constexpr const T& value() const { return value; }
  constexpr const std::string& error() const { return error; }
};

// Modern RAII resource wrapper with lambda optimization
template <typename Handle, typename Deleter>
class ResourceWrapper {
 public:
  using handle_type = Handle;
  using deleter_type = Deleter;

  explicit ResourceWrapper(Handle handle = nullptr,
                           Deleter deleter = {}) noexcept
      : handle_{handle}, deleter_{std::move(deleter)} {}

  ResourceWrapper(ResourceWrapper&& other) noexcept
      : handle_{exchange(other.handle_, nullptr)},
        deleter_{std::move(other.deleter_)} {}

  ResourceWrapper& operator=(ResourceWrapper&& other) noexcept {
    if (this != &other) {
      reset();
      handle_ = exchange(other.handle_, nullptr);
      deleter_ = std::move(other.deleter_);
    }
    return *this;
  }

  ~ResourceWrapper() noexcept { reset(); }

  Handle get() const noexcept { return handle_; }
  explicit operator bool() const noexcept { return handle_ != nullptr; }

  Handle release() noexcept { return exchange(handle_, nullptr); }

  void reset(Handle handle = nullptr) noexcept {
    if (handle_ != handle) {
      if (handle_) deleter_(handle_);
      handle_ = handle;
    }
  }

  void swap(ResourceWrapper& other) noexcept {
    std::swap(handle_, other.handle_);
    std::swap(deleter_, other.deleter_);
  }

 private:
  Handle handle_;
  Deleter deleter_;

  template <typename T>
  static T exchange(T& obj, T new_value) noexcept {  // NOLINT
    T old_value = std::move(obj);
    obj = std::move(new_value);
    return old_value;
  }
};

// Modern unified linear descriptor with compile-time optimization
struct UnifiedLinearDescriptor {
  const DenseTensor* input = nullptr;
  const DenseTensor* weight = nullptr;
  const DenseTensor* bias = nullptr;
  DenseTensor* output = nullptr;

  bool transpose_input = false;
  bool transpose_weight = false;
  bool transpose_output = false;

  ActivationType activation = ActivationType::NONE;

  float alpha = 1.0f;
  float beta = 0.0f;

  const DenseTensor* input_scale = nullptr;
  const DenseTensor* weight_scale = nullptr;
  const DenseTensor* output_scale = nullptr;

  DataType atype = DataType::FLOAT32;
  DataType btype = DataType::FLOAT32;
  DataType ctype = DataType::FLOAT32;

  // Modern validation with lambda optimization
  [[nodiscard]] bool is_valid() const noexcept {
    auto check_tensor = [](const DenseTensor* tensor) {
      return tensor && tensor->initialized();
    };

    return check_tensor(input) && check_tensor(weight) && output &&
           check_tensor(output);
  }

  // Compile-time dimension calculation
  [[nodiscard]] std::array<int, 3> get_matrix_dimensions() const noexcept {
    if (!is_valid()) return {0, 0, 0};

    const auto& input_dims = input->dims();
    const auto& weight_dims = weight->dims();

    // Lambda for safe dimension extraction
    auto get_dim = [](const DDim& dims, size_t idx) -> int {
      return idx < dims.size() ? dims[idx] : 1;
    };

    const int m =
        get_dim(input_dims, input_dims.size() - (transpose_input ? 2 : 1));
    const int n =
        get_dim(weight_dims, weight_dims.size() - (transpose_weight ? 1 : 2));
    const int k =
        get_dim(input_dims, input_dims.size() - (transpose_input ? 1 : 2));

    return {m, n, k};
  }

  // Modern precision checking
  [[nodiscard]] bool is_narrow_precision() const noexcept {
    return atype == DataType::FLOAT16 || atype == DataType::BFLOAT16;
  }

  [[nodiscard]] bool is_mixed_precision() const noexcept {
    return (atype == DataType::FLOAT16 && ctype == DataType::FLOAT32) ||
           (atype == DataType::BFLOAT16 && ctype == DataType::FLOAT32);
  }
};

// Modern execution context with zero-overhead abstractions
struct ExecutionContext {
  const GPUContext& device_context;
  cudaStream_t stream = nullptr;

  explicit ExecutionContext(const GPUContext& ctx) : device_context(ctx) {
    stream = device_context.stream();
  }

  // Modern synchronization with lambda optimization
  void synchronize() const {
    auto check_stream = [this]() -> Result<void> {
      if (!stream) {
        return std::unexpected("Invalid CUDA stream");
      }
      return {};
    };

    auto stream_result = check_stream();
    if (!stream_result) {
      throw std::runtime_error(stream_result.error());
    }

    // Lambda for safe synchronization
    auto safe_synchronize = [](cudaStream_t s) {
      const auto status = cudaStreamSynchronize(s);
      if (status != cudaSuccess) {
        throw std::runtime_error("Stream synchronization failed: " +
                                 std::string(cudaGetErrorString(status)));
      }
    };
    safe_synchronize(stream);
  }
};

// Modern utility functions with compile-time optimization
namespace utils {

// Zero-overhead data type conversion with lambda optimization
template <typename From, typename To>
[[nodiscard]] constexpr auto convert_data_type() noexcept {
  if constexpr (std::is_same_v<From, float>) {
    return CUDA_R_32F;
  } else if constexpr (std::is_same_v<From, double>) {
    return CUDA_R_64F;
  } else if constexpr (std::is_same_v<From, phi::float16>) {
    return CUDA_R_16F;
  } else if constexpr (std::is_same_v<From, phi::bfloat16>) {
    return CUDA_R_16BF;
  } else {
    static_assert(sizeof(From) != 0, "Unsupported data type");
    return CUDA_R_32F;
  }
}

// Modern compute type selection with compile-time optimization
[[nodiscard]] constexpr auto select_compute_type(DataType atype,
                                                 DataType btype,
                                                 DataType ctype) noexcept {
  if constexpr (CUDA_VERSION >= 11000) {
    if (atype == DataType::FLOAT16 && btype == DataType::FLOAT16 &&
        ctype == DataType::FLOAT32) {
      return CUBLAS_COMPUTE_32F_FAST_16F;
    } else if (atype == DataType::FLOAT16 && btype == DataType::FLOAT16 &&
               ctype == DataType::FLOAT16) {
      return CUBLAS_COMPUTE_16F;
    } else if (atype == DataType::BFLOAT16 && btype == DataType::BFLOAT16) {
      return CUBLAS_COMPUTE_32F_FAST_16BF;
    }
  }
  return CUBLAS_COMPUTE_32F;
}

// Modern dimension calculation with lambda optimization
[[nodiscard]] constexpr auto calculate_matrix_layout(const DDim& dims,
                                                     bool transpose) noexcept {
  struct MatrixLayout {
    int rows;
    int cols;
    int leading_dim;
  };

  // Lambda for safe dimension extraction
  auto get_dim = [](const DDim& dims, size_t idx) -> int {
    return idx < dims.size() ? dims[idx] : 1;
  };

  const int dim0 = get_dim(dims, dims.size() - 2);
  const int dim1 = get_dim(dims, dims.size() - 1);

  return transpose ? MatrixLayout{dim1, dim0, dim1}
                   : MatrixLayout{dim0, dim1, dim0};
}

// Modern error checking with lambda optimization
template <typename Status>
[[nodiscard]] Result<void> check_status(Status status,
                                        std::string_view operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    return std::unexpected(std::string(operation) +
                           " failed with status: " + std::to_string(status));
  }
  return {};
}

}  // namespace utils

// Modern factory functions with compile-time optimization
template <SupportedType T>
[[nodiscard]] auto create_unified_linear_kernel(const GPUContext& dev_ctx) {
  // Forward declaration - implementation would be in the .cu file
  class UnifiedLinearKernelModern;
  return UnifiedLinearKernelModern{dev_ctx};
}

// Modern type traits with zero-overhead abstractions
template <typename T>
struct TypeTraits {
  static constexpr bool is_supported = SupportedType<T>;
  static constexpr DataType data_type = []() {
    if constexpr (std::is_same_v<T, float>)
      return DataType::FLOAT32;
    else if constexpr (std::is_same_v<T, double>)
      return DataType::FLOAT64;
    else if constexpr (std::is_same_v<T, phi::float16>)
      return DataType::FLOAT16;
    else if constexpr (std::is_same_v<T, phi::bfloat16>)
      return DataType::BFLOAT16;
    else
      return DataType::FLOAT32;
  }();
};

}  // namespace phi
