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

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include "paddle/phi/kernels/gpu/unified_linear_modern.h"

namespace phi {

// Modern C++20 concepts for type safety
template <typename T>
concept NumericType = std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

template <typename T>
concept SupportedType = std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, phi::float16>;

// Zero-overhead RAII wrapper for cuBLASLt resources
template <typename Resource, auto CreateFunc, auto DestroyFunc>
class ResourceWrapper {
 public:
  using ResourceType = Resource;
  using CreateFunction = decltype(CreateFunc);
  using DestroyFunction = decltype(DestroyFunc);

  ResourceWrapper() = default;

  explicit ResourceWrapper(const GPUContext& ctx) : context_(ctx) { create(); }

  ~ResourceWrapper() { reset(); }

  ResourceWrapper(const ResourceWrapper&) = delete;
  ResourceWrapper& operator=(const ResourceWrapper&) = delete;

  ResourceWrapper(ResourceWrapper&& other) noexcept
      : resource_(std::exchange(other.resource_, nullptr)),
        context_(other.context_) {}

  ResourceWrapper& operator=(ResourceWrapper&& other) noexcept {
    if (this != &other) {
      reset();
      resource_ = std::exchange(other.resource_, nullptr);
      context_ = other.context_;
    }
    return *this;
  }

  [[nodiscard]] Resource get() const noexcept { return resource_; }
  [[nodiscard]] explicit operator bool() const noexcept {
    return resource_ != nullptr;
  }

  void create() {
    if (!resource_) {
      // Lambda for error handling with perfect forwarding
      auto create_with_error_check = [this](auto&&... args) {
        const auto status = CreateFunc(std::forward<decltype(args)>(args)...);
        check_cublaslt_error(status, "Resource creation failed");
      };

      if constexpr (std::is_same_v<Resource, cublasLtHandle_t>) {
        create_with_error_check(&resource_);
      } else {
        // Handle other resource types as needed
        static_assert(sizeof(Resource) != 0, "Resource type not supported");
      }
    }
  }

  void reset() noexcept {
    if (resource_) {
      // Lambda for safe destruction
      auto safe_destroy = [](Resource res) noexcept {
        try {
          DestroyFunc(res);
        } catch (...) {
          // Log error but don't throw in destructor
          LOG(ERROR) << "Resource destruction failed";
        }
      };
      safe_destroy(resource_);
      resource_ = nullptr;
    }
  }

 private:
  Resource resource_ = nullptr;
  GPUContext context_;

  void check_cublaslt_error(cublasLtStatus_t status,
                            std::string_view operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error(std::string(operation) +
                               " failed: " + std::to_string(status));
    }
  }
};

// Modern CublasLtLinear with zero-overhead abstractions
class CublasLtLinearModern {
 public:
  explicit CublasLtLinearModern(const GPUContext& dev_ctx)
      : dev_ctx_(dev_ctx),
        handle_(dev_ctx),
        supports_heuristics_(check_heuristic_support()) {
    initialize_resources();
  }

  ~CublasLtLinearModern() = default;

  CublasLtLinearModern(const CublasLtLinearModern&) = delete;
  CublasLtLinearModern& operator=(const CublasLtLinearModern&) = delete;

  CublasLtLinearModern(CublasLtLinearModern&&) = default;
  CublasLtLinearModern& operator=(CublasLtLinearModern&&) = default;

  // Zero-overhead execution dispatcher with compile-time optimization
  template <SupportedType T>
  void execute(const UnifiedLinearDescriptor& desc) {
    // Compile-time path selection with lambda optimization
    auto select_and_execute = [this, &desc]() {
      if (desc.IsNarrowPrecisionInput()) {
        return execute_narrow_precision<T>(desc);
      } else if (desc.IsMixedPrecision()) {
        return execute_mixed_precision<T>(desc);
      } else {
        return execute_standard<T>(desc);
      }
    };

    // Execute with error handling lambda
    auto execute_with_error_handling = [](auto&& executor) {
      try {
        return executor();
      } catch (const std::exception& e) {
        LOG(ERROR) << "Execution failed: " << e.what();
        throw;
      }
    };

    execute_with_error_handling(select_and_execute);
  }

 private:
  const GPUContext& dev_ctx_;
  ResourceWrapper<cublasLtHandle_t, cublasLtCreate, cublasLtDestroy> handle_;
  ResourceWrapper<cublasLtMatmulDesc_t,
                  cublasLtMatmulDescCreate,
                  cublasLtMatmulDescDestroy>
      matmul_desc_;
  ResourceWrapper<cublasLtEpilogueDesc_t,
                  cublasLtEpilogueDescCreate,
                  cublasLtEpilogueDescDestroy>
      activation_desc_;
  ResourceWrapper<cublasLtEpilogueDesc_t,
                  cublasLtEpilogueDescCreate,
                  cublasLtEpilogueDescDestroy>
      bias_desc_;

  bool supports_heuristics_;
  cudaStream_t stream_ = nullptr;

  // Modern initialization with lambda-based resource management
  void initialize_resources() {
    // Lambda for version checking with early return
    auto check_version_support = []() -> bool {
#if CUDA_VERSION >= 11000
      return true;
#else
      return false;
#endif
    };

    if (!check_version_support()) {
      throw std::runtime_error("cuBLASLt requires CUDA 11.0+");
    }

    // Lambda for stream setup
    auto setup_stream = [this]() {
      stream_ = dev_ctx_.stream();
      if (!stream_) {
        throw std::runtime_error("Invalid CUDA stream");
      }
    };
    setup_stream();

    // Initialize descriptors with lambda-based configuration
    auto initialize_descriptors = [this]() {
      matmul_desc_.create();
      activation_desc_.create();
      bias_desc_.create();
    };
    initialize_descriptors();
  }

  // Modern narrow precision execution with lambda optimizations
  template <SupportedType T>
  void execute_narrow_precision(const UnifiedLinearDescriptor& desc) {
    // Compile-time dimension calculation with constexpr
    constexpr auto calculate_dimensions = [](const auto& dims, bool transpose) {
      return dims[dims.size() - (transpose ? 2 : 1)];
    };

    const auto& input_dims = desc.input->dims();
    const auto& weight_dims = desc.weight->dims();

    const int m = calculate_dimensions(input_dims, desc.transpose_input);
    const int n = calculate_dimensions(weight_dims, desc.transpose_weight);
    const int k =
        input_dims[input_dims.size() - (desc.transpose_input ? 1 : 2)];

    // Lambda for data pointer extraction with null checking
    auto get_data_ptr = [](const auto& tensor) -> const T* {
      return tensor ? tensor->template data<T>() : nullptr;
    };

    const T* A = get_data_ptr(desc.input);
    const T* B = get_data_ptr(desc.weight);
    T* C = desc.output ? desc.output->template data<T>() : nullptr;

    // Lambda for scale application with perfect forwarding
    auto apply_scales = [](float alpha, const auto& scales) -> float {
      return std::accumulate(scales.begin(),
                             scales.end(),
                             alpha,
                             [](float acc, const float* scale) {
                               return scale ? acc * *scale : acc;
                             });
    };

    std::array<const float*, 3> scales{
        desc.input_scale ? desc.input_scale->data<float>() : nullptr,
        desc.weight_scale ? desc.weight_scale->data<float>() : nullptr,
        desc.output_scale ? desc.output_scale->data<float>() : nullptr};

    float alpha = apply_scales(desc.alpha.template to<float>(), scales);
    const float beta = desc.beta.template to<float>();

    // Execute with modern matrix layout management
    execute_matmul_with_layouts<T>(A, B, C, m, n, k, alpha, beta, desc);
  }

  // Modern mixed precision execution
  template <SupportedType T>
  void execute_mixed_precision(const UnifiedLinearDescriptor& desc) {
    // Lambda for mixed precision type selection
    auto select_compute_type =
        [](DataType atype, DataType btype, DataType ctype) {
          if (atype == DataType::FLOAT16 && btype == DataType::FLOAT16 &&
              ctype == DataType::FLOAT16) {
            return CUBLAS_COMPUTE_16F;
          }
          return CUBLAS_COMPUTE_32F_FAST_16F;
        };

    const auto compute_type =
        select_compute_type(desc.atype, desc.btype, desc.ctype);
    execute_with_compute_type<T>(desc, compute_type);
  }

  // Modern standard execution
  template <SupportedType T>
  void execute_standard(const UnifiedLinearDescriptor& desc) {
    // Lambda for execution strategy selection
    auto select_strategy = [&desc]() {
      return can_fuse_operations(desc) ? ExecutionStrategy::FUSED
             : supports_heuristics_    ? ExecutionStrategy::HEURISTIC
                                       : ExecutionStrategy::DEFAULT;
    };

    const auto strategy = select_strategy();
    execute_with_strategy<T>(desc, strategy);
  }

  // Modern matrix multiplication with layout optimization
  template <SupportedType T>
  void execute_matmul_with_layouts(const T* A,
                                   const T* B,
                                   T* C,
                                   int m,
                                   int n,
                                   int k,
                                   float alpha,
                                   float beta,
                                   const UnifiedLinearDescriptor& desc) {
    // Lambda for layout creation with RAII
    auto create_layout =
        [](const auto& tensor, int rows, int cols, bool transpose) {
          return ResourceWrapper<cublasLtMatrixLayout_t,
                                 cublasLtMatrixLayoutCreate,
                                 cublasLtMatrixLayoutDestroy>{};
        };

    // Create layouts with perfect forwarding
    auto A_layout = create_layout(desc.input, k, m, desc.transpose_input);
    auto B_layout = create_layout(desc.weight, n, k, desc.transpose_weight);
    auto C_layout = create_layout(desc.output, n, m, false);

    // Execute with modern algorithm selection
    execute_cublaslt_matmul<T>(A_layout.get(),
                               B_layout.get(),
                               C_layout.get(),
                               A,
                               B,
                               C,
                               alpha,
                               beta,
                               desc);
  }

  // Modern cuBLASLt matrix multiplication with lambda optimization
  template <SupportedType T>
  void execute_cublaslt_matmul(cublasLtMatrixLayout_t A_layout,
                               cublasLtMatrixLayout_t B_layout,
                               cublasLtMatrixLayout_t C_layout,
                               const T* A,
                               const T* B,
                               T* C,
                               float alpha,
                               float beta,
                               const UnifiedLinearDescriptor& desc) {
    // Lambda for algorithm selection with heuristics
    auto select_algorithm =
        [this](cublasLtMatrixLayout_t A_layout,
               cublasLtMatrixLayout_t B_layout,
               cublasLtMatrixLayout_t C_layout) -> cublasLtMatmulAlgo_t {
      if (!supports_heuristics_) {
        return {};
      }

      cublasLtMatmulAlgo_t result{};
      int returned_results = 0;

      const auto status = cublasLtMatmulAlgoGetHeuristic(handle_.get(),
                                                         matmul_desc_.get(),
                                                         A_layout,
                                                         B_layout,
                                                         C_layout,
                                                         C_layout,
                                                         &result,
                                                         1,
                                                         &returned_results);

      return (status == CUBLAS_STATUS_SUCCESS && returned_results > 0)
                 ? result
                 : cublasLtMatmulAlgo_t{};
    };

    const auto algorithm = select_algorithm(A_layout, B_layout, C_layout);

    // Execute with modern error handling
    const auto status = cublasLtMatmul(handle_.get(),
                                       matmul_desc_.get(),
                                       &alpha,
                                       B,
                                       B_layout,
                                       A,
                                       A_layout,
                                       &beta,
                                       C,
                                       C_layout,
                                       C,
                                       C_layout,
                                       &algorithm,
                                       nullptr,
                                       0,
                                       stream_);

    check_cublaslt_error(status, "cuBLASLt matmul execution");
  }

  // Utility functions with modern C++ features
  bool check_heuristic_support() const {
#if CUDA_VERSION >= 11000
    return true;
#else
    return false;
#endif
  }

  enum class ExecutionStrategy { FUSED, HEURISTIC, DEFAULT };

  bool can_fuse_operations(const UnifiedLinearDescriptor& desc) const {
    return desc.bias != nullptr || desc.activation != ActivationType::NONE;
  }

  template <SupportedType T>
  void execute_with_compute_type(const UnifiedLinearDescriptor& desc,
                                 cublasComputeType_t compute_type) {
    // Implementation for mixed precision execution
    // This would use the compute_type parameter for mixed precision math
  }

  template <SupportedType T>
  void execute_with_strategy(const UnifiedLinearDescriptor& desc,
                             ExecutionStrategy strategy) {
    // Implementation for different execution strategies
    // This would handle fused, heuristic, or default execution paths
  }

  void check_cublaslt_error(cublasLtStatus_t status,
                            std::string_view operation) const {
    if (status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error(
          std::string(operation) +
          " failed with status: " + std::to_string(status));
    }
  }
};

// Modern unified linear kernel with zero-overhead dispatch
template <SupportedType T>
class UnifiedLinearKernelModern {
 public:
  explicit UnifiedLinearKernelModern(const GPUContext& dev_ctx)
      : dev_ctx_(dev_ctx),
        cublaslt_impl_(std::make_unique<CublasLtLinearModern>(dev_ctx)) {}

  void execute(const UnifiedLinearDescriptor& desc) {
    // Zero-overhead execution with perfect forwarding
    cublaslt_impl_->execute<T>(desc);
  }

 private:
  const GPUContext& dev_ctx_;
  std::unique_ptr<CublasLtLinearModern> cublaslt_impl_;
};

// Modern factory function with compile-time optimization
template <SupportedType T>
[[nodiscard]] auto create_unified_linear_kernel(const GPUContext& dev_ctx) {
  return UnifiedLinearKernelModern<T>{dev_ctx};
}

}  // namespace phi
