/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/unified_linear_kernel.h"
#include "paddle/phi/kernels/unified_linear_utils.h"

#ifdef PADDLE_WITH_CUDA
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/kernels/gpu/cublas_linear.h"
#include "paddle/phi/kernels/gpu/cublaslt_linear.h"
#endif

// For convenience, define namespace aliases
namespace utils = phi::unified_linear::utils;
namespace cublaslt_impl = phi::unified_linear::cuda::cublasLt;

namespace phi {
namespace unified_linear {
namespace cuda {

// Backend type selection strategy - aligned with PyTorch
enum class BackendType {
  kCublas,    // Use standard cuBLAS
  kCublasLt,  // Use cuBLASLt for better performance with scale support
  kAuto  // Automatically select based on hardware and operation characteristics
};

// Operation configuration with backend selection heuristics - aligned with
// PyTorch
struct OperationConfig {
  BackendType backend_type = BackendType::kAuto;
  bool use_fused_epilogue = false;  // Use fused bias/activation in cuBLASLt
  bool enable_fast_math = true;     // Enable fast math optimizations
  bool allow_tf32 = true;           // Allow TF32 computation on Ampere+

  // Constructor with default values
  OperationConfig() = default;

  // Constructor with explicit backend type
  explicit OperationConfig(BackendType backend) : backend_type(backend) {}
};

// Unified linear CUDA implementation - simplified and aligned with PyTorch
template <typename T>
class UnifiedLinearCuda {
 public:
  // Constructor and destructor
  explicit UnifiedLinearCuda(const phi::DeviceContext& dev_ctx);
  ~UnifiedLinearCuda() = default;

  // Core operations - simplified interface
  void Dot(const utils::ScaledTensor<T>& A,
           const utils::ScaledTensor<T>& B,
           T alpha,
           T beta,
           DenseTensor* out);

  void Mv(const utils::ScaledTensor<T>& A,
          const utils::ScaledTensor<T>& B,
          bool trans_A,
          T alpha,
          T beta,
          DenseTensor* out,
          const OperationConfig& config = OperationConfig());

  void Mm(const utils::ScaledTensor<T>& A,
          const utils::ScaledTensor<T>& B,
          bool trans_A,
          bool trans_B,
          T alpha,
          T beta,
          DenseTensor* out,
          const OperationConfig& config = OperationConfig());

  void Bmm(const utils::ScaledTensor<T>& A,
           const utils::ScaledTensor<T>& B,
           bool trans_A,
           bool trans_B,
           T alpha,
           T beta,
           DenseTensor* out,
           const OperationConfig& config = OperationConfig());

  void Linear(const utils::ScaledTensor<T>& A,
              const utils::ScaledTensor<T>& B,
              const paddle::optional<DenseTensor>& bias,
              bool trans_A,
              bool trans_B,
              T alpha,
              T beta,
              DenseTensor* out,
              const OperationConfig& config = OperationConfig(),
              ActivationType activation = ActivationType::kNone);

  void ComputeOutputScale(const utils::ScaledTensor<T>& A,
                          const utils::ScaledTensor<T>& B,
                          const utils::ScaledTensor<T>& C,
                          const paddle::optional<DenseTensor>& D_scale,
                          DenseTensor* out_D_scale);

 private:
  // Device context reference
  const phi::DeviceContext& dev_ctx_;

  // Backend selection helpers - aligned with PyTorch's strategy
  BackendType DetermineOptimalBackend(const utils::ScaledTensor<T>& A,
                                      const utils::ScaledTensor<T>& B,
                                      const OperationConfig& config) const;

  // Tensor preparation helpers - simplified
  void PrepareTensors(const utils::ScaledTensor<T>& A,
                      const utils::ScaledTensor<T>& B,
                      bool trans_A,
                      bool trans_B,
                      DenseTensor* prepared_A,
                      DenseTensor* prepared_B,
                      bool need_trans_A,
                      bool need_trans_B) const;

  // Apply bias and activation - simplified
  void ApplyBiasAndActivation(const DenseTensor& input,
                              const paddle::optional<DenseTensor>& bias,
                              DenseTensor* output,
                              ActivationType activation) const;
};

// Hardware-specific function dispatch - simplified
template <typename T>
void Dot(const phi::DeviceContext& dev_ctx,
         const utils::ScaledTensor<T>& A,
         const utils::ScaledTensor<T>& B,
         T alpha,
         T beta,
         DenseTensor* out);

template <typename T>
void Mv(const phi::DeviceContext& dev_ctx,
        const utils::ScaledTensor<T>& A,
        const utils::ScaledTensor<T>& B,
        bool trans_A,
        T alpha,
        T beta,
        DenseTensor* out,
        const OperationConfig& config = OperationConfig());

template <typename T>
void Mm(const phi::DeviceContext& dev_ctx,
        const utils::ScaledTensor<T>& A,
        const utils::ScaledTensor<T>& B,
        bool trans_A,
        bool trans_B,
        T alpha,
        T beta,
        DenseTensor* out,
        const OperationConfig& config = OperationConfig());

template <typename T>
void Bmm(const phi::DeviceContext& dev_ctx,
         const utils::ScaledTensor<T>& A,
         const utils::ScaledTensor<T>& B,
         bool trans_A,
         bool trans_B,
         T alpha,
         T beta,
         DenseTensor* out,
         const OperationConfig& config = OperationConfig());

template <typename T>
void Linear(const phi::DeviceContext& dev_ctx,
            const utils::ScaledTensor<T>& A,
            const utils::ScaledTensor<T>& B,
            const paddle::optional<DenseTensor>& bias,
            bool trans_A,
            bool trans_B,
            T alpha,
            T beta,
            DenseTensor* out,
            const OperationConfig& config = OperationConfig(),
            ActivationType activation = ActivationType::kNone);

template <typename T>
void ComputeOutputScale(const phi::DeviceContext& dev_ctx,
                        const utils::ScaledTensor<T>& A,
                        const utils::ScaledTensor<T>& B,
                        const utils::ScaledTensor<T>& C,
                        const paddle::optional<DenseTensor>& D_scale,
                        DenseTensor* out_D_scale);

}  // namespace cuda
}  // namespace unified_linear
}  // namespace phi
