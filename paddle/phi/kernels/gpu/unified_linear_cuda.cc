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

#include "paddle/phi/kernels/gpu/unified_linear_cuda.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

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

// Implementation of UnifiedLinearCuda class
template <typename T>
UnifiedLinearCuda<T>::UnifiedLinearCuda(const phi::DeviceContext& dev_ctx)
    : dev_ctx_(dev_ctx) {
  // Constructor implementation - simplified
}

// Backend selection - aligned with PyTorch's strategy
template <typename T>
BackendType UnifiedLinearCuda<T>::DetermineOptimalBackend(
    const utils::ScaledTensor<T>& A,
    const utils::ScaledTensor<T>& B,
    const OperationConfig& config) const {
  // If backend is explicitly specified, use it
  if (config.backend_type != BackendType::kAuto) {
    return config.backend_type;
  }

  // Get GPU device properties
  auto* gpu_ctx = dynamic_cast<const phi::GPUContext*>(&dev_ctx_);
  PADDLE_ENFORCE_NOT_NULL(gpu_ctx,
                          phi::errors::InvalidArgument(
                              "GPU context is required for UnifiedLinearCuda"));

  const auto& device_props = gpu_ctx->GetDeviceProperties();

  // Check if we should use cuBLASLt based on PyTorch's heuristics
  bool use_cublaslt = true;

  // Disable cuBLASLt for very small tensors (PyTorch heuristic)
  if (A.tensor->numel() < 1024 || B.tensor->numel() < 1024) {
    use_cublaslt = false;
  }

  // Check for specific device/architecture limitations
  // Similar to PyTorch's checks for CUDA version and architecture
  if (device_props.major < 7) {
    // Older architectures may not benefit from cuBLASLt
    use_cublaslt = false;
  }

  // Check for specific data type limitations
  if (std::is_same<T, phi::dtype::float16>::value ||
      std::is_same<T, phi::dtype::bfloat16>::value) {
    // For half precision, prefer cuBLASLt for better performance
    use_cublaslt = true;
  }

  // Check for specific operation types
  if (config.use_fused_epilogue) {
    // For fused operations, cuBLASLt is preferred
    use_cublaslt = true;
  }

  return use_cublaslt ? BackendType::kCublasLt : BackendType::kCublas;
}

// Tensor preparation - simplified
template <typename T>
void UnifiedLinearCuda<T>::PrepareTensors(const utils::ScaledTensor<T>& A,
                                          const utils::ScaledTensor<T>& B,
                                          bool trans_A,
                                          bool trans_B,
                                          DenseTensor* prepared_A,
                                          DenseTensor* prepared_B,
                                          bool need_trans_A,
                                          bool need_trans_B) const {
  // Check if tensors are already in optimal format
  bool A_optimal = A.tensor->is_contiguous();
  bool B_optimal = B.tensor->is_contiguous();

  need_trans_A = !A_optimal;
  need_trans_B = !B_optimal;

  // Prepare A tensor
  if (A_optimal) {
    *prepared_A = *A.tensor;
  } else {
    phi::Copy(dev_ctx_, *A.tensor, false, prepared_A);
  }

  // Prepare B tensor
  if (B_optimal) {
    *prepared_B = *B.tensor;
  } else {
    phi::Copy(dev_ctx_, *B.tensor, false, prepared_B);
  }
}

// Apply bias and activation - simplified
template <typename T>
void UnifiedLinearCuda<T>::ApplyBiasAndActivation(
    const DenseTensor& input,
    const paddle::optional<DenseTensor>& bias,
    DenseTensor* output,
    ActivationType activation) const {
  // If no bias and no activation, just copy
  if (!bias.is_initialized() && activation == ActivationType::kNone) {
    phi::Copy(dev_ctx_, input, false, output);
    return;
  }

  // Apply bias if provided
  if (bias.is_initialized()) {
    // Add bias to input
    phi::AddKernel<T>(dev_ctx_, input, bias.get(), output);
  } else {
    // Just copy input to output
    phi::Copy(dev_ctx_, input, false, output);
  }

  // Apply activation if needed
  if (activation != ActivationType::kNone) {
    switch (activation) {
      case ActivationType::kRelu:
        phi::ReluKernel<T>(dev_ctx_, *output, output);
        break;
      case ActivationType::kGelu:
        phi::GeluKernel<T>(dev_ctx_, *output, output);
        break;
      case ActivationType::kSigmoid:
        phi::SigmoidKernel<T>(dev_ctx_, *output, output);
        break;
      case ActivationType::kTanh:
        phi::TanhKernel<T>(dev_ctx_, *output, output);
        break;
      default:
        break;
    }
  }
}

// Dot product implementation
template <typename T>
void UnifiedLinearCuda<T>::Dot(const utils::ScaledTensor<T>& A,
                               const utils::ScaledTensor<T>& B,
                               T alpha,
                               T beta,
                               DenseTensor* out) {
  // For dot product, always use cuBLAS (cuBLASLt doesn't have a direct dot
  // product)
  phi::cublas::CublasLinear<T> cublas_linear(dev_ctx_);
  cublas_linear.DotProduct(*A.tensor, *B.tensor, *A.tensor, alpha, beta, out);
}

// Matrix-vector multiplication implementation
template <typename T>
void UnifiedLinearCuda<T>::Mv(const utils::ScaledTensor<T>& A,
                              const utils::ScaledTensor<T>& B,
                              bool trans_A,
                              T alpha,
                              T beta,
                              DenseTensor* out,
                              const OperationConfig& config) {
  // For matrix-vector multiplication, always use cuBLAS
  // cuBLASLt doesn't have a dedicated matrix-vector operation
  phi::cublas::CublasLinear<T> cublas_linear(dev_ctx_);
  cublas_linear.MatrixVector(
      *A.tensor, *B.tensor, *A.tensor, trans_A, false, alpha, beta, out);
}

// Matrix-matrix multiplication implementation
template <typename T>
void UnifiedLinearCuda<T>::Mm(const utils::ScaledTensor<T>& A,
                              const utils::ScaledTensor<T>& B,
                              bool trans_A,
                              bool trans_B,
                              T alpha,
                              T beta,
                              DenseTensor* out,
                              const OperationConfig& config) {
  // Determine optimal backend
  BackendType backend = DetermineOptimalBackend(A, B, config);

  // Dispatch to appropriate backend
  if (backend == BackendType::kCublasLt) {
    cublaslt_impl::CublasLtLinear<T> cublaslt_linear(dev_ctx_);
    cublaslt_linear.MatrixMatrix(*A.tensor,
                                 *B.tensor,
                                 *A.tensor,
                                 trans_A,
                                 trans_B,
                                 alpha,
                                 beta,
                                 out,
                                 config);
  } else {
    phi::cublas::CublasLinear<T> cublas_linear(dev_ctx_);
    cublas_linear.MatrixMatrix(
        *A.tensor, *B.tensor, *A.tensor, trans_A, trans_B, alpha, beta, out);
  }
}

// Batched matrix-matrix multiplication implementation
template <typename T>
void UnifiedLinearCuda<T>::Bmm(const utils::ScaledTensor<T>& A,
                               const utils::ScaledTensor<T>& B,
                               bool trans_A,
                               bool trans_B,
                               T alpha,
                               T beta,
                               DenseTensor* out,
                               const OperationConfig& config) {
  // Determine optimal backend
  BackendType backend = DetermineOptimalBackend(A, B, config);

  // Dispatch to appropriate backend
  if (backend == BackendType::kCublasLt) {
    cublaslt_impl::CublasLtLinear<T> cublaslt_linear(dev_ctx_);
    cublaslt_linear.BatchedMatrixMatrix(*A.tensor,
                                        *B.tensor,
                                        *A.tensor,
                                        trans_A,
                                        trans_B,
                                        alpha,
                                        beta,
                                        out,
                                        config);
  } else {
    phi::cublas::CublasLinear<T> cublas_linear(dev_ctx_);
    cublas_linear.BatchedMatrixMatrix(
        *A.tensor, *B.tensor, *A.tensor, trans_A, trans_B, alpha, beta, out);
  }
}

// Linear transformation implementation
template <typename T>
void UnifiedLinearCuda<T>::Linear(const utils::ScaledTensor<T>& A,
                                  const utils::ScaledTensor<T>& B,
                                  const paddle::optional<DenseTensor>& Bias,
                                  bool trans_A,
                                  bool trans_B,
                                  T alpha,
                                  T beta,
                                  DenseTensor* out,
                                  const OperationConfig& config,
                                  ActivationType activation) {
  // Determine optimal backend
  BackendType backend = DetermineOptimalBackend(A, B, config);

  // For fused operations, prefer cuBLASLt
  bool use_fused =
      config.use_fused_epilogue &&
      (bias.is_initialized() || activation != ActivationType::kNone) &&
      backend == BackendType::kCublasLt;

  if (use_fused) {
    // Use cuBLASLt with fused epilogue for bias and activation
    cublaslt_impl::CublasLtLinear<T> cublaslt_linear(dev_ctx_);
    // Convert ActivationType to unified_linear::ActivationType
    unified_linear::ActivationType unified_activation;
    switch (activation) {
      case ActivationType::kNone:
        unified_activation = unified_linear::ActivationType::kNone;
        break;
      case ActivationType::kRelu:
        unified_activation = unified_linear::ActivationType::kRelu;
        break;
      case ActivationType::kGelu:
        unified_activation = unified_linear::ActivationType::kGelu;
        break;
      case ActivationType::kSigmoid:
        unified_activation = unified_linear::ActivationType::kSigmoid;
        break;
      case ActivationType::kTanh:
        unified_activation = unified_linear::ActivationType::kTanh;
        break;
    }
    cublaslt_linear.Linear(*A.tensor,
                           *B.tensor,
                           *A.tensor,
                           bias,
                           trans_A,
                           trans_B,
                           alpha,
                           beta,
                           out,
                           config,
                           unified_activation);
  } else {
    // Use cuBLAS for matrix multiplication and then apply bias and activation
    DenseTensor matmul_out;
    matmul_out.Resize(out->dims());
    dev_ctx_.template Alloc<T>(&matmul_out);

    // Perform matrix multiplication
    if (backend == BackendType::kCublasLt) {
      cublaslt_impl::CublasLtLinear<T> cublaslt_linear(dev_ctx_);
      cublaslt_linear.MatrixMatrix(*A.tensor,
                                   *B.tensor,
                                   *A.tensor,
                                   trans_A,
                                   trans_B,
                                   alpha,
                                   beta,
                                   &matmul_out,
                                   config);
    } else {
      phi::cublas::CublasLinear<T> cublas_linear(dev_ctx_);
      cublas_linear.MatrixMatrix(*A.tensor,
                                 *B.tensor,
                                 *A.tensor,
                                 trans_A,
                                 trans_B,
                                 alpha,
                                 beta,
                                 &matmul_out);
    }

    // Apply bias and activation
    ApplyBiasAndActivation(matmul_out, bias, out, activation);
  }
}

// Compute output scale implementation
template <typename T>
void UnifiedLinearCuda<T>::ComputeOutputScale(
    const utils::ScaledTensor<T>& A,
    const utils::ScaledTensor<T>& B,
    const utils::ScaledTensor<T>& C,
    const paddle::optional<DenseTensor>& D_scale,
    DenseTensor* out_D_scale) {
  // For scale computation, always use cuBLASLt (cuBLAS doesn't support this)
  cublaslt_impl::CublasLtLinear<T> cublaslt_linear(dev_ctx_);
  cublaslt_linear.ComputeOutputScale(
      *A.tensor, *B.tensor, *C.tensor, D_scale, out_D_scale);
}

// Hardware-specific function implementations
template <typename T>
void Dot(const phi::DeviceContext& dev_ctx,
         const utils::ScaledTensor<T>& A,
         const utils::ScaledTensor<T>& B,
         T alpha,
         T beta,
         DenseTensor* out) {
  UnifiedLinearCuda<T> impl(dev_ctx);
  impl.Dot(A, B, alpha, beta, out);
}

template <typename T>
void Mv(const phi::DeviceContext& dev_ctx,
        const utils::ScaledTensor<T>& A,
        const utils::ScaledTensor<T>& B,
        bool trans_A,
        T alpha,
        T beta,
        DenseTensor* out,
        const OperationConfig& config) {
  UnifiedLinearCuda<T> impl(dev_ctx);
  impl.Mv(A, B, trans_A, alpha, beta, out, config);
}

template <typename T>
void Mm(const phi::DeviceContext& dev_ctx,
        const utils::ScaledTensor<T>& A,
        const utils::ScaledTensor<T>& B,
        bool trans_A,
        bool trans_B,
        T alpha,
        T beta,
        DenseTensor* out,
        const OperationConfig& config) {
  UnifiedLinearCuda<T> impl(dev_ctx);
  impl.Mm(A, B, trans_A, trans_B, alpha, beta, out, config);
}

template <typename T>
void Bmm(const phi::DeviceContext& dev_ctx,
         const utils::ScaledTensor<T>& A,
         const utils::ScaledTensor<T>& B,
         bool trans_A,
         bool trans_B,
         T alpha,
         T beta,
         DenseTensor* out,
         const OperationConfig& config) {
  UnifiedLinearCuda<T> impl(dev_ctx);
  impl.Bmm(A, B, trans_A, trans_B, alpha, beta, out, config);
}

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
            const OperationConfig& config,
            ActivationType activation) {
  UnifiedLinearCuda<T> impl(dev_ctx);
  impl.Linear(
      A, B, bias, trans_A, trans_B, alpha, beta, out, config, activation);
}

template <typename T>
void ComputeOutputScale(const phi::DeviceContext& dev_ctx,
                        const utils::ScaledTensor<T>& A,
                        const utils::ScaledTensor<T>& B,
                        const utils::ScaledTensor<T>& C,
                        const paddle::optional<DenseTensor>& D_scale,
                        DenseTensor* out_D_scale) {
  UnifiedLinearCuda<T> impl(dev_ctx);
  impl.ComputeOutputScale(A, B, C, D_scale, out_D_scale);
}

// Explicit instantiation for common data types
template class UnifiedLinearCuda<float>;
template class UnifiedLinearCuda<double>;
template class UnifiedLinearCuda<phi::dtype::float16>;
template class UnifiedLinearCuda<phi::dtype::bfloat16>;

}  // namespace cuda
}  // namespace unified_linear
}  // namespace phi
