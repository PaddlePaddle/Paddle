/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/operators/elementwise/elementwise_add_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast.cu.h"
#include "paddle/fluid/operators/softmax_cudnn_op.cu.h"
#include "paddle/fluid/operators/transpose_op.cu.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class FMHARef {
 public:
  FMHARef(const platform::CUDADeviceContext& dev_ctx, int64_t batch_size,
          int64_t seq_len, int64_t num_head, int64_t head_dim)
      : dev_ctx_(dev_ctx),
        batch_size_(batch_size),
        seq_len_(seq_len),
        num_head_(num_head),
        head_dim_(head_dim) {}

  ~FMHARef() {}

  void ComputeForward(const Tensor& qkv_input_tensor,
                      const Tensor& src_mask_tensor,
                      Tensor* transpose_2_out_tensor, Tensor* qk_out_tensor,
                      Tensor* src_mask_out_tensor, Tensor* softmax_out_tensor,
                      Tensor* qktv_out_tensor, Tensor* fmha_out_tensor) {
    // input shape: [bs, seq_len, 3, num_head, head_dim]
    // transpose with perm [2, 0, 1, 3, 4],
    // output_shape: [3, bs, num_head, seq_len, head_dim]
    int ndims = 5;
    std::vector<int> perm_1 = {2, 0, 3, 1, 4};
    TransposeGPUKernelDriver<T>(dev_ctx_, ndims, qkv_input_tensor, perm_1,
                                transpose_2_out_tensor);
#if 1
    T* qkv_data = transpose_2_out_tensor->data<T>();
    T* qk_out_data = qk_out_tensor->data<T>();
    T* qktv_out_data = qktv_out_tensor->data<T>();
    T* softmax_out_data = softmax_out_tensor->data<T>();
    T* fmha_out_data = fmha_out_tensor->data<T>();
#endif

    int q_size = batch_size_ * seq_len_ * num_head_ * head_dim_;
    int k_size = q_size;
#if 1
    T* q_ptr = qkv_data;
    T* k_ptr = q_ptr + q_size;
    T* v_ptr = k_ptr + k_size;
#endif

// q*k^t, batched_gemm
#if 1
    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_TRANSPOSE transB = CblasTrans;
    auto blas = math::GetBlas<platform::CUDADeviceContext, T>(dev_ctx_);
    int gemm_batch_size = batch_size_ * num_head_;
    int gemm_m = seq_len_;
    int gemm_n = seq_len_;
    int gemm_k = head_dim_;
    T alpha = static_cast<T>(1.0 / sqrt(head_dim_));
    T beta = static_cast<T>(0.0);
    int64_t stride_a = gemm_m * gemm_k;
    int64_t stride_b = gemm_k * gemm_n;
#endif
#if 1
    blas.BatchedGEMM(transA, transB, gemm_m, gemm_n, gemm_k, alpha, q_ptr,
                     k_ptr, beta, qk_out_data, gemm_batch_size, stride_a,
                     stride_b);
#endif

    std::vector<const Tensor*> ins;
    std::vector<Tensor*> outs;
    ins.emplace_back(qk_out_tensor);
    ins.emplace_back(&src_mask_tensor);
    outs.emplace_back(src_mask_out_tensor);
    int elewise_add_axis = -1;
    int softmax_axis = -1;
    if (&src_mask_tensor != nullptr) {
      // mask_out = qk_out + src_mask
      //   LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
      //       dev_ctx_, ins, &outs, elewise_add_axis, CudaAddFunctor<T>());
      LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
          dev_ctx_, ins, &outs, elewise_add_axis, AddFunctor<T>());
      // softmax(mask_out)
      SoftmaxForwardCUDAKernelDriver<T>(dev_ctx_, *src_mask_out_tensor,
                                        softmax_axis, softmax_out_tensor);
    } else {
// softmax_out = softmax(qk_out)
#if 1
      SoftmaxForwardCUDAKernelDriver<T>(dev_ctx_, *qk_out_tensor, softmax_axis,
                                        softmax_out_tensor);
#endif
    }
// todo: add dropout

// softmax_out * v, batched_gemm
// output shape: [batch_size, num_heads, seq_len, head_dim]
#if 1
    transB = CblasNoTrans;
    gemm_m = seq_len_;
    gemm_n = head_dim_;
    gemm_k = seq_len_;
    alpha = static_cast<T>(1.0);
    stride_a = gemm_m * gemm_k;
    stride_b = gemm_k * gemm_n;
    blas.BatchedGEMM(transA, transB, gemm_m, gemm_n, gemm_k, alpha,
                     softmax_out_data, v_ptr, beta, qktv_out_data,
                     gemm_batch_size, stride_a, stride_b);
#endif
    // transpose: [0, 2, 1, 3]
    // output shape: [batch_size, seq_len, num_heads, head_dim]
    std::vector<int> perm_3 = {0, 2, 1, 3};
    ndims = 4;
    TransposeGPUKernelDriver<T>(dev_ctx_, ndims, *qktv_out_tensor, perm_3,
                                fmha_out_tensor);
  }

  void ComputeBackward(
      const Tensor& transpose_2_out_tensor, const Tensor& src_mask_tensor,
      const Tensor& softmax_out_tensor, const Tensor& qk_out_tensor,
      const Tensor& src_mask_out_tensor, const Tensor& fmha_out_grad_tensor,
      Tensor* qktv_out_grad_tensor, Tensor* softmax_out_grad_tensor,
      Tensor* src_mask_out_grad_tensor, Tensor* qk_out_grad_tensor,
      Tensor* transpose_2_out_grad_tensor, Tensor* src_mask_grad_tensor,
      Tensor* qkv_input_grad_tensor) {
    auto blas = math::GetBlas<platform::CUDADeviceContext, T>(dev_ctx_);
    int q_size = batch_size_ * seq_len_ * num_head_ * head_dim_;
    int k_size = q_size;
    int softmax_axis = -1;

#if 1
    T* qkv_grad_data = transpose_2_out_grad_tensor->data<T>();
    T* q_grad_ptr = qkv_grad_data;
    T* k_grad_ptr = q_grad_ptr + q_size;
    T* v_grad_ptr = k_grad_ptr + k_size;
#endif
#if 1
    const T* qkv_data = transpose_2_out_tensor.data<T>();
    const T* q_ptr = qkv_data;
    const T* k_ptr = q_ptr + q_size;
    const T* v_ptr = k_ptr + k_size;
#endif
    const T* softmax_out_data = softmax_out_tensor.data<T>();
    T* softmax_out_grad_data = softmax_out_grad_tensor->data<T>();
    T* qktv_out_grad_data = qktv_out_grad_tensor->data<T>();

    // transpose bw
    int ndims = 4;
    std::vector<int> perm_3 = {0, 2, 1, 3};
    TransposeGPUKernelDriver<T>(dev_ctx_, ndims, fmha_out_grad_tensor, perm_3,
                                qktv_out_grad_tensor);

    // recall batchedgemm(nn) fw: softmax_out_data(x) * v_ptr(y) =
    // qktv_out_data(out)
    CBLAS_TRANSPOSE transA = CblasTrans;
    CBLAS_TRANSPOSE transB = CblasNoTrans;
    int gemm_batch_size = batch_size_ * num_head_;
    int gemm_m = seq_len_;
    int gemm_n = head_dim_;
    int gemm_k = seq_len_;
    T alpha = static_cast<T>(1.0);
    T beta = static_cast<T>(0.0);
    int64_t stride_a = gemm_m * gemm_k;
    int64_t stride_b = gemm_k * gemm_n;
    // bw: dy = x^t * dout
    blas.BatchedGEMM(transA, transB, gemm_m, gemm_n, gemm_k, alpha,
                     softmax_out_data, qktv_out_grad_data, beta, v_grad_ptr,
                     gemm_batch_size, stride_a, stride_b);
    // bw: dx = dout * y^t
    transA = CblasNoTrans;
    transB = CblasTrans;
    gemm_m = seq_len_;
    gemm_n = seq_len_;
    gemm_k = head_dim_;
    stride_a = gemm_m * gemm_k;
    stride_b = gemm_k * gemm_n;
    blas.BatchedGEMM(transA, transB, gemm_m, gemm_n, gemm_k, alpha,
                     qktv_out_grad_data, v_ptr, beta, softmax_out_grad_data,
                     gemm_batch_size, stride_a, stride_b);

    // dropout bw
    if (&src_mask_tensor != nullptr) {
      SoftmaxBackwardCUDAKernelDriver<T>(dev_ctx_, softmax_out_tensor,
                                         *softmax_out_grad_tensor, softmax_axis,
                                         src_mask_out_grad_tensor);

      // recall LaunchElementwiseCudaKernel fw:  src_mask_out = qk_out +
      // src_mask
      // Special case when dy is not needed and dx doesn't reduce
      if (qk_out_grad_tensor != nullptr && src_mask_grad_tensor == nullptr &&
          qk_out_tensor.dims() == src_mask_out_tensor.dims()) {
        VLOG(4) << "Special case when dy is not needed and dx doesn't "
                   "reduce";
        framework::TensorCopy(*src_mask_out_grad_tensor, dev_ctx_.GetPlace(),
                              dev_ctx_, qk_out_grad_tensor);
      } else {
        // todo:
        std::cout << "NotImplemented\n";
        return;
      }

    } else {
      SoftmaxBackwardCUDAKernelDriver<T>(dev_ctx_, softmax_out_tensor,
                                         *softmax_out_grad_tensor, softmax_axis,
                                         qk_out_grad_tensor);
    }

    T* qk_out_grad_data = qk_out_grad_tensor->data<T>();
    alpha = static_cast<T>(1.0 / sqrt(head_dim_));
    // recall batchedgemm(nt) fw:  q_ptr * (k_ptr)^t = qk_out
    // bw: dy (seq_len * head_dim) = (dout)^t * x
    transA = CblasTrans;
    transB = CblasNoTrans;
    gemm_m = seq_len_;
    gemm_n = head_dim_;
    gemm_k = seq_len_;
    stride_a = gemm_m * gemm_k;
    stride_b = gemm_k * gemm_n;
    blas.BatchedGEMM(transA, transB, gemm_m, gemm_n, gemm_k, alpha,
                     qk_out_grad_data, q_ptr, beta, k_grad_ptr, gemm_batch_size,
                     stride_a, stride_b);
    // dx (seq_len * head_dim) = dout * y
    transA = CblasNoTrans;
    transB = CblasNoTrans;
    gemm_m = seq_len_;
    gemm_n = head_dim_;
    gemm_k = seq_len_;
    stride_a = gemm_m * gemm_k;
    stride_b = gemm_k * gemm_n;
    blas.BatchedGEMM(transA, transB, gemm_m, gemm_n, gemm_k, alpha,
                     qk_out_grad_data, k_ptr, beta, q_grad_ptr, gemm_batch_size,
                     stride_a, stride_b);

    // transpose bw
    ndims = 5;
    std::vector<int> perm_1 = {1, 3, 0, 2, 4};
    TransposeGPUKernelDriver<T>(dev_ctx_, ndims, *transpose_2_out_grad_tensor,
                                perm_1, qkv_input_grad_tensor);
  }

 private:
  const platform::CUDADeviceContext& dev_ctx_;

  int64_t batch_size_;
  int64_t seq_len_;
  int64_t num_head_;
  int64_t head_dim_;
};

}  // namespace operators
}  // namespace paddle
