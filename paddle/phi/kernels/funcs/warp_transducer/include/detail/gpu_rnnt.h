// Copyright 2018-2019, Mingkun Huang
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

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <tuple>

#if !defined(RNNT_DISABLE_OMP) && !defined(APPLE)
#include <omp.h>
#endif

#include "gpu_rnnt_kernel.h"
#include "reduce.h"

template <typename ProbT>
class GpuRNNT {
 public:
  // Noncopyable
  GpuRNNT(int minibatch,
          int maxT,
          int maxU,
          int alphabet_size,
          void* workspace,
          int blank,
          float fastemit_lambda,
          int num_threads,
          GPUstream stream)
      : minibatch_(minibatch),
        maxT_(maxT),
        maxU_(maxU),
        alphabet_size_(alphabet_size),
        gpu_workspace(workspace),
        blank_(blank),
        fastemit_lambda_(fastemit_lambda),
        num_threads_(num_threads),
        stream_(stream) {
#if defined(RNNT_DISABLE_OMP) || defined(APPLE)
#else
    if (num_threads > 0) {
      omp_set_num_threads(num_threads);
    } else {
      num_threads_ = omp_get_max_threads();
    }
#endif
  };

  GpuRNNT(const GpuRNNT&) = delete;
  GpuRNNT& operator=(const GpuRNNT&) = delete;

  void log_softmax(const ProbT* const acts, ProbT* denom);

  rnntStatus_t compute_cost_and_score(const ProbT* const acts,
                                      ProbT* grad,
                                      ProbT* costs,
                                      const int* const pad_labels,
                                      const int* const label_lengths,
                                      const int* const input_lengths);

  rnntStatus_t cost_and_grad(const ProbT* const acts,
                             ProbT* grad,
                             ProbT* costs,
                             const int* const pad_labels,
                             const int* const label_lengths,
                             const int* const input_lengths);

  rnntStatus_t score_forward(const ProbT* const acts,
                             ProbT* costs,
                             const int* const pad_labels,
                             const int* const label_lengths,
                             const int* const input_lengths);

 private:
  int minibatch_;
  int maxT_;
  int maxU_;
  int alphabet_size_;  // Number of characters plus blank
  void* gpu_workspace;
  int blank_;
  float fastemit_lambda_;
  int num_threads_;
  GPUstream stream_;
};

template <typename ProbT>
void GpuRNNT<ProbT>::log_softmax(const ProbT* const acts, ProbT* denom) {
  // trans_acts + pred_acts -> log_softmax denominator
  reduce_max(
      acts, denom, alphabet_size_, minibatch_ * maxT_ * maxU_, 0, stream_);
  reduce_exp(
      acts, denom, alphabet_size_, minibatch_ * maxT_ * maxU_, 1, stream_);
}

template <typename ProbT>
rnntStatus_t GpuRNNT<ProbT>::compute_cost_and_score(
    const ProbT* const acts,
    ProbT* grads,
    ProbT* costs,
    const int* const labels,
    const int* const label_lengths,
    const int* const input_lengths) {
  bool training = (grads != nullptr);
  size_t bytes_used = 0;
  // denom
  ProbT* denom =
      reinterpret_cast<ProbT*>(static_cast<char*>(gpu_workspace) + bytes_used);
  bytes_used += sizeof(ProbT) * maxT_ * maxU_ * minibatch_;
  // alphas & betas
  ProbT* alphas =
      reinterpret_cast<ProbT*>(static_cast<char*>(gpu_workspace) + bytes_used);
  bytes_used += sizeof(ProbT) * maxT_ * maxU_ * minibatch_;
  ProbT* betas =
      reinterpret_cast<ProbT*>(static_cast<char*>(gpu_workspace) + bytes_used);
  bytes_used += sizeof(ProbT) * maxT_ * maxU_ * minibatch_;
  // logllh
  ProbT* llForward =
      reinterpret_cast<ProbT*>(static_cast<char*>(gpu_workspace) + bytes_used);
  bytes_used += sizeof(ProbT) * minibatch_;
  ProbT* llBackward =
      reinterpret_cast<ProbT*>(static_cast<char*>(gpu_workspace) + bytes_used);
  bytes_used += sizeof(ProbT) * minibatch_;

  if (training) {
    // zero grads
#ifdef __HIPCC__
    hipMemsetAsync(grads,
                   0,
                   sizeof(ProbT) * minibatch_ * maxT_ * maxU_ * alphabet_size_,
                   stream_);
#else
    cudaMemsetAsync(grads,
                    0,
                    sizeof(ProbT) * minibatch_ * maxT_ * maxU_ * alphabet_size_,
                    stream_);
#endif
  }
  // denom
#if defined(DEBUG_TIME)
  auto start = std::chrono::high_resolution_clock::now();
#endif
  log_softmax(acts, denom);
#if defined(DEBUG_TIME)
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "DEBUG: log_softmax " << elapsed.count() * 1000 << " ms\n";
  // alphas
  start = std::chrono::high_resolution_clock::now();
#endif
#if defined(USE_NAIVE_KERNEL)
  compute_alphas_kernel_naive<ProbT>
      <<<1, minibatch_, 0, stream_>>>(acts,
                                      denom,
                                      alphas,
                                      llForward,
                                      input_lengths,
                                      label_lengths,
                                      labels,
                                      minibatch_,
                                      maxT_,
                                      maxU_,
                                      alphabet_size_,
                                      blank_);
#else
  compute_alphas_kernel<ProbT>
      <<<minibatch_, maxU_, 0, stream_>>>(acts,
                                          denom,
                                          alphas,
                                          llForward,
                                          input_lengths,
                                          label_lengths,
                                          labels,
                                          minibatch_,
                                          maxT_,
                                          maxU_,
                                          alphabet_size_,
                                          blank_);
#endif
#if defined(DEBUG_TIME)
#ifdef __HIPCC__
  hipStreamSynchronize(stream_);
#else
  cudaStreamSynchronize(stream_);
#endif
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "DEBUG: compute_alphas_kernel " << elapsed.count() * 1000
            << " ms\n";
#endif
#if defined(DEBUG_KERNEL)
  ProbT* cpu_alphas = new ProbT[minibatch_ * maxT_ * maxU_];
  int* cpu_xlen = new int[minibatch_];
  int* cpu_ylen = new int[minibatch_];
#ifdef __HIPCC__
  hipMemcpy(cpu_alphas,
            alphas,
            sizeof(ProbT) * minibatch_ * maxT_ * maxU_,
            hipMemcpyDeviceToHost);
  hipMemcpy(
      cpu_xlen, input_lengths, sizeof(int) * minibatch_, hipMemcpyDeviceToHost);
  hipMemcpy(
      cpu_ylen, label_lengths, sizeof(int) * minibatch_, hipMemcpyDeviceToHost);
#else
  cudaMemcpy(cpu_alphas,
             alphas,
             sizeof(ProbT) * minibatch_ * maxT_ * maxU_,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_xlen,
             input_lengths,
             sizeof(int) * minibatch_,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_ylen,
             label_lengths,
             sizeof(int) * minibatch_,
             cudaMemcpyDeviceToHost);
#endif
  printf("gpu alphas\n");
  for (int b = 0; b < minibatch_; b++) {
    int T = cpu_xlen[b];
    int U = cpu_ylen[b] + 1;
    printf("B %d, T %d, U %d\n", b, T, U);
    for (int t = 0; t < T; t++) {
      for (int u = 0; u < U; u++) {
        printf("%.2f ", cpu_alphas[(b * maxT_ + t) * maxU_ + u]);
      }
      printf("\n");
    }
    printf("\n");
  }
#endif
  if (training) {
    // betas
#if defined(DEBUG_TIME)
    start = std::chrono::high_resolution_clock::now();
#endif
#if defined(USE_NAIVE_KERNEL)
    compute_betas_kernel_naive<ProbT>
        <<<1, minibatch_, 0, stream_>>>(acts,
                                        denom,
                                        betas,
                                        llBackward,
                                        input_lengths,
                                        label_lengths,
                                        labels,
                                        minibatch_,
                                        maxT_,
                                        maxU_,
                                        alphabet_size_,
                                        blank_);
#else
    compute_betas_kernel<ProbT>
        <<<minibatch_, maxU_, 0, stream_>>>(acts,
                                            denom,
                                            betas,
                                            llBackward,
                                            input_lengths,
                                            label_lengths,
                                            labels,
                                            minibatch_,
                                            maxT_,
                                            maxU_,
                                            alphabet_size_,
                                            blank_);
#endif
#if defined(DEBUG_TIME)
#ifdef __HIPCC__
    hipStreamSynchronize(stream_);
#else
    cudaStreamSynchronize(stream_);
#endif
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "DEBUG: compute_betas_kernel " << elapsed.count() * 1000
              << " ms\n";
#endif
#if defined(DEBUG_KERNEL)
    ProbT* cpu_betas = new ProbT[minibatch_ * maxT_ * maxU_];
#ifdef __HIPCC__
    hipMemcpy(cpu_betas,
              betas,
              sizeof(ProbT) * minibatch_ * maxT_ * maxU_,
              hipMemcpyDeviceToHost);
#else
    cudaMemcpy(cpu_betas,
               betas,
               sizeof(ProbT) * minibatch_ * maxT_ * maxU_,
               cudaMemcpyDeviceToHost);
#endif
    printf("gpu betas\n");
    for (int b = 0; b < minibatch_; b++) {
      int T = cpu_xlen[b];
      int U = cpu_ylen[b] + 1;
      printf("B %d, T %d, U %d\n", b, T, U);
      for (int t = 0; t < T; t++) {
        for (int u = 0; u < U; u++) {
          printf("%.2f ", cpu_betas[(b * maxT_ + t) * maxU_ + u]);
        }
        printf("\n");
      }
      printf("\n");
    }
#endif

    // gradient
#if defined(DEBUG_TIME)
    start = std::chrono::high_resolution_clock::now();
#endif
    // TODO optimize gradient kernel

    if (fastemit_lambda_ > 0.0f) {
      compute_fastemit_grad_kernel<128, ProbT>
          <<<minibatch_ * maxT_ * maxU_, 128, 0, stream_>>>(grads,
                                                            acts,
                                                            denom,
                                                            alphas,
                                                            betas,
                                                            llForward,
                                                            input_lengths,
                                                            label_lengths,
                                                            labels,
                                                            minibatch_,
                                                            maxT_,
                                                            maxU_,
                                                            alphabet_size_,
                                                            blank_,
                                                            fastemit_lambda_);
    } else {
      compute_grad_kernel<128, ProbT>
          <<<minibatch_ * maxT_ * maxU_, 128, 0, stream_>>>(grads,
                                                            acts,
                                                            denom,
                                                            alphas,
                                                            betas,
                                                            llForward,
                                                            input_lengths,
                                                            label_lengths,
                                                            labels,
                                                            minibatch_,
                                                            maxT_,
                                                            maxU_,
                                                            alphabet_size_,
                                                            blank_);
    }
#if defined(DEBUG_TIME)
#ifdef __HIPCC__
    hipStreamSynchronize(stream_);
#else
    cudaStreamSynchronize(stream_);
#endif
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "DEBUG: compute_grad_kernel " << elapsed.count() * 1000
              << " ms\n";
#endif
  }
  // cost
#ifdef __HIPCC__
  hipMemcpyAsync(costs,
                 llForward,
                 sizeof(ProbT) * minibatch_,
                 hipMemcpyDeviceToHost,
                 stream_);
  hipStreamSynchronize(stream_);
#else
  cudaMemcpyAsync(costs,
                  llForward,
                  sizeof(ProbT) * minibatch_,
                  cudaMemcpyDeviceToHost,
                  stream_);
  cudaStreamSynchronize(stream_);
#endif
  for (int mb = 0; mb < minibatch_; ++mb) {
    costs[mb] = -costs[mb];
  }
  return RNNT_STATUS_SUCCESS;
}

template <typename ProbT>
rnntStatus_t GpuRNNT<ProbT>::cost_and_grad(const ProbT* const acts,
                                           ProbT* grads,
                                           ProbT* costs,
                                           const int* const pad_labels,
                                           const int* const label_lengths,
                                           const int* const input_lengths) {
  if (acts == nullptr || grads == nullptr || costs == nullptr ||
      pad_labels == nullptr || label_lengths == nullptr ||
      input_lengths == nullptr)
    return RNNT_STATUS_INVALID_VALUE;

  return compute_cost_and_score(
      acts, grads, costs, pad_labels, label_lengths, input_lengths);
}

template <typename ProbT>
rnntStatus_t GpuRNNT<ProbT>::score_forward(const ProbT* const acts,
                                           ProbT* costs,
                                           const int* const pad_labels,
                                           const int* const label_lengths,
                                           const int* const input_lengths) {
  if (acts == nullptr || costs == nullptr || pad_labels == nullptr ||
      label_lengths == nullptr || input_lengths == nullptr)
    return RNNT_STATUS_INVALID_VALUE;

  return compute_cost_and_score(
      acts, nullptr, costs, pad_labels, label_lengths, input_lengths);
}
