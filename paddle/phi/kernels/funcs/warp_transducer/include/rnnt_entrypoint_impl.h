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

#include <algorithm>
#include <cstddef>
#include <iostream>

#include "rnnt.h"

#include "detail/cpu_rnnt.h"
#if (defined(__HIPCC__) || defined(__CUDACC__))
#include "detail/gpu_rnnt.h"
#endif

extern "C" {

int get_warprnnt_version() { return 1; }

const char* rnntGetStatusString(rnntStatus_t status) {
  switch (status) {
    case RNNT_STATUS_SUCCESS:
      return "no error";
    case RNNT_STATUS_MEMOPS_FAILED:
      return "cuda memcpy or memset failed";
    case RNNT_STATUS_INVALID_VALUE:
      return "invalid value";
    case RNNT_STATUS_EXECUTION_FAILED:
      return "execution failed";

    case RNNT_STATUS_UNKNOWN_ERROR:
    default:
      return "unknown error";
  }
}

rnntStatus_t compute_rnnt_loss(const float* const activations,  // BTUV
                               float* gradients,
                               const int* const flat_labels,
                               const int* const label_lengths,
                               const int* const input_lengths,
                               int alphabet_size,
                               int minibatch,
                               float* costs,
                               void* workspace,
                               rnntOptions options) {
  if (activations == nullptr || flat_labels == nullptr ||
      label_lengths == nullptr || input_lengths == nullptr ||
      costs == nullptr || workspace == nullptr || alphabet_size <= 0 ||
      minibatch <= 0 || options.maxT <= 0 || options.maxU <= 0 ||
      options.fastemit_lambda < 0)
    return RNNT_STATUS_INVALID_VALUE;

  if (options.loc == RNNT_CPU) {
    CpuRNNT<float> rnnt(minibatch,
                        options.maxT,
                        options.maxU,
                        alphabet_size,
                        workspace,
                        options.blank_label,
                        options.fastemit_lambda,
                        options.num_threads,
                        options.batch_first);

    if (gradients != NULL) {
      return rnnt.cost_and_grad(activations,
                                gradients,
                                costs,
                                flat_labels,
                                label_lengths,
                                input_lengths);
    } else {
      return rnnt.score_forward(
          activations, costs, flat_labels, label_lengths, input_lengths);
    }
  } else if (options.loc == RNNT_GPU) {
#if (defined(__HIPCC__) || defined(__CUDACC__))
    GpuRNNT<float> rnnt(minibatch,
                        options.maxT,
                        options.maxU,
                        alphabet_size,
                        workspace,
                        options.blank_label,
                        options.fastemit_lambda,
                        options.num_threads,
                        options.stream);

    if (gradients != NULL)
      return rnnt.cost_and_grad(activations,
                                gradients,
                                costs,
                                flat_labels,
                                label_lengths,
                                input_lengths);
    else
      return rnnt.score_forward(
          activations, costs, flat_labels, label_lengths, input_lengths);
#else
    std::cerr << "GPU execution requested, but not compiled with GPU support"
              << std::endl;
    return RNNT_STATUS_EXECUTION_FAILED;
#endif
  } else {
    return RNNT_STATUS_INVALID_VALUE;
  }
}

rnntStatus_t get_rnnt_workspace_size(int maxT,
                                     int maxU,
                                     int minibatch,
                                     bool gpu,
                                     size_t* size_bytes,
                                     size_t dtype_size) {
  if (minibatch <= 0 || maxT <= 0 || maxU <= 0)
    return RNNT_STATUS_INVALID_VALUE;

  *size_bytes = 0;

  // per minibatch memory
  size_t per_minibatch_bytes = 0;

  // alphas & betas
  per_minibatch_bytes += dtype_size * maxT * maxU * 2;

  if (!gpu) {
    // blank & label log probability cache
    per_minibatch_bytes += dtype_size * maxT * maxU * 2;
  } else {
    // softmax denominator
    per_minibatch_bytes += dtype_size * maxT * maxU;
    // forward-backward loglikelihood
    per_minibatch_bytes += dtype_size * 2;
  }

  *size_bytes = per_minibatch_bytes * minibatch;

  return RNNT_STATUS_SUCCESS;
}

rnntStatus_t compute_rnnt_loss_fp64(const double* const activations,  // BTUV
                                    double* gradients,
                                    const int* const flat_labels,
                                    const int* const label_lengths,
                                    const int* const input_lengths,
                                    int alphabet_size,
                                    int minibatch,
                                    double* costs,
                                    void* workspace,
                                    rnntOptions options) {
  if (activations == nullptr || flat_labels == nullptr ||
      label_lengths == nullptr || input_lengths == nullptr ||
      costs == nullptr || workspace == nullptr || alphabet_size <= 0 ||
      minibatch <= 0 || options.maxT <= 0 || options.maxU <= 0 ||
      options.fastemit_lambda < 0)
    return RNNT_STATUS_INVALID_VALUE;

  if (options.loc == RNNT_CPU) {
    CpuRNNT<double> rnnt(minibatch,
                         options.maxT,
                         options.maxU,
                         alphabet_size,
                         workspace,
                         options.blank_label,
                         options.fastemit_lambda,
                         options.num_threads,
                         options.batch_first);

    if (gradients != NULL)
      return rnnt.cost_and_grad(activations,
                                gradients,
                                costs,
                                flat_labels,
                                label_lengths,
                                input_lengths);
    else
      return rnnt.score_forward(
          activations, costs, flat_labels, label_lengths, input_lengths);
  } else if (options.loc == RNNT_GPU) {
#if (defined(__HIPCC__) || defined(__CUDACC__))
    GpuRNNT<double> rnnt(minibatch,
                         options.maxT,
                         options.maxU,
                         alphabet_size,
                         workspace,
                         options.blank_label,
                         options.fastemit_lambda,
                         options.num_threads,
                         options.stream);

    if (gradients != NULL)
      return rnnt.cost_and_grad(activations,
                                gradients,
                                costs,
                                flat_labels,
                                label_lengths,
                                input_lengths);
    else
      return rnnt.score_forward(
          activations, costs, flat_labels, label_lengths, input_lengths);
#else
    std::cerr << "GPU execution requested, but not compiled with GPU support"
              << std::endl;
    return RNNT_STATUS_EXECUTION_FAILED;
#endif
  } else {
    return RNNT_STATUS_INVALID_VALUE;
  }
}
}
