// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <ctc.h>

#include "detail/cpu_ctc.h"
#ifdef __CUDACC__
#include "detail/gpu_ctc.h"
#endif

extern "C" {

int get_warpctc_version() { return 2; }

const char* ctcGetStatusString(ctcStatus_t status) {
  switch (status) {
    case CTC_STATUS_SUCCESS:
      return "no error";
    case CTC_STATUS_MEMOPS_FAILED:
      return "cuda memcpy or memset failed";
    case CTC_STATUS_INVALID_VALUE:
      return "invalid value";
    case CTC_STATUS_EXECUTION_FAILED:
      return "execution failed";

    case CTC_STATUS_UNKNOWN_ERROR:
    default:
      return "unknown error";
  }
}

ctcStatus_t compute_ctc_loss(const float* const activations,
                             float* gradients,
                             const int* const flat_labels,
                             const int* const label_lengths,
                             const int* const input_lengths,
                             int alphabet_size,
                             int minibatch,
                             float* costs,
                             void* workspace,
                             ctcOptions options) {
  if (activations == nullptr || flat_labels == nullptr ||
      label_lengths == nullptr || input_lengths == nullptr ||
      costs == nullptr || workspace == nullptr || alphabet_size <= 0 ||
      minibatch <= 0)
    return CTC_STATUS_INVALID_VALUE;

  if (options.loc == CTC_CPU) {
    CpuCTC<float> ctc(alphabet_size,
                      minibatch,
                      workspace,
                      options.num_threads,
                      options.blank_label);

    if (gradients != NULL)
      return ctc.cost_and_grad(activations,
                               gradients,
                               costs,
                               flat_labels,
                               label_lengths,
                               input_lengths);
    else
      return ctc.score_forward(
          activations, costs, flat_labels, label_lengths, input_lengths);
  } else if (options.loc == CTC_GPU) {
#ifdef __CUDACC__
    GpuCTC<float> ctc(alphabet_size,
                      minibatch,
                      workspace,
                      options.stream,
                      options.blank_label);

    if (gradients != NULL)
      return ctc.cost_and_grad(activations,
                               gradients,
                               costs,
                               flat_labels,
                               label_lengths,
                               input_lengths);
    else
      return ctc.score_forward(
          activations, costs, flat_labels, label_lengths, input_lengths);
#else
    std::cerr << "GPU execution requested, but not compiled with GPU support"
              << std::endl;
    return CTC_STATUS_EXECUTION_FAILED;
#endif
  } else {
    return CTC_STATUS_INVALID_VALUE;
  }
}

ctcStatus_t get_workspace_size(const int* const label_lengths,
                               const int* const input_lengths,
                               int alphabet_size,
                               int minibatch,
                               ctcOptions options,
                               size_t* size_bytes) {
  if (label_lengths == nullptr || input_lengths == nullptr ||
      size_bytes == nullptr || alphabet_size <= 0 || minibatch <= 0)
    return CTC_STATUS_INVALID_VALUE;

  // This is the max of all S and T for all examples in the minibatch.
  int maxL = *std::max_element(label_lengths, label_lengths + minibatch);
  int maxT = *std::max_element(input_lengths, input_lengths + minibatch);

  const int S = 2 * maxL + 1;

  *size_bytes = 0;

  if (options.loc == CTC_GPU) {
    // GPU storage
    // nll_forward, nll_backward
    *size_bytes += 2 * sizeof(float) * minibatch;

    // repeats
    *size_bytes += sizeof(int) * minibatch;

    // label offsets
    *size_bytes += sizeof(int) * minibatch;

    // utt_length
    *size_bytes += sizeof(int) * minibatch;

    // label lengths
    *size_bytes += sizeof(int) * minibatch;

    // labels without blanks - overallocate for now
    *size_bytes += sizeof(int) * maxL * minibatch;

    // labels with blanks
    *size_bytes += sizeof(int) * S * minibatch;

    // alphas
    *size_bytes += sizeof(float) * S * maxT * minibatch;

    // denoms
    *size_bytes += sizeof(float) * maxT * minibatch;

    // probs (since we will pass in activations)
    *size_bytes += sizeof(float) * alphabet_size * maxT * minibatch;

  } else {
    // cpu can eventually replace all minibatch with
    // max number of concurrent threads if memory is
    // really tight

    // per minibatch memory
    size_t per_minibatch_bytes = 0;

    // output
    per_minibatch_bytes += sizeof(float) * alphabet_size;

    // alphas
    per_minibatch_bytes += sizeof(float) * S * maxT;

    // betas
    per_minibatch_bytes += sizeof(float) * S;

    // labels w/blanks, e_inc, s_inc
    per_minibatch_bytes += 3 * sizeof(int) * S;

    *size_bytes = per_minibatch_bytes * minibatch;

    // probs
    *size_bytes += sizeof(float) * alphabet_size * maxT * minibatch;
  }

  return CTC_STATUS_SUCCESS;
}
}
