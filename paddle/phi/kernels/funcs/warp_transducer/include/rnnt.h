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

/** \file rnnt.h
 * Contains a simple C interface to call fast CPU and GPU based computation
 * of the RNNT loss.
 */

#pragma once

#ifdef __cplusplus
#include <cstddef>
extern "C" {
#endif

#ifdef PADDLE_WITH_HIP
// forward declare of HIP typedef to avoid needing to pull in HIP headers
typedef struct ihipStream_t* GPUstream;
#else
// forward declare of CUDA typedef to avoid needing to pull in CUDA headers
typedef struct CUstream_st* GPUstream;
#endif

typedef enum {
  RNNT_STATUS_SUCCESS = 0,
  RNNT_STATUS_MEMOPS_FAILED = 1,
  RNNT_STATUS_INVALID_VALUE = 2,
  RNNT_STATUS_EXECUTION_FAILED = 3,
  RNNT_STATUS_UNKNOWN_ERROR = 4
} rnntStatus_t;

/** Returns a single integer which specifies the API version of the warprnnt
 * library */
int get_warprnnt_version();

/** Returns a string containing a description of status that was passed in
 *  \param[in] status identifies which string should be returned
 *  \return C style string containing the text description
 *  */
const char* rnntGetStatusString(rnntStatus_t status);

typedef enum { RNNT_CPU = 0, RNNT_GPU = 1 } rnntComputeLocation;

/** Structure used for options to the RNNT compution.  Applications
 *  should zero out the array using memset and sizeof(struct
 *  rnntOptions) in C or default initialization (e.g. 'rnntOptions
 *  options{};' or 'auto options = rnntOptions{}') in C++ to ensure
 *  forward compatibility with added options. */
struct rnntOptions {
  /// indicates where the rnnt calculation should take place {RNNT_CPU |
  /// RNNT_GPU}
  rnntComputeLocation loc;

  /// The maximum number of threads that can be used
  unsigned int num_threads;

  /// used when loc == RNNT_GPU, which stream the kernels should be launched in
  GPUstream stream;

  /// the label value/index that the RNNT calculation should use as the blank
  /// label
  int blank_label;

  /// the maximum length of time steps
  int maxT;

  /// the maximum length of label sequence
  int maxU;

  /// memory structure
  bool batch_first;

  float fastemit_lambda;
};

/** Compute the RNN Transducer loss between a sequence
 *  of probabilities and a ground truth labeling.  Optionally compute the
 *  gradient with respect to the inputs.
 * \param [in] activations pointer to the activations in either CPU or GPU
 *             addressable memory, depending on info.  We assume a fixed
 *             memory layout for this 4 dimensional tensor, which has dimension
 *             (b, t, u, v), where b is the minibatch index, t is the time
 * index, u is the prediction index, and v indexes over probabilities of each
 * symbol in the alphabet. The memory layout is (b, t, u, v) in C order (slowest
 * to fastest changing index, aka row-major). We also assume strides are equal
 * to dimensions - there is no padding between dimensions. More precisely,
 * element (b, t, u, v), for a problem with mini_batch examples in the mini
 * batch, and alphabet_size symbols in the alphabet, is located at:
 *             activations[((b * max_time + t) * max_u + u) * alphabet_size + v]
 * \param [out] gradients if not NULL, then gradients are computed.  Should be
 *              allocated in the same memory space as probs and memory
 *              ordering is identical.
 * \param [in]  flat_labels Always in CPU memory.  A concatenation
 *              of all the labels for the minibatch (should be padded to same
 * length)
 * \param [in]  label_lengths Always in CPU memory. The length of each
 * label for each example in the minibatch.
 * \param [in]  input_lengths Always in
 * CPU memory.  The number of time steps for each sequence in the minibatch.
 * \param [in]  alphabet_size The number of possible output symbols.  There
 *              should be this many probabilities for each time step.
 * \param [in]  mini_batch How many examples in a minibatch.
 * \param [in]  maxT maximum length along time dimension.
 * \param [in]  maxU maximum length along prediction dimention.
 * \param [out] costs Always in CPU memory.  The cost of each example in the
 *              minibatch.
 * \param [in,out] workspace In same memory space as probs. Should be of
 *                 size requested by get_rnnt_workspace_size.
 * \param [in]  options see struct rnntOptions
 *
 *  \return Status information
 *
 * */
rnntStatus_t compute_rnnt_loss(const float* const activations,
                               float* gradients,
                               const int* const flat_labels,
                               const int* const label_lengths,
                               const int* const input_lengths,
                               int alphabet_size,
                               int minibatch,
                               float* costs,
                               void* workspace,
                               rnntOptions options);

rnntStatus_t compute_rnnt_loss_fp64(const double* const activations,
                                    double* gradients,
                                    const int* const flat_labels,
                                    const int* const label_lengths,
                                    const int* const input_lengths,
                                    int alphabet_size,
                                    int minibatch,
                                    double* costs,
                                    void* workspace,
                                    rnntOptions options);

/** For a given set of max sequence length and minibatch size return the
 *required workspace size. This will need to be allocated in the same memory
 *space as your probabilities. \param [in]  mini_batch How many examples in a
 *minibatch. \param [in]  info see struct rnntOptions \param [out] size_bytes is
 *pointer to a scalar where the memory requirement in bytes will be placed. This
 *memory should be allocated at the same place, CPU or GPU, that the probs are
 *in
 *
 *  \return Status information
 **/
rnntStatus_t get_rnnt_workspace_size(int maxT,
                                     int maxU,
                                     int minibatch,
                                     bool gpu,
                                     size_t* size_bytes,
                                     size_t dtype_size = sizeof(float));

#ifdef __cplusplus
}
#endif
