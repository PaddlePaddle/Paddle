/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifndef HL_WARPCTC_WRAP_H_
#define HL_WARPCTC_WRAP_H_

#include "ctc.h"
#include "hl_base.h"

typedef ctcStatus_t hl_warpctc_status_t;
typedef ctcOptions hl_warpctc_options_t;

/**
 * @brief Init ctc options.
 *
 * @param[in]   blank     blank label used in ctc loss function.
 * @param[in]   useGpu    whether use gpu.
 * @param[out]  options   handle to store cpu or gpu informations.
 *
 */
extern void hl_warpctc_init(const size_t blank,
                            bool useGpu,
                            hl_warpctc_options_t* options);

/**
 * @brief Compute the connectionist temporal classification loss,
 *        and optionally compute the gradient with respect to the inputs.
 *
 * if batchGrad == nullptr
 *
 *    only compute the ctc loss.
 *
 * if batchGrad != nullptr
 *
 *    compute both ctc loss and gradient.
 *
 * @param[in]   batchInput      batch matrix of input probabilities,
 *                              in maxSequenceLength x numSequence x numClasses
 *                              (row-major) format.
 * @param[out]  batchGrad       batch matrix of gradient.
 * @param[in]   cpuLabels       labels always in CPU memory.
 * @param[in]   cpuLabelLengths length of all labels in CPU memory.
 * @param[in]   cpuInputLengths length of all sequences in CPU memory.
 * @param[in]   numClasses      number of possible output symbols.
 * @param[in]   numSequences    number of sequence.
 * @param[out]  cpuCosts        cost of each sequence in CPU memory.
 * @param[out]  workspace       workspace to store some temporary results.
 * @param[in]   options         handle to store cpu or gpu informations.
 *
 */
extern void hl_warpctc_compute_loss(const real* batchInput,
                                    real* batchGrad,
                                    const int* cpuLabels,
                                    const int* cpuLabelLengths,
                                    const int* cpuInputLengths,
                                    const size_t numClasses,
                                    const size_t numSequences,
                                    real* cpuCosts,
                                    void* workspace,
                                    hl_warpctc_options_t* options);

/**
 * @brief Compute the required workspace size.
 *        There is no memory allocated operations within warp-ctc.
 *
 * @param[in]   cpuLabelLengths length of all labels in CPU memory.
 * @param[in]   cpuInputLengths length of all sequences in CPU memory.
 * @param[in]   numClasses      number of possible output symbols.
 * @param[in]   numSequences    number of sequence.
 * @param[in]   options         handle to store cpu or gpu informations.
 * @param[out]  bytes           pointer to a scalar where the memory
 *                              requirement in bytes will be placed.
 *
 */
extern void hl_warpctc_get_workspace_size(const int* cpuLabelLengths,
                                          const int* cpuInputLengths,
                                          const size_t numClasses,
                                          const size_t numSequences,
                                          hl_warpctc_options_t* options,
                                          size_t* bytes);

#endif  // HL_WARPCTC_WRAP_H_
