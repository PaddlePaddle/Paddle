/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifndef HL_TOP_K_H_
#define HL_TOP_K_H_

#include "hl_base.h"

/**
 * @brief   find top k element.
 *
 * @param[out]  topVal         top k element.
 * @param[in]   ldv            leading dimension of topVal.
 * @param[out]  topIds         top k index.
 * @param[in]   src            input value.
 * @param[in]   lds            leading dimension of src.
 * @param[in]   dim            width of input value.
 * @param[in]   beamSize       beam size.
 * @param[in]   numSamples     height of input value.
 *
 */
extern void hl_matrix_top_k(real* topVal,
                            int ldv,
                            int* topIds,
                            real* src,
                            int lds,
                            int dim,
                            int beamSize,
                            int numSamples);

/**
 * @brief   find top k element for each row in sparse matrix.
 *
 * @param[out]  topVal         top k element.
 * @param[in]   ldv            leading dimension of topVal.
 * @param[out]  topIds         top k index.
 * @param[in]   src            sparse matrix.
 * @param[in]   beamSize       beam size.
 * @param[in]   numSamples     height of input value.
 *
 * @note    Only support HL_SPARSE_CSR format.
 */
extern void hl_sparse_matrix_top_k(real* topVal,
                                   int ldv,
                                   int* topIds,
                                   hl_sparse_matrix_s src,
                                   int beamSize,
                                   int numSamples);

/**
 * @brief   Matrix classification error.
 *
 * @param[out]  topVal         top k element.
 * @param[in]   ldv            leading dimension of topVal.
 * @param[out]  topIds         top k index.
 * @param[in]   src            input value.
 * @param[in]   lds            leading dimension of src.
 * @param[in]   dim            width of input value.
 * @param[in]   topkSize       size of top k element.
 * @param[in]   numSamples     height of input value.
 * @param[in]   label          ground truth label.
 * @param[out]  recResult      top-k classification error.
 *
 */
extern void hl_matrix_classification_error(real* topVal,
                                           int ldv,
                                           int* topIds,
                                           real* src,
                                           int lds,
                                           int dim,
                                           int topkSize,
                                           int numSamples,
                                           int* label,
                                           real* recResult);

#endif  // HL_TOP_K_H_
