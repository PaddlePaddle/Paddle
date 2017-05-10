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

#ifndef HL_BATCH_TRANSPOSE_H_
#define HL_BATCH_TRANSPOSE_H_

#include "hl_base.h"

/**
 * @brief   Perform matrix transpose for each data in the batch.
 *
 * @param[in]   input     height * width elements in batch.
 * @param[out]  output    height * width elements in batch.
 * @param[in]   width     width of batch data.
 * @param[in]   height    height of batch data.
 * @param[in]   batchSize batch size
 *
 * @note    Both the inpt and output are arranged in batch-first
 *          order. Each batch has height * width data, which are
 *          arranged in height-first (or row-first) manner.
 */
extern void batchTranspose(
    const real* input, real* output, int width, int height, int batchSize);

#endif  // HL_BATCH_TRANSPOSE_H_
