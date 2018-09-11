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

#ifndef HL_BATCH_NORM_H_
#define HL_BATCH_NORM_H_

#include "hl_base.h"

/**
 * @brief   batch norm inferece.
 *
 * @param[in]   input         input data.
 * @param[out]  output        output data.
 * @param[in]   scale         batch normalization scale parameter (in original
 *                            paper scale is referred to as gamma).
 * @param[in]   bias          batch normalization bias parameter (in original
 *                            paper scale is referred to as beta).
 * @param[in]   estimatedMean
 * @param[in]   estimatedVar  The moving mean and variance
 *                            accumulated during the training phase are passed
 *                            as inputs here.
 * @param[in]   epsilon       Epsilon value used in the batch
 *                            normalization formula.
 */
extern void hl_batch_norm_cuda_inference(const real* input,
                                         real* output,
                                         const real* scale,
                                         const real* bias,
                                         const real* estimatedMean,
                                         const real* estimatedVar,
                                         const double epsilon,
                                         size_t batchSize,
                                         size_t channel,
                                         size_t height,
                                         size_t width);

#endif  // HL_BATCH_NORM_H_
