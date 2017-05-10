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

#pragma once

#include <gtest/gtest.h>
#include "paddle/math/Matrix.h"

namespace paddle {

std::string randStr(const int len);

inline int uniformRandom(int n) { return n == 0 ? 0 : rand() % n; }

inline bool approximatelyEqual(float a, float b, float epsilon) {
  return fabs(a - b) <= ((fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}

MatrixPtr makeRandomSparseMatrix(size_t height,
                                 size_t width,
                                 bool withValue,
                                 bool useGpu,
                                 bool equalNnzPerSample = false);

/**
 * @brief generate sequenceStartPositions for INPUT_SEQUENCE_DATA,
 *        INPUT_HASSUB_SEQUENCE_DATA and INPUT_SEQUENCE_LABEL
 *
 * @param batchSize                      batchSize
 *        sequenceStartPositions[out] generation output
 */
void generateSequenceStartPositions(size_t batchSize,
                                    IVectorPtr& sequenceStartPositions);

void generateSequenceStartPositions(size_t batchSize,
                                    ICpuGpuVectorPtr& sequenceStartPositions);

/**
 * @brief generate subSequenceStartPositions for INPUT_HASSUB_SEQUENCE_DATA
 *        according to sequenceStartPositions
 *
 * @param sequenceStartPositions[in]     input
 *        subSequenceStartPositions[out] generation output
 */
void generateSubSequenceStartPositions(const IVectorPtr& sequenceStartPositions,
                                       IVectorPtr& subSequenceStartPositions);

void generateSubSequenceStartPositions(
    const ICpuGpuVectorPtr& sequenceStartPositions,
    ICpuGpuVectorPtr& subSequenceStartPositions);

/**
 * @brief generate cpuSequenceDims for INPUT_SEQUENCE_MDIM_DATA according to
 *        sequenceStartPositions
 *
 * @param sequenceStartPositions[in]     input
 *        cpuSequenceDims[out]              generation output
 */
void generateMDimSequenceData(const IVectorPtr& sequenceStartPositions,
                              IVectorPtr& cpuSequenceDims);
void generateMDimSequenceData(const ICpuGpuVectorPtr& sequenceStartPositions,
                              IVectorPtr& cpuSequenceDims);

void checkMatrixEqual(const MatrixPtr& a, const MatrixPtr& b);

void checkVectorEqual(const IVectorPtr& a, const IVectorPtr& b);
}  // namespace paddle
