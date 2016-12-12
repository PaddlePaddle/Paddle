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

namespace paddle {

/**
 * this function is for SparseMatrix initialization except data.
 * It generates a random non-zero pattern for a sparse matrix.
 *
 * if format is SPARSE_CSC,
 *    major is column start index and minor is row index
 *    for each non zero value.
 * else
 *    major is row start index and minor is col
 *    index for each non zero value.
 *
 * Initialize minor value according to major value.
 *
 * For example, A is 5*3  CSC matrix, nnz is 10, then
 *
 * @code
 *   cols[i] = i * nnz / 3
 *   cols=[0, 3, 6, 10]
 * @endcode
 *
 * for column i, we randomly select cols[i+1] - cols[i] rows
 * as non zero number row index.
 *
 * rows is [1, 3, 4, 0, 2, 4, 1, 2, 3, 4]
 */
void sparseRand(
    int* major, int* minor, int nnz, int majorLen, int minorMax, bool useGpu);

/**
 * Calculate output size based on caffeMode_.
 * - input(+padding): 0123456789
 * - imageSize(+padding) = 10;
 * - filterSize = 3;
 * - stride = 2;
 * - caffeMode is true:
     - output: (012), (234), (456), (678)
     - outputSize = 4;
 * - caffeMode is false:
 *   - output: (012), (234), (456), (678), (9)
 *   - outputSize = 5;
 */
int outputSize(
    int imageSize, int filterSize, int padding, int stride, bool caffeMode);

/**
 * Calculate image size based on output size and caffeMode_.
 * It is the reverse function of outputSize()
 */
int imageSize(
    int outputSize, int filterSize, int padding, int stride, bool caffeMode);

}  // namespace paddle
