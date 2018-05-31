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

#ifndef __PADDLE_CAPI_MATRIX_H__
#define __PADDLE_CAPI_MATRIX_H__

#include <stdbool.h>
#include <stdint.h>
#include "config.h"
#include "error.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Matrix functions. Return will be a paddle_error type.
 */
typedef void* paddle_matrix;

/**
 * @brief paddle_matrix_create Create a dense matrix
 * @param height matrix height.
 * @param width matrix width
 * @param useGpu use GPU of not
 * @return Matrix handler
 */
PD_API paddle_matrix paddle_matrix_create(uint64_t height,
                                          uint64_t width,
                                          bool useGpu);

/**
 * @brief paddle_matrix_create_sparse Create a sparse matrix.
 * @param height the matrix height.
 * @param width the matrix width.
 * @param nnz the number of non-zero elements.
 * @param isBinary is binary (either 1 or 0 in matrix) or not.
 * @param useGpu is using GPU or not.
 * @return paddle_matrix.
 * @note Mobile inference does not support this interface.
 */
PD_API paddle_matrix paddle_matrix_create_sparse(
    uint64_t height, uint64_t width, uint64_t nnz, bool isBinary, bool useGpu);

/**
 * @brief paddle_matrix_destroy Destroy a matrix.
 * @param mat
 * @return paddle_error
 */
PD_API paddle_error paddle_matrix_destroy(paddle_matrix mat);

/**
 * @brief paddle_matrix_set_row Set a row to matrix.
 * @param mat Target Matrix
 * @param rowID Index of row
 * @param rowArray Row data.
 * @return paddle_error
 */
PD_API paddle_error paddle_matrix_set_row(paddle_matrix mat,
                                          uint64_t rowID,
                                          paddle_real* rowArray);

/**
 * @brief paddle_matrix_set_value Set value to matrix.
 * @param mat Target Matrix
 * @param value Row data.
 * @return paddle_error
 * @note  value should contain enough element of data to init the mat
 */
PD_API paddle_error paddle_matrix_set_value(paddle_matrix mat,
                                            paddle_real* value);

/**
 * @brief PDMatGetRow Get raw row buffer from matrix
 * @param [in] mat Target matrix
 * @param [in] rowID Index of row.
 * @param [out] rawRowBuffer Row Buffer
 * @return paddle_error
 */
PD_API paddle_error paddle_matrix_get_row(paddle_matrix mat,
                                          uint64_t rowID,
                                          paddle_real** rawRowBuffer);

/**
 * @brief copy data from the matrix
 * @param [in] mat Target matrix
 * @param [out] result pointer to store the matrix data
 * @return paddle_error
 * @note the space of the result should allocated before invoke this API
 */
PD_API paddle_error paddle_matrix_get_value(paddle_matrix mat,
                                            paddle_real* result);
/**
 * @brief PDMatCreateNone Create None Matrix
 * @return
 */
PD_API paddle_matrix paddle_matrix_create_none();

/**
 * @brief PDMatGetShape get the shape of matrix
 * @param mat target matrix
 * @param height The height of matrix
 * @param width The width of matrix
 * @return paddle_error
 */
PD_API paddle_error paddle_matrix_get_shape(paddle_matrix mat,
                                            uint64_t* height,
                                            uint64_t* width);

/**
 * @brief paddle_matrix_sparse_copy_from Copy from a CSR format matrix
 * @param [out] mat output matrix
 * @param [in] rowArray row array. The array slices in column array.
 * @param [in] rowSize length of row array.
 * @param [in] colArray the column array. It means the non-zero element indices
 * in each row.
 * @param [in] colSize length of column array.
 * @param [in] valueArray the value array. It means the non-zero elemnt values.
 * NULL if the matrix is binary.
 * @param [in] valueSize length of value array. Zero if the matrix is binary.
 * @return paddle_error
 * @note Mobile inference does not support this interface.
 */
PD_API paddle_error paddle_matrix_sparse_copy_from(paddle_matrix mat,
                                                   int* rowArray,
                                                   uint64_t rowSize,
                                                   int* colArray,
                                                   uint64_t colSize,
                                                   float* valueArray,
                                                   uint64_t valueSize);

#ifdef __cplusplus
}
#endif
#endif
