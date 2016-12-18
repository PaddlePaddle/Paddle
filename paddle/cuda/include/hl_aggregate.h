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

#ifndef HL_AGGREGATE_H_
#define HL_AGGREGATE_H_

#include "hl_base.h"

/**
 * @brief   Calculate the sum of each row of the matrix A_d.
 *
 * @param[in]    A_d     input matrix (M x N).
 * @param[out]   C_d     output matrix (M x 1).
 * @param[in]    dimM    matrix height.
 * @param[in]    dimN    matrix width.
 *
 */
extern void hl_matrix_row_sum(real *A_d, real *C_d, int dimM, int dimN);

/**
 * @brief   Calculate the maximum value of each row of the matrix A_d.
 *
 * @param[in]   A_d     input matrix (M x N).
 * @param[out]  C_d     output matrix (M x 1).
 * @param[in]   dimM    matrix height.
 * @param[in]   dimN    matrix width.
 *
 */
extern void hl_matrix_row_max(real *A_d, real *C_d, int dimM, int dimN);

/**
 * @brief   Calculate the minimum value of each row of the matrix A_d.
 *
 * @param[in]   A_d     input matrix (M x N).
 * @param[out]  C_d     output matrix (M x 1).
 * @param[in]   dimM    matrix height.
 * @param[in]   dimN    matrix width.
 *
 */
extern void hl_matrix_row_min(real *A_d, real *C_d, int dimM, int dimN);

/**
 * @brief   Calculate the sum of each column of the matrix A_d.
 *
 * @param[in]   A_d     input matrix (M x N).
 * @param[out]  C_d     output Matrix (1 x N).
 * @param[in]   dimM    matrix height.
 * @param[in]   dimN    matrix width.
 *
 */
extern void hl_matrix_column_sum(real *A_d, real *C_d, int dimM, int dimN);

/**
 * @brief   Calculate the maximum value of each column of the matrix A_d.
 *
 * @param[in]   A_d     input matrix (M x N).
 * @param[out]  C_d     output matrix (1 x N).
 * @param[in]   dimM    matrix height.
 * @param[in]   dimN    matrix width.
 *
 */
extern void hl_matrix_column_max(real *A_d, real *C_d, int dimM, int dimN);

/**
 * @brief   Calculate the minimum value of each column of the matrix A_d.
 *
 * @param[in]   A_d     input matrix (M x N).
 * @param[out]  C_d     output matrix (1 x N).
 * @param[in]   dimM    matrix height.
 * @param[in]   dimN    matrix width.
 *
 */
extern void hl_matrix_column_min(real *A_d, real *C_d, int dimM, int dimN);

/**
 * @brief   C_h = sum(A_d[i]).
 *
 * @param[in]   A_d     input(m).
 * @param[out]  C_h     output(host memory).
 * @param[in]   dimM    size of vector.
 *
 */
extern void hl_vector_sum(real *A_d, real *C_h, int dimM);

/**
 * @brief   C_h = sum(abs(A_d[i])).
 *
 * @param[in]   A_d     input(m).
 * @param[out]  C_h     output(host memory).
 * @param[in]   dimM    size of vector.
 *
 */
extern void hl_vector_abs_sum(real *A_d, real *C_h, int dimM);

#endif /* HL_AGGREGATE_H_ */
