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

#ifndef HL_SPARSE_STUB_H_
#define HL_SPARSE_STUB_H_

#include "hl_sparse.h"

inline void hl_malloc_sparse_matrix(hl_sparse_matrix_s *A_d,
                                    hl_matrix_format_t format,
                                    hl_matrix_value_t value_type,
                                    int dimM,
                                    int dimN,
                                    int nnz) {}

inline void hl_free_sparse_matrix(hl_sparse_matrix_s A_d) {}

inline void hl_construct_sparse_matrix(hl_sparse_matrix_s *A_d,
                                       void *dest_d,
                                       size_t size,
                                       hl_matrix_format_t format,
                                       hl_matrix_value_t value_type,
                                       int dimM,
                                       int dimN,
                                       int nnz) {}

inline void hl_construct_sparse_matrix(hl_sparse_matrix_s *A_d,
                                       real *value_d,
                                       int *rows_d,
                                       int *cols_d,
                                       hl_matrix_format_t format,
                                       hl_matrix_value_t value_type,
                                       int dimM,
                                       int dimN,
                                       int nnz) {}

inline void hl_destruct_sparse_matrix(hl_sparse_matrix_s A_d) {}

inline void hl_memcpy_csr_matrix(hl_sparse_matrix_s csr_matrix,
                                 real *csr_val,
                                 int *csr_row,
                                 int *csr_col,
                                 hl_stream_t stream) {}

inline void hl_memcpy_csc_matrix(hl_sparse_matrix_s csc_matrix,
                                 real *csc_val,
                                 int *csc_row,
                                 int *csc_col,
                                 hl_stream_t stream) {}

inline void hl_memcpy_sparse_matrix(hl_sparse_matrix_s dst,
                                    hl_sparse_matrix_s src,
                                    hl_stream_t stream) {}

inline void hl_matrix_csr2dense(hl_sparse_matrix_s A_d,
                                real *C_d,
                                int dimM,
                                int dimN) {}

inline void hl_matrix_csc2dense(hl_sparse_matrix_s A_d,
                                real *C_d,
                                int dimM,
                                int dimN) {}

inline void hl_matrix_csr_mul_dense(hl_sparse_matrix_s A_d,
                                    hl_trans_op_t transa,
                                    real *B_d,
                                    hl_trans_op_t transb,
                                    real *C_d,
                                    int dimM,
                                    int dimN,
                                    int dimK,
                                    real alpha,
                                    real beta) {}

inline void hl_matrix_csc_mul_dense(hl_sparse_matrix_s A_d,
                                    hl_trans_op_t transa,
                                    real *B_d,
                                    hl_trans_op_t transb,
                                    real *C_d,
                                    int dimM,
                                    int dimN,
                                    int dimK,
                                    real alpha,
                                    real beta) {}

inline void hl_matrix_dense_mul_csc(real *A_d,
                                    hl_trans_op_t transa,
                                    hl_sparse_matrix_s B_d,
                                    hl_trans_op_t transb,
                                    real *C_d,
                                    int dimM,
                                    int dimN,
                                    int dimK,
                                    real alpha,
                                    real beta) {}

inline void hl_sparse_matrix_mul(real *A_d,
                                 hl_trans_op_t transa,
                                 real *B_d,
                                 hl_trans_op_t transb,
                                 hl_sparse_matrix_s C_d,
                                 int dimM,
                                 int dimN,
                                 int dimK,
                                 real alpha,
                                 real beta) {}

inline void hl_matrix_dense_mul_csr(real *A_d,
                                    hl_trans_op_t transa,
                                    hl_sparse_matrix_s B_d,
                                    hl_trans_op_t transb,
                                    real *C_d,
                                    int dimM,
                                    int dimN,
                                    int dimK,
                                    real alpha,
                                    real beta) {}

inline void hl_memcpy_from_csc_matrix(real *csc_val,
                                      size_t val_size,
                                      int *csc_row,
                                      size_t row_size,
                                      int *csc_col,
                                      size_t col_size,
                                      hl_sparse_matrix_s csc_matrix,
                                      hl_stream_t stream) {}

inline void hl_memcpy_from_csr_matrix(real *csr_val,
                                      size_t val_size,
                                      int *csr_row,
                                      size_t row_size,
                                      int *csr_col,
                                      size_t col_size,
                                      hl_sparse_matrix_s csr_matrix,
                                      hl_stream_t stream) {}

inline void hl_sparse_matrix_column_sum(
    real *A_d, hl_sparse_matrix_s B_d, int dimM, int dimN, real scale) {}

inline void hl_matrix_csr_column_sum(
    real *A_d, hl_sparse_matrix_s B_d, int dimM, int dimN, real scale) {}

inline void hl_sparse_matrix_add_bias(hl_sparse_matrix_s A_d,
                                      real *B_d,
                                      real scale) {}

inline void hl_matrix_csr_add_bias(hl_sparse_matrix_s A_d,
                                   real *B_d,
                                   real scale) {}

inline void hl_sparse_matrix_add_dense(hl_sparse_matrix_s A_d,
                                       real *B_d,
                                       int dimM,
                                       int dimN,
                                       real alpha,
                                       real beta) {}

inline void hl_matrix_csr_add_dense(hl_sparse_matrix_s A_d,
                                    real *B_d,
                                    int dimM,
                                    int dimN,
                                    real alpha,
                                    real beta) {}

inline int *hl_sparse_matrix_get_rows(hl_sparse_matrix_s sMat) { return NULL; }

inline int *hl_sparse_matrix_get_cols(hl_sparse_matrix_s sMat) { return NULL; }

inline real *hl_sparse_matrix_get_value(hl_sparse_matrix_s sMat) {
  return NULL;
}

#endif  // HL_SPARSE_STUB_H_
