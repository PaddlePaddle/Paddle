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

#ifndef HL_CPU_MATRIX_KERNEL_CUH_
#define HL_CPU_MATRIX_KERNEL_CUH_

#include <stdio.h>
#include "hl_base.h"

#ifndef __CUDA_ARCH__
#include "hl_cpu_matrix_kernel_detail.cuh"
#endif

/**
 * @brief   cpu element wise unary operator.
 */
template <class T, class Op>
void hl_cpu_apply_unary_op(Op op, T* A_h, int dimM, int dimN, int lda) {
  for (int i = 0; i < dimM; i ++) {
    for (int j = 0; j < dimN; j++) {
      op.cpuOperator(A_h[i*lda + j]);
    }
  }
}

/**
 * @brief   cpu element wise binary operator.
 */
template <class T, class Op, bool BAsRowVector, bool BAsColVector>
void hl_cpu_apply_binary_op(Op op,
                            T* A_h,
                            T* B_h,
                            int dimM,
                            int dimN,
                            int lda,
                            int ldb) {
  for (int i = 0; i < dimM; i ++) {
    for (int j = 0; j < dimN; j++) {
      if (BAsRowVector == 0 && BAsColVector == 0) {
        op.cpuOperator(A_h[i * lda + j], B_h[i * ldb + j]);
      } else if (BAsRowVector == 1 && BAsColVector == 0) {
        op.cpuOperator(A_h[i * lda + j], B_h[j]);
      } else if (BAsRowVector == 0 && BAsColVector == 1) {
        op.cpuOperator(A_h[i * lda + j], B_h[i * ldb]);
      } else {
        op.cpuOperator(A_h[i * lda + j], B_h[0]);
      }
    }
  }
}

/**
 * @brief   cpu element wise ternary operator.
 */
template <class T, class Op, bool CAsRowVector, bool CAsColVector>
void hl_cpu_apply_ternary_op(Op op,
                             T* A_h,
                             T* B_h,
                             T* C_h,
                             int dimM,
                             int dimN,
                             int lda,
                             int ldb,
                             int ldc) {
  for (int i = 0; i < dimM; i ++) {
    for (int j = 0; j < dimN; j++) {
      if (CAsRowVector == 0 && CAsColVector == 0) {
        op.cpuOperator(A_h[i*lda + j], B_h[i*ldb + j], C_h[i*ldc + j]);
      } else if (CAsRowVector == 1 && CAsColVector == 0) {
        op.cpuOperator(A_h[i*lda + j], B_h[i*ldb + j], C_h[j]);
      } else if (CAsRowVector == 0 && CAsColVector == 1) {
        op.cpuOperator(A_h[i*lda + j], B_h[i*ldb + j], C_h[i*ldc]);
      } else {
        op.cpuOperator(A_h[i*lda + j], B_h[i*ldb + j], C_h[0]);
      }
    }
  }
}

/**
 * @brief   cpu element wise quaternary operator.
 */
template <class T, class Op>
void hl_cpu_apply_quaternary_op(Op op,
                                T* A_h,
                                T* B_h,
                                T* C_h,
                                T* D_h,
                                int dimM,
                                int dimN,
                                int lda,
                                int ldb,
                                int ldc,
                                int ldd) {
  for (int i = 0; i < dimM; i ++) {
    for (int j = 0; j < dimN; j++) {
      op.cpuOperator(A_h[i*lda + j],
                     B_h[i*ldb + j],
                     C_h[i*ldc + j],
                     D_h[i*ldd + j]);
    }
  }
}

template <class Agg, class Op, class Saver>
void hl_cpu_matrix_row_op(Agg agg, Op op, Saver sv,
                          int dimM, int dimN,
                          real *dst, int ld,
                          real *A, int lda) {
#ifndef __CUDA_ARCH__
  if (!Agg::sse || !Op::sse || !Saver::sse) {
    hl_matrix_row_op(agg, op, sv, dimM, dimN, dst, ld, A, lda);
  } else {
    if (hl_check_align(A) && hl_check_align(lda*sizeof(real))) {
      hl_sse_matrix_row_op(agg, op, sv, dimM, dimN, dst, ld, A, lda);
    } else {
      hl_matrix_row_op(agg, op, sv, dimM, dimN, dst, ld, A, lda);
    }
  }
#endif
}

template <class Agg, class Op, class Saver>
void hl_cpu_matrix_row_op(Agg agg, Op op, Saver sv,
                          int dimM, int dimN,
                          real *dst, int ld,
                          real *A, int lda,
                          real *B, int ldb) {
#ifndef __CUDA_ARCH__
  if (!Agg::sse || !Op::sse || !Saver::sse) {
    hl_matrix_row_op(agg, op, sv, dimM, dimN, dst, ld, A, lda, B, ldb);
  } else {
    if (hl_check_align(A) && hl_check_align(lda*sizeof(real))
      && hl_check_align(B) && hl_check_align(ldb*sizeof(real))) {
      hl_sse_matrix_row_op(
        agg, op, sv, dimM, dimN, dst, ld, A, lda, B, ldb);
    } else {
      hl_matrix_row_op(agg, op, sv, dimM, dimN, dst, ld, A, lda, B, ldb);
    }
  }
#endif
}

template <class Agg, class Op, class Saver>
void hl_cpu_matrix_column_op(Agg agg, Op op, Saver sv,
                             int dimM, int dimN,
                             real *dst,
                             real *A, int lda) {
#ifndef __CUDA_ARCH__
  if (!Agg::sse || !Op::sse || !Saver::sse) {
    hl_matrix_column_op(agg, op, sv, dimM, dimN, dst, A, lda);
  } else {
    if (hl_check_align(A) && hl_check_align(lda*sizeof(real))
      && hl_check_align(dst)) {
      hl_sse_matrix_column_op(agg, op, sv, dimM, dimN, dst, A, lda);
    } else {
      hl_matrix_column_op(agg, op, sv, dimM, dimN, dst, A, lda);
    }
  }
#endif
}

template <class Agg, class Op, class Saver>
void hl_cpu_matrix_column_op(Agg agg, Op op, Saver sv,
                             int dimM, int dimN,
                             real *dst,
                             real *A, int lda,
                             real *B, int ldb) {
#ifndef __CUDA_ARCH__
  if (!Agg::sse || !Op::sse || !Saver::sse) {
    hl_matrix_column_op(agg, op, sv, dimM, dimN, dst, A, lda, B, ldb);
  } else {
    if (hl_check_align(A) && hl_check_align(lda*sizeof(real))
      && hl_check_align(B) && hl_check_align(ldb*sizeof(real))
      && hl_check_align(dst)) {
      hl_sse_matrix_column_op(
        agg, op, sv, dimM, dimN, dst, A, lda, B, ldb);
    } else {
      hl_matrix_column_op(agg, op, sv, dimM, dimN, dst, A, lda, B, ldb);
    }
  }
#endif
}

#endif /* HL_CPU_MATRIX_KERNEL_CUH_ */
