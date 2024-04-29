// Copyright (c) 2021 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
//! \file This file defines some C APIs to trigger CBLAS methods.
#include "paddle/cinn/runtime/cinn_runtime.h"

#ifdef CINN_WITH_MKL_CBLAS
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#else
#include <cblas.h>
#endif

// define some C APIs
extern "C" {

/**
 * \brief Do GEMM on buffer A and B and write result to buffer C.
 * We pass the \param M, \param N, \param K although the shape can retrieve from
 * cinn_buffer_t because the size of a matrix not equals the shape of a buffer
 * it is stored.
 * @param alpha The scaling factor of the product of A and B
 * @param M Number of the rows of A
 * @param N the number of the columns in both B and C
 * @param K the number of columns of A
 * @param ta whether to transpose A
 * @param tb whether to transpose B
 * @param lda The size of the first dimension of A
 * @param ldb The size of the first dimension of B
 * @param ldc The size of the first dimension of C
 * @param beta The scaling factor of C
 * @param A The matrix A
 * @param B The matrix B
 * @param C The output matrix
 */
void cinn_cpu_mkl_gemm_fp32(float alpha,
                            int M,
                            int N,
                            int K,
                            bool ta,
                            bool tb,
                            int lda,
                            int ldb,
                            int ldc,
                            float beta,
                            cinn_buffer_t* A,
                            cinn_buffer_t* B,
                            cinn_buffer_t* C);

/**
 * \brief Do GEMM on buffer A and B and write result to buffer C.
 * We pass the \param M, \param N, \param K although the shape can retrieve from
 * cinn_buffer_t because the size of a matrix not equals the shape of a buffer
 * it is stored.
 * @param alpha The scaling factor of the product of A and B
 * @param batch_size the batch size of A and B
 * @param M Number of the rows of A
 * @param N the number of the columns in both B and C
 * @param K the number of columns of A
 * @param ta whether to transpose A
 * @param tb whether to transpose B
 * @param lda The size of the first dimension of A
 * @param ldb The size of the first dimension of B
 * @param ldc The size of the first dimension of C
 * @param a_stride The stride of A(number of elements, not bytes) between
 * batches
 * @param b_stride The stride of B(number of elements, not bytes) between
 * batches
 * @param c_stride The stride of C(number of elements, not bytes) between
 * batches
 * @param beta The scaling factor of C
 * @param A The matrix A
 * @param B The matrix B
 * @param C The output matrix
 */
void cinn_cpu_mkl_gemm_batch_fp32(float alpha,
                                  int batch_size,
                                  int M,
                                  int N,
                                  int K,
                                  bool ta,
                                  bool tb,
                                  int lda,
                                  int ldb,
                                  int ldc,
                                  int a_stride,
                                  int b_stride,
                                  int c_stride,
                                  float beta,
                                  cinn_buffer_t* A,
                                  cinn_buffer_t* B,
                                  cinn_buffer_t* C);

void cinn_call_cholesky_host(
    void* v_args, int num_args, int batch_size, int m, bool upper);
}  // extern "C"
