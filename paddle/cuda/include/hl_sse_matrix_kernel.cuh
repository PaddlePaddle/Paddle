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


#ifndef HL_SSE_MATRIX_KERNEL_CUH_
#define HL_SSE_MATRIX_KERNEL_CUH_

#include "hl_matrix_type.cuh"

#define VECTOR_SIZE     16

#ifndef PADDLE_TYPE_DOUBLE
/* number of float in vector */
#define     VECTOR_LEN      4
#define     VECTOR_SET      _mm_set_ps1
#else
#if   defined(__APPLE__) || defined(__OSX__)
#define     _mm_set_pd1     _mm_set1_pd
#endif
/* number of double in vector */
#define     VECTOR_LEN      2
#define     VECTOR_SET      _mm_set_pd1
#endif

inline bool hl_check_align(size_t size) {
  return !(size & (VECTOR_SIZE - 1));
}

inline bool hl_check_align(void *ptr) {
  return hl_check_align(reinterpret_cast<size_t>(ptr));
}

#ifndef PADDLE_TYPE_DOUBLE
template <class Agg>
inline real hl_agg_op(Agg agg, vecType mm) {
  __m128 lo = _mm_unpacklo_ps(mm, mm);
  __m128 hi = _mm_unpackhi_ps(mm, mm);
  __m128 tmp1 = agg.vecOp(lo, hi);
  __m128 tmp2 = _mm_movehl_ps(tmp1, tmp1);
  __m128 ret = agg.vecOp(tmp1, tmp2);

  return _mm_cvtss_f32(ret);
}
#else
template <class Agg>
inline real hl_agg_op(Agg agg, vecType mm) {
  __m128d lo = _mm_unpacklo_pd(mm, mm);
  __m128d hi = _mm_unpackhi_pd(mm, mm);
  __m128d ret = agg.vecOp(lo, hi);

  return _mm_cvtsd_f64(ret);
}
#endif

template <class Agg, class Op, class Saver>
void hl_sse_matrix_row_op(Agg agg, Op op, Saver sv,
                          int dimM, int dimN,
                          real *dst, int ld,
                          real *A, int lda) {
  for (int i = 0; i < dimM; i++, A += lda) {
    vecType mm = VECTOR_SET(agg.init());
    vecType *a = (vecType*)(A);
    for (int j = 0; j < dimN / VECTOR_LEN; j++, a++) {
        mm = agg.vecOp(mm, op.vecOp(*a));
    }

    int rem = dimN % VECTOR_LEN;
    if (rem) {
      real tmp = hl_agg_op(agg, mm);
      real *a = A + (dimN / VECTOR_LEN) * VECTOR_LEN;
      for (int j = 0; j < rem; j++) {
          tmp = agg(tmp, op(a[j]));
      }
      dst[i*ld] = sv(dst[i*ld], tmp);
    } else {
        dst[i*ld] = sv(dst[i*ld], hl_agg_op(agg, mm));
    }
  }
}

template <class Agg, class Op, class Saver>
void hl_sse_matrix_row_op(Agg agg, Op op, Saver sv,
                          int dimM, int dimN,
                          real *dst, int ld,
                          real *A, int lda,
                          real *B, int ldb) {
  for (int i = 0; i < dimM; i++, A += lda, B += ldb) {
    vecType mm = VECTOR_SET(agg.init());
    vecType *a = (vecType*)(A);
    vecType *b = (vecType*)(B);
    for (int j = 0; j < dimN / VECTOR_LEN; j++, a++, b++) {
        mm = agg.vecOp(mm, op.vecOp(*a, *b));
    }

    int rem = dimN % VECTOR_LEN;
    if (rem) {
      real tmp = hl_agg_op(agg, mm);
      real *a = A + (dimN / VECTOR_LEN) * VECTOR_LEN;
      real *b = B + (dimN / VECTOR_LEN) * VECTOR_LEN;
      for (int j = 0; j < rem; j++) {
          tmp = agg(tmp, op(a[j], b[j]));
      }
      dst[i*ld] = sv(dst[i*ld], tmp);
    } else {
        dst[i*ld] = sv(dst[i*ld], hl_agg_op(agg, mm));
    }
  }
}

template <class Agg, class Op, class Saver>
void hl_matrix_column_op(Agg agg, Op op, Saver sv,
                         int dimM, int dimN,
                         real *dst,
                         real *A, int lda) {
  for (int j = 0; j < dimN; j++) {
    real tmp = agg.init();
    for (int i = 0; i < dimM; i++) {
        tmp = agg(tmp, op(A[i * lda + j]));
    }
    dst[j] = sv(dst[j], tmp);
  }
}

template <class Agg, class Op, class Saver>
void hl_matrix_column_op(Agg agg, Op op, Saver sv,
                         int dimM, int dimN,
                         real *dst,
                         real *A, int lda,
                         real *B, int ldb) {
  for (int j = 0; j < dimN; j++) {
    real tmp = agg.init();
    for (int i = 0; i < dimM; i++) {
        tmp = agg(tmp, op(A[i * lda + j], B[i * ldb + j]));
    }
    dst[j] = sv(dst[j], tmp);
  }
}

/*
 * MaxRow greater than or equal dimN
 * dimN is multiples of VECTOR_LEN
 * so rem <= MaxRow / VECTOR_LEN
 */
template <int MaxRow, class Agg, class Op, class Saver>
void hl_sse_column_op_with_rem(Agg agg, Op op, Saver sv,
                               int dimM, int dimN,
                               real *dst,
                               real *A, int lda) {
  vecType mm[MaxRow / VECTOR_LEN];
  for (int n = 0; n < MaxRow / VECTOR_LEN; n++) {
    mm[n] = VECTOR_SET(agg.init());
  }

  for (int i = 0; i < dimM; i++) {
    vecType *a = (vecType*)(A + i * lda);
    for (int n = 0; n < dimN / VECTOR_LEN; n++) {
      mm[n] = agg.vecOp(mm[n], op.vecOp(a[n]));
    }
  }

  vecType *result = (vecType*)(dst);
  for (int n = 0; n < dimN / VECTOR_LEN; n++) {
    result[n] = sv.vecOp(result[n], mm[n]);
  }

  int rem = dimN % VECTOR_LEN;
  if (rem) {
    A += (dimN / VECTOR_LEN) * VECTOR_LEN;
    dst += (dimN / VECTOR_LEN) * VECTOR_LEN;
    hl_matrix_column_op(agg, op, sv, dimM, rem, dst, A, lda);
  }
}

/*
 * dimN is multiples of VECTOR_LEN
 * dimN greater than Step
 */
template <int Step, class Agg, class Op, class Saver>
void hl_sse_matrix_column_op(Agg agg, Op op, Saver sv,
                             int dimM, int dimN,
                             real *dst,
                             real *A, int lda) {
  for (int j = 0; j < dimN / Step; j++, dst += Step, A += Step) {
    vecType mm[Step / VECTOR_LEN];
    for (int n = 0; n < Step / VECTOR_LEN; n++) {
      mm[n] = VECTOR_SET(agg.init());
    }

    for (int i = 0; i < dimM; i++) {
      vecType *a = (vecType*)(A + i * lda);
      for (int n = 0; n < Step / VECTOR_LEN; n++) {
        mm[n] = agg.vecOp(mm[n], op.vecOp(a[n]));
      }
    }

    vecType *result = (vecType*)(dst);
    for (int n = 0; n < Step / VECTOR_LEN; n++) {
      result[n] = sv.vecOp(result[n], mm[n]);
    }
  }

  int remRow = dimN % Step;
  if (remRow) {
    hl_sse_column_op_with_rem<Step>(agg, op, sv, dimM, remRow, dst, A, lda);
  }
}

template <class Agg, class Op, class Saver>
void hl_sse_matrix_column_op(Agg agg, Op op, Saver sv,
                             int dimM, int dimN,
                             real *dst,
                             real *A, int lda) {
  if (dimN <= 16) {
    hl_sse_matrix_column_op<16>(agg, op, sv, dimM, dimN, dst, A, lda);
  } else if (dimN <= 32) {
    hl_sse_matrix_column_op<32>(agg, op, sv, dimM, dimN, dst, A, lda);
  } else if (dimN <= 1024 || dimM <= 512) {
    hl_sse_matrix_column_op<64>(agg, op, sv, dimM, dimN, dst, A, lda);
  } else {
    hl_sse_matrix_column_op<1024>(agg, op, sv, dimM, dimN, dst, A, lda);
  }
}

template <int MaxRow, class Agg, class Op, class Saver>
void hl_sse_column_op_with_rem(Agg agg, Op op, Saver sv,
                               int dimM, int dimN,
                               real *dst,
                               real *A, int lda,
                               real *B, int ldb) {
  vecType mm[MaxRow / VECTOR_LEN];
  for (int n = 0; n < MaxRow / VECTOR_LEN; n++) {
    mm[n] = VECTOR_SET(agg.init());
  }

  for (int i = 0; i < dimM; i++) {
    vecType *a = (vecType*)(A + i * lda);
    vecType *b = (vecType*)(B + i * ldb);
    for (int n = 0; n < dimN / VECTOR_LEN; n++) {
      mm[n] = agg.vecOp(mm[n], op.vecOp(a[n], b[n]));
    }
  }

  vecType *result = (vecType*)(dst);
  for (int n = 0; n < dimN / VECTOR_LEN; n++) {
    result[n] = sv.vecOp(result[n], mm[n]);
  }

  int rem = dimN % VECTOR_LEN;
  if (rem) {
    A += (dimN / VECTOR_LEN) * VECTOR_LEN;
    B += (dimN / VECTOR_LEN) * VECTOR_LEN;
    dst += (dimN / VECTOR_LEN) * VECTOR_LEN;
    hl_matrix_column_op(agg, op, sv, dimM, rem, dst, A, lda, B, ldb);
  }
}

template <int Step, class Agg, class Op, class Saver>
void hl_sse_matrix_column_op(Agg agg, Op op, Saver sv,
                             int dimM, int dimN,
                             real *dst,
                             real *A, int lda,
                             real *B, int ldb) {
  for (int j = 0; j < dimN / Step; j++, dst += Step, A += Step, B += Step) {
    vecType mm[Step / VECTOR_LEN];
    for (int n = 0; n < Step / VECTOR_LEN; n++) {
      mm[n] = VECTOR_SET(agg.init());
    }

    for (int i = 0; i < dimM; i++) {
      vecType *a = (vecType*)(A + i * lda);
      vecType *b = (vecType*)(B + i * ldb);
      for (int n = 0; n < Step / VECTOR_LEN; n++) {
        mm[n] = agg.vecOp(mm[n], op.vecOp(a[n], b[n]));
      }
    }

    vecType *result = (vecType*)(dst);
    for (int n = 0; n < Step / VECTOR_LEN; n++) {
      result[n] = sv.vecOp(result[n], mm[n]);
    }
  }

  int remRow = dimN % Step;
  if (remRow) {
    hl_sse_column_op_with_rem<Step>(
        agg, op, sv, dimM, remRow, dst, A, lda, B, ldb);
  }
}

template <class Agg, class Op, class Saver>
void hl_sse_matrix_column_op(Agg agg, Op op, Saver sv,
                             int dimM, int dimN,
                             real *dst,
                             real *A, int lda,
                             real *B, int ldb) {
  if (dimN <= 16) {
    hl_sse_matrix_column_op<16>(agg, op, sv, dimM, dimN, dst, A, lda, B, ldb);
  } else if (dimN <= 32) {
    hl_sse_matrix_column_op<32>(agg, op, sv, dimM, dimN, dst, A, lda, B, ldb);
  } else if (dimN <= 1024 || dimM <= 512) {
    hl_sse_matrix_column_op<64>(agg, op, sv, dimM, dimN, dst, A, lda, B, ldb);
  } else {
    hl_sse_matrix_column_op<1024>(agg, op, sv, dimM, dimN, dst, A, lda, B, ldb);
  }
}

#endif /* HL_SSE_MATRIX_KERNEL_CUH_ */
