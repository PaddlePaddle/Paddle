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

#include "hl_base.h"
#include "MulOp.h"
#include "paddle/math/Matrix.h"
#include "paddle/math/SparseMatrix.h"

namespace paddle {
/**
 * out = scaleT * out + scaleAB * (a * b)
 * out : output matrix, M * N
 */
template <>
void MulOp<DEVICE_TYPE_GPU>(GpuMatrix& out,
                            const GpuMatrix& a,
                            const GpuMatrix& b,
                            real scaleAB,
                            real scaleT) {
  CHECK(!out.isTransposed()) << "Transpose not supported for out matrix";
  if (!a.isTransposed() && !b.isTransposed()) {
      /// a : M * K, b: K * N
      CHECK(out.getWidth() == b.getWidth() &&
              out.getHeight() == a.getHeight() &&
              a.getWidth() == b.getHeight());
  } else if (a.isTransposed() && !b.isTransposed()) {
      /// a : K * M, b : K * N
      CHECK(out.getWidth() == b.getWidth() &&
              out.getHeight() == a.getWidth() &&
              a.getHeight() == b.getHeight());
  } else if (!a.isTransposed() && b.isTransposed()) {
      /// a: M * K, b : N * K
      CHECK(out.getWidth() == b.getHeight() &&
              out.getHeight() == a.getHeight() &&
              a.getWidth() == b.getWidth());
  } else {
    LOG(FATAL) << "Not support for both a and b are Transposed Matrices";
  }

  real* aData = const_cast<real*>(a.getData());
  real* bData = const_cast<real*>(b.getData());
  real* outData = const_cast<real*>(out.getData());
  hl_matrix_mul(aData,
                !a.isTransposed() ? HPPL_OP_N : HPPL_OP_T,
                bData,
                !b.isTransposed() ? HPPL_OP_N : HPPL_OP_T,
                outData,
                out.getHeight(),
                out.getWidth(),
                !a.isTransposed() ? a.getWidth() : a.getHeight(),
                scaleAB,
                scaleT,
                a.getStride(),
                b.getStride(),
                out.getStride());
}

/**
 * out = scaleT * out + scaleAB * (a * b)
 * out : M * N
 */
template <>
void MulOp<DEVICE_TYPE_GPU>(GpuMatrix& out,
                            const GpuSparseMatrix& a,
                            const GpuMatrix& b,
                            real scaleAB,
                            real scaleT) {
  CHECK(out.isContiguous());
  CHECK(b.isContiguous());
  CHECK(b.useGpu_) << "Matrix type are not equal";
  CHECK(!out.isTransposed() && !b.isTransposed()) << "not supported";
  if (!a.isTransposed()) {
    /// a: M * K,  b: K * N
    CHECK(out.getWidth() == b.getWidth() && out.getHeight() == a.getHeight()
        && a.getWidth() == b.getHeight()) << "Matrix dimensions are not equal";
  } else {
    /// a: K * M, transpose,  b: K * N
    CHECK(out.getWidth() == b.getWidth() && out.getHeight() == a.getWidth()
        && a.getHeight() == b.getHeight()) << "Matrix dimensions are not equal";
  }

  hl_trans_op_t aTrans = a.isTransposed() ? HPPL_OP_T : HPPL_OP_N;
  hl_sparse_matrix_s aData = a.sMatrix_.get();
  real* bData = const_cast<real*>(b.getData());
  real* outData = const_cast<real*>(out.getData());
  hl_matrix_csr_mul_dense(aData,
                          aTrans,
                          bData,
                          HPPL_OP_N,
                          outData,
                          out.getHeight(),
                          out.getWidth(),
                          b.getHeight(),
                          scaleAB,
                          scaleT);
}

/**
 * out = scaleT * out + scaleAB * (a * b)
 * out : M * N
 */
template <>
void MulOp<DEVICE_TYPE_GPU>(GpuMatrix& out,
                            const GpuMatrix& a,
                            const GpuSparseMatrix& b,
                            real scaleAB,
                            real scaleT) {
  CHECK(out.isContiguous());
  CHECK(a.isContiguous());
  CHECK(a.useGpu_) << "Matrix type are not equal";
  if (!b.isTransposed()) {
      /// a : M * K, b : K * N
      CHECK(out.getWidth() == b.getWidth() &&
              out.getHeight() == a.getHeight() &&
              a.getWidth() == b.getHeight())
          << "Matrix dimensions are not equal";
  } else {
      /// a : M * K, b : N * K, transpose
      CHECK(out.getWidth() == b.getHeight() &&
              out.getHeight() == a.getHeight() &&
              a.getWidth() == b.getWidth())
          << "Matrix dimensions are not equal";
  }

  hl_trans_op_t bTrans = b.isTransposed() ? HPPL_OP_T : HPPL_OP_N;
  hl_sparse_matrix_s bData = b.sMatrix_.get();
  real* aData = const_cast<real*>(a.getData());
  real* outData = const_cast<real*>(out.getData());

  if (b.format_ == SPARSE_CSC) {
    hl_matrix_dense_mul_csc(aData,
                            HPPL_OP_N,
                            bData,
                            bTrans,
                            outData,
                            out.getHeight(),
                            out.getWidth(),
                            a.getWidth(),
                            scaleAB,
                            scaleT);
  } else {
    hl_matrix_dense_mul_csr(aData,
                            HPPL_OP_N,
                            bData,
                            bTrans,
                            outData,
                            out.getHeight(),
                            out.getWidth(),
                            a.getWidth(),
                            scaleAB,
                            scaleT);
  }
}

template <>
void MulOp<DEVICE_TYPE_GPU>(GpuSparseMatrix& out,
                            const GpuMatrix& a,
                            const GpuMatrix& b,
                            real scaleAB,
                            real scaleT) {
  CHECK(a.useGpu_ && b.useGpu_) << "matrix device type not match";
  CHECK(!out.isTransposed()) << "Transpose is not supported for out matrix";

  if (!a.isTransposed() && !b.isTransposed()) {
    CHECK(out.getHeight() == a.getHeight() &&
         out.getWidth() == b.getWidth() &&
         a.getWidth() == b.getHeight());
  } else if (a.isTransposed() && !b.isTransposed()) {
    CHECK(out.getHeight() == a.getWidth() &&
          out.getWidth() == b.getWidth() &&
          a.getHeight() == b.getHeight());
  } else if (!a.isTransposed() && b.isTransposed()) {
    CHECK(out.getHeight() == a.getHeight() &&
          out.getWidth() == b.getHeight() &&
          a.getWidth() == b.getWidth());
  } else {
    LOG(FATAL) << "Not support for both a and b are Transposed Matrices";
  }

  hl_trans_op_t aTrans = a.isTransposed() ? HPPL_OP_T : HPPL_OP_N;
  hl_trans_op_t bTrans = b.isTransposed() ? HPPL_OP_T : HPPL_OP_N;
  int dimK = !b.isTransposed() ? b.getHeight() : b.getWidth();
  real* aData = const_cast<real*>(a.getData());
  real* bData = const_cast<real*>(b.getData());
  hl_sparse_matrix_s outData = out.sMatrix_.get();

  hl_sparse_matrix_mul(aData, aTrans, bData, bTrans, outData,
      out.getHeight(), out.getWidth(), dimK, scaleAB, scaleT);
}

}  // namespace paddle
