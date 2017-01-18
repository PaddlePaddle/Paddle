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
 * out = scale_t * out + scale_ab * (a * b)
 * out : output matrix, M * N
 */
template <>
void MulOp<DEVICE_TYPE_GPU>(GpuMatrix& out,
                            const GpuMatrix& a,
                            const GpuMatrix& b,
                            real scale_ab,
                            real scale_t) {
  CHECK(!out.isTransposed()) << "Not supported";

  if (!a.isTransposed() && !b.isTransposed()) {
    /// a : M * K, b: K * N
    CHECK_EQ(out.width_, b.width_);
    CHECK_EQ(out.height_, a.height_);
    CHECK_EQ(a.width_, b.height_);
  } else if (a.isTransposed() && !b.isTransposed()) {
    /// a : K * M, b : K * N
    CHECK_EQ(out.width_, b.width_);
    CHECK_EQ(out.height_, a.width_);
    CHECK_EQ(a.height_, b.height_);
  } else if (!a.isTransposed() && b.isTransposed()) {
    /// a: M * K, b : N * K
    CHECK_EQ(out.width_, b.height_);
    CHECK_EQ(out.height_, a.height_);
    CHECK_EQ(a.width_, b.width_);
  } else {
    LOG(FATAL) << "Is not supported";
  }

  real* a_data = a.data_;
  real* b_data = b.data_;
  real* out_data = out.data_;
  int dim_m = out.getHeight();
  int dim_n = out.getWidth();
  int dim_k = !a.isTransposed() ? a.width_ : a.height_;
  int lda = a.getStride();
  int ldb = b.getStride();
  int ldc = out.getStride();
  hl_trans_op_t trans_a = !a.isTransposed() ? HPPL_OP_N : HPPL_OP_T;
  hl_trans_op_t trans_b = !b.isTransposed() ? HPPL_OP_N : HPPL_OP_T;

  hl_matrix_mul(a_data,
                trans_a,
                b_data,
                trans_b,
                out_data,
                dim_m,
                dim_n,
                dim_k,
                scale_ab,
                scale_t,
                lda,
                ldb,
                ldc);
}

/**
 * out = scale_t * out + scale_ab * (a * b)
 * out : M * N
 */
template <>
void MulOp<DEVICE_TYPE_GPU>(GpuMatrix& out,
                            const GpuSparseMatrix& a,
                            const GpuMatrix& b,
                            real scale_ab,
                            real scale_t) {
  CHECK(out.isContiguous());
  CHECK(b.isContiguous());
  CHECK(b.useGpu_ == true) << "Matrix type are not equal";
  CHECK(!out.trans_ && !b.trans_) << "not supported";
  if (!a.trans_) {
    /// a: M * K,  b: K * N
    CHECK(out.width_ == b.width_ && out.height_ == a.height_
        && a.width_ == b.height_) << "Matrix dimensions are not equal";
  } else {
    /// a: K * M, transpose,  b: K * N
    CHECK(out.width_ == b.width_ && out.height_ == a.width_
        && a.height_ == b.height_) << "Matrix dimensions are not equal";
  }

  hl_trans_op_t a_trans = a.trans_ ? HPPL_OP_T : HPPL_OP_N;
  hl_sparse_matrix_s a_data = a.sMatrix_.get();
  real* b_data = b.data_;
  real* out_data = out.data_;
  hl_matrix_csr_mul_dense(a_data,
                          a_trans,
                          b_data,
                          HPPL_OP_N,
                          out_data,
                          out.height_,
                          out.width_,
                          b.height_,
                          scale_ab,
                          scale_t);
}

/**
 * out = scale_t * out + scale_ab * (a * b)
 * out : M * N
 */
template <>
void MulOp<DEVICE_TYPE_GPU>(GpuMatrix& out,
                            const GpuMatrix& a,
                            const GpuSparseMatrix& b,
                            real scale_ab,
                            real scale_t) {
  CHECK(out.isContiguous());
  CHECK(a.isContiguous());
  CHECK(a.useGpu_ == true) << "Matrix type are not equal";

  hl_sparse_matrix_s b_data = b.sMatrix_.get();
  real* a_data = a.data_;
  real* out_data = out.data_;
  hl_trans_op_t trans_b = b.trans_ ? HPPL_OP_T : HPPL_OP_N;
  if (!b.trans_) {
    /// a : M * K, b : K * N
    CHECK(out.width_ == b.width_ &&
          out.height_ == a.height_ && a.width_ == b.height_)
        << "Matrix dimensions are not equal";
  } else {
    /// a : M * K, b : N * K, transpose
    CHECK(out.width_ == b.height_ &&
          out.height_ == a.height_ && a.width_ == b.width_)
        << "Matrix dimensions are not equal";
  }
  if (b.format_ == SPARSE_CSC) {
    hl_matrix_dense_mul_csc(a_data,
                            HPPL_OP_N,
                            b_data,
                            trans_b,
                            out_data,
                            out.height_,
                            out.width_,
                            a.width_,
                            scale_ab,
                            scale_t);
  } else {
    hl_matrix_dense_mul_csr(a_data,
                            HPPL_OP_N,
                            b_data,
                            trans_b,
                            out_data,
                            out.height_,
                            out.width_,
                            a.width_,
                            scale_ab,
                            scale_t);
  }
}

template <>
void MulOp<DEVICE_TYPE_GPU>(GpuSparseMatrix& out,
                            const GpuMatrix& a,
                            const GpuMatrix& b,
                            real scale_ab,
                            real scale_t) {
  /// todo(tianbing), clean the code
  CHECK(a.useGpu_ && b.useGpu_) << "type not match";
  CHECK(!out.trans_) << "trans not supported";
  real* a_data = const_cast<real*>(a.getData());
  real* b_data = const_cast<real*>(b.getData());
  hl_sparse_matrix_s out_data = out.sMatrix_.get();
  hl_trans_op_t a_trans = a.trans_ ? HPPL_OP_T : HPPL_OP_N;
  hl_trans_op_t b_trans = b.trans_ ? HPPL_OP_T : HPPL_OP_N;

  if (!a.trans_ && !b.trans_) {
    CHECK(out.height_ == a.getHeight());
    CHECK(out.width_ == b.getWidth());
    CHECK(a.getWidth() == b.getHeight());
  } else if (a.trans_ && !b.trans_) {
    CHECK(out.height_ == a.getWidth());
    CHECK(out.width_ == b.getWidth());
    CHECK(a.getHeight() == b.getHeight());
  } else if (!a.trans_ && b.trans_) {
    CHECK(out.height_ == a.getHeight());
    CHECK(out.width_ == b.getHeight());
    CHECK(a.getWidth() == b.getWidth());
  } else {
    LOG(INFO) << "Not support";
  }
  int dim_m = out.height_;
  int dim_n = out.width_;
  int dim_k = !b.trans_ ? b.getHeight() : b.getWidth();
  hl_sparse_matrix_mul(
      a_data, a_trans, b_data, b_trans, out_data,
      dim_m, dim_n, dim_k, scale_ab, scale_t);
}

}  // namespace paddle
