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

#include "MulOp.h"
#include "hl_base.h"
#include "paddle/math/Matrix.h"
#include "paddle/math/SparseMatrix.h"

namespace paddle {
/// dense matrix (+)= dense matrix * dense matrix
template <>
void MulOp<DEVICE_TYPE_GPU>(GpuMatrix& out,
                            const GpuMatrix& a,
                            const GpuMatrix& b,
                            real scaleAB,
                            real scaleT,
                            bool aTrans,
                            bool bTrans) {
  CHECK(a.useGpu_ && b.useGpu_) << "matrix device type not match";
  hl_matrix_mul(const_cast<real*>(a.getData()),
                !aTrans ? HPPL_OP_N : HPPL_OP_T,
                const_cast<real*>(b.getData()),
                !bTrans ? HPPL_OP_N : HPPL_OP_T,
                const_cast<real*>(out.getData()),
                out.getHeight(),
                out.getWidth(),
                !aTrans ? a.getWidth() : a.getHeight(),
                scaleAB,
                scaleT,
                a.getStride(),
                b.getStride(),
                out.getStride());
}

/// dense matrix (+)= sparse matrix * dense matrix
template <>
void MulOp<DEVICE_TYPE_GPU>(GpuMatrix& out,
                            const GpuSparseMatrix& a,
                            const GpuMatrix& b,
                            real scaleAB,
                            real scaleT,
                            bool aTrans,
                            bool bTrans) {
  CHECK(out.isContiguous());
  CHECK(b.isContiguous());
  CHECK(a.useGpu_ && b.useGpu_) << "matrix device type not match";
  hl_matrix_csr_mul_dense(a.sMatrix_.get(),
                          aTrans ? HPPL_OP_T : HPPL_OP_N,
                          const_cast<real*>(b.getData()),
                          HPPL_OP_N,
                          const_cast<real*>(out.getData()),
                          out.getHeight(),
                          out.getWidth(),
                          b.getHeight(),
                          scaleAB,
                          scaleT);
}

/// dense matrix (+)= dense matrix * sparse matrix
template <>
void MulOp<DEVICE_TYPE_GPU>(GpuMatrix& out,
                            const GpuMatrix& a,
                            const GpuSparseMatrix& b,
                            real scaleAB,
                            real scaleT,
                            bool aTrans,
                            bool bTrans) {
  CHECK(out.isContiguous());
  CHECK(a.isContiguous());
  CHECK(a.useGpu_ && b.useGpu_) << "matrix device type not match";

  if (b.format_ == SPARSE_CSC) {
    hl_matrix_dense_mul_csc(const_cast<real*>(a.getData()),
                            HPPL_OP_N,
                            b.sMatrix_.get(),
                            bTrans ? HPPL_OP_T : HPPL_OP_N,
                            const_cast<real*>(out.getData()),
                            out.getHeight(),
                            out.getWidth(),
                            a.getWidth(),
                            scaleAB,
                            scaleT);
  } else {
    hl_matrix_dense_mul_csr(const_cast<real*>(a.getData()),
                            HPPL_OP_N,
                            b.sMatrix_.get(),
                            bTrans ? HPPL_OP_T : HPPL_OP_N,
                            const_cast<real*>(out.getData()),
                            out.getHeight(),
                            out.getWidth(),
                            a.getWidth(),
                            scaleAB,
                            scaleT);
  }
}

/// sparse matrix (+)= dense matrix * dense matrix
template <>
void MulOp<DEVICE_TYPE_GPU>(GpuSparseMatrix& out,
                            const GpuMatrix& a,
                            const GpuMatrix& b,
                            real scaleAB,
                            real scaleT,
                            bool aTrans,
                            bool bTrans) {
  CHECK(a.useGpu_ && b.useGpu_) << "matrix device type not match";
  hl_sparse_matrix_mul(const_cast<real*>(a.getData()),
                       aTrans ? HPPL_OP_T : HPPL_OP_N,
                       const_cast<real*>(b.getData()),
                       bTrans ? HPPL_OP_T : HPPL_OP_N,
                       out.sMatrix_.get(),
                       out.getHeight(),
                       out.getWidth(),
                       !bTrans ? b.getHeight() : b.getWidth(),
                       scaleAB,
                       scaleT);
}

}  // namespace paddle
