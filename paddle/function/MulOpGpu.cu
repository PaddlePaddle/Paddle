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
                            real scaleT,
                            bool aTrans,
                            bool bTrans,
                            bool cTrans) {
  CHECK(a.useGpu_ && b.useGpu_) << "matrix device type not match";
  real* aData = const_cast<real*>(a.getData());
  real* bData = const_cast<real*>(b.getData());
  real* outData = const_cast<real*>(out.getData());
  hl_matrix_mul(aData,
                !aTrans ? HPPL_OP_N : HPPL_OP_T,
                bData,
                !bTrans ? HPPL_OP_N : HPPL_OP_T,
                outData,
                out.getHeight(),
                out.getWidth(),
                !aTrans ? a.getWidth() : a.getHeight(),
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
                            real scaleT,
                            bool aTrans,
                            bool bTrans,
                            bool cTrans) {
  CHECK(out.isContiguous());
  CHECK(b.isContiguous());
  CHECK(a.useGpu_ && b.useGpu_) << "matrix device type not match";

  hl_sparse_matrix_s aData = a.sMatrix_.get();
  real* bData = const_cast<real*>(b.getData());
  real* outData = const_cast<real*>(out.getData());
  hl_matrix_csr_mul_dense(aData,
                          aTrans ? HPPL_OP_T : HPPL_OP_N,
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
                            real scaleT,
                            bool aTrans,
                            bool bTrans,
                            bool cTrans) {
  CHECK(out.isContiguous());
  CHECK(a.isContiguous());
  CHECK(a.useGpu_ && b.useGpu_) << "matrix device type not match";

  hl_sparse_matrix_s bData = b.sMatrix_.get();
  real* aData = const_cast<real*>(a.getData());
  real* outData = const_cast<real*>(out.getData());

  if (b.format_ == SPARSE_CSC) {
    hl_matrix_dense_mul_csc(aData,
                            HPPL_OP_N,
                            bData,
                            bTrans ? HPPL_OP_T : HPPL_OP_N,
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
                            bTrans ? HPPL_OP_T : HPPL_OP_N,
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
                            real scaleT,
                            bool aTrans,
                            bool bTrans,
                            bool cTrans) {
  CHECK(a.useGpu_ && b.useGpu_) << "matrix device type not match";

  real* aData = const_cast<real*>(a.getData());
  real* bData = const_cast<real*>(b.getData());
  hl_sparse_matrix_s outData = out.sMatrix_.get();

  hl_sparse_matrix_mul(aData,
                       aTrans ? HPPL_OP_T : HPPL_OP_N,
                       bData,
                       bTrans ? HPPL_OP_T : HPPL_OP_N,
                       outData,
                       out.getHeight(),
                       out.getWidth(),
                       !bTrans ? b.getHeight() : b.getWidth(),
                       scaleAB,
                       scaleT);
}

}  // namespace paddle
