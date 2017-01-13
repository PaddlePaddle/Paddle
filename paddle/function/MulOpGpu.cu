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
    CHECK(out.width_ == b.width_ && out.height_ == a.height_
          && a.width_ == b.height_) << "Matrix dimensions are not equal";
  } else {
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

}  // namespace paddle
