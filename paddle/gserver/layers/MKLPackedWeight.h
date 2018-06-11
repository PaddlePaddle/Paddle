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

#pragma once

#include "paddle/math/MathFunctions.h"
#include "paddle/parameter/Parameter.h"
#include "paddle/parameter/Weight.h"

namespace paddle {

class MKLPackedWeight {
 protected:
  /// The pointer of weight
  real *weight_;
  /// The pointer of cblas packed gemm to weight
  real *packedWeight_;
  size_t height_;
  size_t width_;
  bool transW_;

 public:
  explicit MKLPackedWeight(MatrixPtr weight, bool transW = false) {
    packedWeight_ = nullptr;
    weight_ = weight->getData();
    height_ = weight->getHeight();
    width_ = weight->getWidth();
    transW_ = transW;
  }

  ~MKLPackedWeight() { free_(); }

  void pack() { pack_(weight_); }

  void gemm_compute(const MatrixPtr src, MatrixPtr dst) {
    cblas_sgemm_compute(CblasRowMajor,
                        CblasNoTrans,
                        CblasPacked,
                        src->getHeight(),
                        transW_ ? height_ : width_,
                        transW_ ? width_ : height_,
                        src->getData(),
                        src->getWidth(),
                        packedWeight_,
                        width_,
                        1.0,
                        dst->getData(),
                        dst->getWidth());
  }

 protected:
  void pack_(real *src) {
    if (!packedWeight_) {
      packedWeight_ = cblas_sgemm_alloc(CblasBMatrix, 1, width_, height_);
    }
    cblas_sgemm_pack(CblasRowMajor,
                     CblasBMatrix,
                     transW_ ? CblasTrans : CblasNoTrans,
                     1,
                     transW_ ? height_ : width_,
                     transW_ ? width_ : height_,
                     1.0,
                     src,
                     width_,
                     packedWeight_);
  }

  void free_() {
    if (packedWeight_) {
      cblas_sgemm_free(packedWeight_);
    }
  }
};

}  // namespace paddle
