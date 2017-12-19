/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserve.

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
#include "paddle/math/Matrix.h"

namespace paddle {

class MKLPackedGemm {
protected:
  real* weightPacked_;
  real* weightTPacked_;
  size_t weightHeight_;
  size_t weightWidth_;

public:
  explicit MKLPackedGemm(MatrixPtr weight) {
    weightHeight_ = weight->getHeight();
    weightWidth_ = weight->getWidth();
    weightPacked_ =
        cblas_sgemm_alloc(CblasBMatrix, 1, weightWidth_, weightHeight_);
    weightTPacked_ =
        cblas_sgemm_alloc(CblasBMatrix, 1, weightWidth_, weightHeight_);
    cblas_sgemm_pack(CblasRowMajor,
                     CblasBMatrix,
                     CblasNoTrans,
                     1,
                     weightWidth_,
                     weightHeight_,
                     1.0,
                     weight->getData(),
                     weightWidth_,
                     weightPacked_);
    cblas_sgemm_pack(CblasRowMajor,
                     CblasBMatrix,
                     CblasTrans,
                     1,
                     weightWidth_,
                     weightHeight_,
                     1.0,
                     weight->getData(),
                     weightWidth_,
                     weightTPacked_);
  }
  void compute(MatrixPtr batch2, MatrixPtr batch1, bool transW = false) {
    if (transW) {
      cblas_sgemm_compute(CblasRowMajor,
                          CblasNoTrans,
                          CblasPacked,
                          batch2->getHeight(),
                          weightWidth_,
                          weightHeight_,
                          batch1->getData(),
                          weightHeight_,
                          weightTPacked_,
                          weightWidth_,
                          1,
                          batch2->getData(),
                          weightWidth_);
    } else {
      cblas_sgemm_compute(CblasRowMajor,
                          CblasNoTrans,
                          CblasPacked,
                          batch2->getHeight(),
                          weightWidth_,
                          weightHeight_,
                          batch1->getData(),
                          weightHeight_,
                          weightPacked_,
                          weightWidth_,
                          1,
                          batch2->getData(),
                          weightWidth_);
    }
  }
  ~MKLPackedGemm() {
    cblas_sgemm_free(weightPacked_);
    cblas_sgemm_free(weightTPacked_);
  }
};

}  // namespace paddle
