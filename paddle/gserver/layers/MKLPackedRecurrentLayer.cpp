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

#include "MKLPackedRecurrentLayer.h"

namespace paddle {

REGISTER_LAYER(mkl_packed_recurrent, MKLPackedRecurrentLayer);

bool MKLPackedRecurrentLayer::init(const LayerMap& layerMap,
                                   const ParameterMap& parameterMap) {
  if (!RecurrentLayer::init(layerMap, parameterMap)) return false;
  packed_weight_.reset(new MKLPackedWeight(weight_->getW()));
  packed_weight_->pack();
  if (needGradient_) {
    packed_weightT_.reset(new MKLPackedWeight(weight_->getW(), true));
    packed_weightT_->pack();
  }
  return true;
}

void MKLPackedRecurrentLayer::backward(const UpdateCallback& callback) {
  RecurrentLayer::backward(callback);
  packed_weight_->pack();
  if (needGradient_) {
    packed_weightT_->pack();
  }
}

void MKLPackedRecurrentLayer::forwardBatch(int batchSize,
                                           size_t numSequences,
                                           const int* starts) {
  if (!batchValue_) {
    batchValue_.reset(new SequenceToBatch(useGpu_));
  }

  batchValue_->resizeOrCreateBatch(batchSize, numSequences, starts, reversed_);

  batchValue_->copyFromSeq(*output_.value);

  {
    REGISTER_TIMER_INFO("RecurrentFwBatch", getName().c_str());
    /* forward one batch */
    for (size_t n = 0; n < batchValue_->getNumBatch(); n++) {
      MatrixPtr batch2 = batchValue_->getBatchValue(n);

      if (n != 0) {
        MatrixPtr batch1 =
            batchValue_->getBatchValue(n - 1, batch2->getHeight());

        // batch2->mul(*batch1, *weight_->getW(), 1, 1);
        packed_weight_->compute(batch2, batch1);
      }

#pragma omp parallel for collapse(2)
      for (size_t i = 0; i < batch2->getHeight(); i++) {
        for (size_t j = 0; j < batch2->getWidth(); j++) {
          *(batch2->getData() + i * batch2->getWidth() + j) =
              *(batch2->getData() + i * batch2->getWidth() + j) > 0
                  ? *(batch2->getData() + i * batch2->getWidth() + j)
                  : 0;
        }
      }
    }
  }

  batchValue_->copyBackSeq(*output_.value);
}

void MKLPackedRecurrentLayer::backwardBatch(int batchSize,
                                            size_t numSequences,
                                            const int* starts) {
  if (!batchGrad_) {
    batchGrad_.reset(new SequenceToBatch(useGpu_));
  }
  batchGrad_->shareIndexWith(*batchValue_);

  size_t numBatch = batchGrad_->getNumBatch();
  bool backwardByBatch = numBatch < numSequences;

  batchGrad_->copyFromSeq(*output_.grad);
  {
    REGISTER_TIMER_INFO("RecurrentBwData", getName().c_str());
    /* backward one batch */
    for (int n = (int)numBatch - 1; n >= 0; n--) {
      MatrixPtr batch2 = batchGrad_->getBatchValue(n);
      MatrixPtr batch1 = batchValue_->getBatchValue(n, batch2->getHeight());

      Argument arg;
      arg.value = batch1;
      arg.grad = batch2;
      activation_->backward(arg).check();

      if (n != 0) {
        batch1 = batchGrad_->getBatchValue(n - 1, batch2->getHeight());
        // batch1->mul(*batch2, *weightT, 1, 1);
        packed_weightT_->compute(batch1, batch2);
      }

      if (backwardByBatch && weight_->getWGrad()) {
        if (n != 0) {
          /* backward weight */
          batch1 = batchValue_->getBatchValue(n - 1, batch2->getHeight());
          weight_->getWGrad()->mul(*batch1->getTranspose(), *batch2, 1, 1);
        }
      }
    }
  }

  batchGrad_->copyBackSeq(*output_.grad);

  if (!backwardByBatch && weight_->getWGrad()) {
    REGISTER_TIMER_INFO("RecurrentBwWeight", getName().c_str());
    for (size_t seq = 0; seq < numSequences; ++seq) {
      int len = starts[seq + 1] - starts[seq];
      if (!reversed_) {
        weight_->getWGrad()->mul(
            *output_.value->subMatrix(starts[seq], len - 1)->getTranspose(),
            *output_.grad->subMatrix(starts[seq] + 1, len - 1),
            1,
            1);
      } else {
        weight_->getWGrad()->mul(
            *output_.value->subMatrix(starts[seq] + 1, len - 1)->getTranspose(),
            *output_.grad->subMatrix(starts[seq], len - 1),
            1,
            1);
      }
    }
  }
}

}  // namespace paddle
