/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "AverageLayer.h"

#include "paddle/utils/Logging.h"

#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_LAYER(average, AverageLayer);

bool AverageLayer::init(const LayerMap& layerMap,
                        const ParameterMap& parameterMap) {
  SequencePoolLayer::init(layerMap, parameterMap);

  dataMtx_ = Matrix::create(nullptr, 1, 1, false, useGpu_);
  outMtx_ = Matrix::create(nullptr, 1, getSize(), false, useGpu_);
  // average strategy
  if (config_.average_strategy() == "average") {
    mode_ = kAverage;
  } else if (config_.average_strategy() == "sum") {
    mode_ = kSum;
  } else if (config_.average_strategy() == "squarerootn") {
    mode_ = kAverageSquareRootN;
  } else {
    LOG(FATAL) << "Unknown average strategy: " << config_.average_strategy();
  }
  return true;
}

void AverageLayer::forward(PassType passType) {
  SequencePoolLayer::forward(passType);

  MatrixPtr inputValue = getInputValue(0);
  getOutputValue()->sequenceAvgForward(
      *inputValue, *startPositions_->getVector(useGpu_), mode_);

  /* add the bias-vector AFTER average operation */
  if (biases_.get() != NULL) {
    MatrixPtr outV = getOutputValue();
    outV->addBias(*(biases_->getW()), 1);
  }

  /* activation */ { forwardActivation(); }
}

void AverageLayer::backward(const UpdateCallback& callback) {
  SequencePoolLayer::backward(callback);

  const int* starts = startPositions_->getData(false);
  MatrixPtr grad = getInputGrad(0);

  if (grad) {
    size_t dim = getSize();
    real* gradientData = getInputGrad(0)->getData();
    real* gradient = getOutputGrad()->getData();
    size_t numSequences = startPositions_->getSize() - 1;
    for (size_t sequenceId = 0; sequenceId < numSequences; ++sequenceId) {
      // TODO(Dangqingqing) optimization for GPU
      int sequenceLength = starts[sequenceId + 1] - starts[sequenceId];
      if (0 == sequenceLength) {
        // empty sequence
        continue;
      }
      dataMtx_->setData(gradientData + starts[sequenceId] * dim, sequenceLength,
                        dim);
      outMtx_->setData(gradient + sequenceId * dim);
      switch (mode_) {
        case kAverage: {
          // plain average
          dataMtx_->addBias(*outMtx_, 1.0f / sequenceLength);
          break;
        }
        case kSum: {
          // sum instead of average
          dataMtx_->addBias(*outMtx_, 1.0f);
          break;
        }
        case kAverageSquareRootN: {
          // divide by square root of sequenceLength
          dataMtx_->addBias(*outMtx_, 1.0f / sqrt(sequenceLength));
          break;
        }
        default: { LOG(FATAL) << "should not reach here"; }
      }
    }
  }
}

}  // namespace paddle
