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

#include "CRFDecodingLayer.h"

namespace paddle {

REGISTER_LAYER(crf_decoding, CRFDecodingLayer);

bool CRFDecodingLayer::init(const LayerMap& layerMap,
                            const ParameterMap& parameterMap) {
  if (!CRFLayer::init(layerMap, parameterMap)) {
    return false;
  }
  if (!useGpu_) {
    crf_.reset(new LinearChainCRF(
        numClasses_, parameter_->getBuf(PARAMETER_VALUE)->getData()));
  }
  return true;
}

void CRFDecodingLayer::forward(PassType passType) {
  Layer::forward(passType);

  if (useGpu_) {
    cpuParam =
        Vector::create(parameter_->getBuf(PARAMETER_VALUE)->getSize(), false);
    cpuParam->copyFrom(*parameter_->getBuf(PARAMETER_VALUE));
    crf_.reset(new LinearChainCRF(numClasses_, cpuParam->getData()));
  }
  const Argument& output = getInput(0);
  CHECK(output.sequenceStartPositions);

  size_t batchSize = output.getBatchSize();
  size_t numSequences = output.sequenceStartPositions->getSize() - 1;

  IVector::resizeOrCreate(output_.ids, batchSize, useGpu_);
  IVectorPtr output_ids = output_.ids;
  MatrixPtr output_arg_val = output.value;
  if (useGpu_) {
    Matrix::resizeOrCreate(cpuOutputArg_,
                           /* height */ output_arg_val->getHeight(),
                           /* width */ output_arg_val->getWidth(),
                           /* trans */ false,
                           /* useGpu */ false);
    IVector::resizeOrCreate(cpuOutputId_, batchSize, false);
    cpuOutputArg_->copyFrom(*output_arg_val);
  } else {
    cpuOutputId_ = output_ids;
    cpuOutputArg_ = output_arg_val;
  }
  const int* starts = output.sequenceStartPositions->getData(false);
  CHECK_EQ(starts[numSequences], (int)batchSize);

  for (size_t i = 0; i < numSequences; ++i) {
    crf_->decode(cpuOutputArg_->getData() + numClasses_ * starts[i],
                 cpuOutputId_->getData() + starts[i],
                 starts[i + 1] - starts[i]);
  }

  if (inputLayers_.size() == 2) {
    const Argument& label = getInput(1);
    resizeOutput(batchSize, 1);
    CHECK(label.ids);
    MatrixPtr output_val = output_.value;
    if (useGpu_) {
      Matrix::resizeOrCreate(cpuOutput_,
                             /* height */ output_val->getHeight(),
                             /* width */ output_val->getWidth(),
                             /* trans */ false,
                             /* useGpu */ false);
      IVector::resizeOrCreate(cpuLabel_, label.ids->getSize(), false);
      cpuOutput_->copyFrom(*output_val);
      cpuLabel_->copyFrom(*label.ids);
    } else {
      cpuOutput_ = output_val;
      cpuLabel_ = label.ids;
    }
    real* error = cpuOutput_->getData();
    int* ids = cpuLabel_->getData();
    int* result = cpuOutputId_->getData();
    for (size_t i = 0; i < batchSize; ++i) {
      error[i] = ids[i] == result[i] ? 0 : 1;
    }
    if (useGpu_) {
      output_val->copyFrom(*cpuOutput_);
    } else {
      output_val = cpuOutput_;
    }
  }
  if (useGpu_) {
    output_ids->copyFrom(*cpuOutputId_);
    output_arg_val->copyFrom(*cpuOutputArg_);
  } else {
    output_ids = cpuOutputId_;
    output_arg_val = cpuOutputArg_;
  }
}

void CRFDecodingLayer::backward(const UpdateCallback& callback) {
  parameter_->incUpdate(callback);
}

}  // namespace paddle
