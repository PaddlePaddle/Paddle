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


#include "CRFLayer.h"

namespace paddle {

REGISTER_LAYER(crf, CRFLayer);

bool CRFLayer::init(const LayerMap& layerMap,
                    const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  if (config_.type() == "crf") {
    CHECK_GE(inputLayers_.size(), 2UL);
    // the third output is sequence weight. one weight for each sequence
    CHECK_LE(inputLayers_.size(), 3UL);
  }

  // coeff only affect bp, keep consistent with CostLayer
  coeff_ = config_.has_coeff() ? config_.coeff() : real(1.0);
  if (inputLayers_.size() == 3) {
    weightLayer_ = inputLayers_[2];
  }

  numClasses_ = inputLayers_[0]->getSize();

  CHECK_GE(numClasses_, 2UL);

  CHECK_EQ(parameters_[0]->getSize(), numClasses_ * (numClasses_ + 2));

  parameter_ = parameters_[0];

  // We don't need sequenceStartPositions because each sample of output_ is
  // for the cost of one sequence.
  setNeedSequenceInfo(false);
  if (useGpu_) {
    tmpCpuInput_.reserve(inputLayers_.size());
    for (size_t i = 0; i < inputLayers_.size(); i++) {
      tmpCpuInput_.push_back(Argument());
    }
  }
  return true;
}

void CRFLayer::forward(PassType passType) {
  Layer::forward(passType);
  if (useGpu_) {
    for (size_t i = 0; i < inputLayers_.size(); i++) {
      tmpCpuInput_[i].resizeAndCopyFrom(getInput(i), false, HPPL_STREAM_1);
    }
    VectorPtr cpuParameterValue;
    VectorPtr cpuParameterGradient;
    cpuParameterValue =
      Vector::create(parameter_->getBuf(PARAMETER_VALUE)->getSize(), false);
    cpuParameterValue->
      copyFrom(*parameter_->getBuf(PARAMETER_VALUE), HPPL_STREAM_1);
    if (parameter_->getBuf(PARAMETER_GRADIENT)) {
      cpuParameterGradient =
        Vector::create(parameter_->getBuf(PARAMETER_GRADIENT)->getSize(),
                       false);
      cpuParameterGradient->
        copyFrom(*parameter_->getBuf(PARAMETER_GRADIENT), HPPL_STREAM_1);
    } else {
      cpuParameterGradient = nullptr;
    }
    forwardImp(tmpCpuInput_[0], tmpCpuInput_[1], cpuParameterValue,
               cpuParameterGradient);
    parameter_->getBuf(PARAMETER_VALUE)->copyFrom(*cpuParameterValue,
                                                  HPPL_STREAM_1);
    if (parameter_->getBuf(PARAMETER_GRADIENT)) {
      parameter_->getBuf(PARAMETER_GRADIENT)->copyFrom(*cpuParameterGradient,
                                                    HPPL_STREAM_1);
    }
  } else {
    forwardImp(getInput(0), getInput(1), parameter_->getBuf(PARAMETER_VALUE),
               parameter_->getBuf(PARAMETER_GRADIENT));
  }
}

void CRFLayer::forwardImp(const Argument&output,
                          const Argument& label,
                          VectorPtr parameterValue,
                          VectorPtr parameterGradient) {
  CHECK(label.sequenceStartPositions);
  CHECK(label.ids);

  int batchSize = output.getBatchSize();
  size_t numSequences = label.sequenceStartPositions->getSize() - 1;
  resizeOutput(numSequences, 1);
  std::vector<real> out(numSequences);

  const int* starts = label.sequenceStartPositions->getData(false);
  CHECK_EQ(starts[numSequences], batchSize);
  VectorPtr cpuParameterValue;
  VectorPtr cpuParameterGradient;


  for (size_t i = 0; i < numSequences; ++i) {
    if (i >= crfs_.size()) {
      crfs_.emplace_back(numClasses_,
                         parameterValue->getData(),
                         parameterGradient
                            ? parameterGradient->getData()
                            : nullptr);
    }
    out[i] = crfs_[i].forward(
        output.value->getData() + numClasses_ * starts[i],
        label.ids->getData() + starts[i], starts[i + 1] - starts[i]);
  }
  output_.value->copyFrom(out.data(), numSequences);
  if (weightLayer_) {
    const MatrixPtr& weight = getInputValue(*weightLayer_);
    getOutputValue()->dotMul(*getOutputValue(), *weight);
  }
}

void CRFLayer::backward(const UpdateCallback &callback) {
  (void)callback;
  if (useGpu_) {
    backwardImp(callback, tmpCpuInput_[0], tmpCpuInput_[1]);
    const_cast<Argument&>(getInput(0)).
            resizeAndCopyFrom(tmpCpuInput_[0], true, HPPL_STREAM_1);
    const_cast<Argument&>(getInput(1)).
            resizeAndCopyFrom(tmpCpuInput_[1], true, HPPL_STREAM_1);

  } else {
    backwardImp(callback, getInput(0), getInput(1));
  }
}

void CRFLayer::backwardImp(const UpdateCallback& callback,
                           const Argument&output,
                           const Argument& label) {
  const int* starts = label.sequenceStartPositions->getData(false);
  int numSequences = label.sequenceStartPositions->getSize() - 1;

  for (int i = 0; i < numSequences; ++i) {
    crfs_[i].backward(output.value->getData() + numClasses_ * starts[i],
                      output.grad->getData() + numClasses_ * starts[i],
                      label.ids->getData() + starts[i],
                      starts[i + 1] - starts[i]);
    if (weightLayer_) {
      real weight = getInputValue(*weightLayer_)->getElement(i, 0);
      MatrixPtr grad = output.grad->subRowMatrix(starts[i], starts[i+1]);
      grad->mulScalar(weight);
    }
  }
  if (coeff_ != real(1.0f)) {
    output.grad->mulScalar(coeff_);
  }
  parameter_->incUpdate(callback);
}

}  // namespace paddle
