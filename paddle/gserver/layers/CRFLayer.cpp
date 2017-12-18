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
  coeff_ = config_.coeff();
  if (inputLayers_.size() == 3) {
    weightLayer_ = inputLayers_[2];
  }

  numClasses_ = inputLayers_[0]->getSize();

  CHECK_GE(numClasses_, 2UL);

  CHECK_EQ(parameters_[0]->getSize(), numClasses_ * (numClasses_ + 2));

  parameter_ = parameters_[0];
  weight_.reset(new Weight(numClasses_ + 2, numClasses_, parameter_));

  // We don't need sequenceStartPositions because each sample of output_ is
  // for the cost of one sequence.
  setNeedSequenceInfo(false);

  return true;
}

void CRFLayer::forward(PassType passType) {
  Layer::forward(passType);

  const Argument& output = getInput(0);
  const Argument& label = getInput(1);
  CHECK(label.sequenceStartPositions);
  CHECK(label.ids);

  int batchSize = output.getBatchSize();
  size_t numSequences = label.sequenceStartPositions->getSize() - 1;
  resizeOutput(numSequences, 1);

  const int* starts = label.sequenceStartPositions->getData(false);
  CHECK_EQ(starts[numSequences], batchSize);

  MatrixPtr weight_val = weight_->getW();
  MatrixPtr output_val = output_.value;
  MatrixPtr output_arg_val = output.value;
  IVectorPtr label_val = label.ids;
  if (useGpu_) {
    Matrix::resizeOrCreate(cpuWeight_,
                           /* height */ weight_val->getHeight(),
                           /* width */ weight_val->getWidth(),
                           /* trans */ false,
                           /* useGpu */ false);
    Matrix::resizeOrCreate(cpuOutput_,
                           /* height */ output_val->getHeight(),
                           /* width */ output_val->getWidth(),
                           /* trans */ false,
                           /* useGpu */ false);
    Matrix::resizeOrCreate(cpuOutputArg_,
                           /* height */ output_arg_val->getHeight(),
                           /* width */ output_arg_val->getWidth(),
                           /* trans */ false,
                           /* useGpu */ false);
    IVector::resizeOrCreate(cpuLabel_, label_val->getSize(), false);
    cpuWeight_->copyFrom(*weight_val);
    cpuOutputArg_->copyFrom(*output_arg_val);
    cpuOutput_->copyFrom(*output_val);
    cpuLabel_->copyFrom(*label_val);
  } else {
    cpuWeight_ = weight_val;
    cpuOutputArg_ = output_arg_val;
    cpuOutput_ = output_val;
    cpuLabel_ = label_val;
  }
  for (size_t i = 0; i < numSequences; ++i) {
    if (i >= crfs_.size()) {
      crfs_.emplace_back(numClasses_, cpuWeight_->getData());
    }
    cpuOutput_->getData()[i] =
        crfs_[i].forward(cpuOutputArg_->getData() + numClasses_ * starts[i],
                         cpuLabel_->getData() + starts[i],
                         starts[i + 1] - starts[i]);
  }
  if (useGpu_) {
    output_val->copyFrom(*cpuOutput_);
    output_arg_val->copyFrom(*cpuOutputArg_);
  } else {
    output_val = cpuOutput_;
    output_arg_val = cpuOutputArg_;
  }
  if (weightLayer_) {
    const MatrixPtr& weight = getInputValue(*weightLayer_);
    getOutputValue()->dotMul(*getOutputValue(), *weight);
  }
}

void CRFLayer::backward(const UpdateCallback& callback) {
  const Argument& output = getInput(0);
  const Argument& label = getInput(1);
  const int* starts = label.sequenceStartPositions->getData(false);
  int numSequences = label.sequenceStartPositions->getSize() - 1;

  bool needWGrad = weight_->getWGrad() ? true : false;
  MatrixPtr output_arg_grad = output.grad;
  MatrixPtr weight_grad = weight_->getWGrad();
  MatrixPtr output_arg_val = output.value;
  IVectorPtr label_val = label.ids;
  if (useGpu_) {
    cpuOutputArg_->copyFrom(*output_arg_val);
    cpuLabel_->copyFrom(*label_val);
    if (output_arg_grad) {
      Matrix::resizeOrCreate(cpuOutputArgGrad_,
                             /* height */ output_arg_grad->getHeight(),
                             /* width */ output_arg_grad->getWidth(),
                             /* trans */ false,
                             /* useGpu */ false);
      cpuOutputArgGrad_->copyFrom(*output_arg_grad);
    }
    if (needWGrad) {
      Matrix::resizeOrCreate(cpuWeightGrad_,
                             /* height */ weight_grad->getHeight(),
                             /* width */ weight_grad->getWidth(),
                             /* trans */ false,
                             /* useGpu */ false);
      cpuWeightGrad_->copyFrom(*weight_grad);
    }
  } else {
    cpuOutputArg_ = output_arg_val;
    cpuLabel_ = label_val;
    if (output_arg_grad) {
      cpuOutputArgGrad_ = output_arg_grad;
    }
    if (needWGrad) {
      cpuWeightGrad_ = weight_grad;
    }
  }
  for (int i = 0; i < numSequences; ++i) {
    crfs_[i].backward(cpuOutputArg_->getData() + numClasses_ * starts[i],
                      cpuLabel_->getData() + starts[i],
                      starts[i + 1] - starts[i],
                      needWGrad);
    real instanceWeight = weightLayer_
                              ? getInputValue(*weightLayer_)->getElement(i, 0)
                              : real(1.0f);
    instanceWeight *= coeff_;

    if (output.grad) {
      MatrixPtr grad =
          cpuOutputArgGrad_->subRowMatrix(starts[i], starts[i + 1]);
      grad->add(*crfs_[i].getXGrad(), real(1.0f), instanceWeight);
    }
    if (needWGrad) {
      cpuWeightGrad_->add(*crfs_[i].getWGrad(), real(1.0f), instanceWeight);
    }
  }
  if (useGpu_) {
    if (output.grad) {
      output_arg_grad->copyFrom(*cpuOutputArgGrad_);
    }
    if (needWGrad) {
      weight_grad->copyFrom(*cpuWeightGrad_);
    }
    output_arg_val->copyFrom(*cpuOutputArg_);
  } else {
    if (output.grad) {
      output_arg_grad = cpuOutputArgGrad_;
    }
    if (needWGrad) {
      weight_grad = cpuWeightGrad_;
    }
    output_arg_val = cpuOutputArg_;
  }

  parameter_->incUpdate(callback);
}

}  // namespace paddle
