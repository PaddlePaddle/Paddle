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

#include "paddle/utils/Util.h"

#include "CostLayer.h"
#include "paddle/math/SparseMatrix.h"
#include "paddle/utils/Error.h"
#include "paddle/utils/Logging.h"

#ifndef PADDLE_MOBILE_INFERENCE
#include "ValidationLayer.h"
#endif

DEFINE_bool(log_error_clipping, false, "enable log error clipping or not");

namespace paddle {

Layer::Layer(const LayerConfig& config, bool useGpu)
    : config_(config),
      useGpu_(useGpu),
      deviceId_(CPU_DEVICE),
      needSequenceInfo_(true) {}

bool Layer::init(const LayerMap& layerMap, const ParameterMap& parameterMap) {
  if (useGpu_ && FLAGS_parallel_nn) {
    /* gpu environment is specified by device property */
    deviceId_ = config_.device();
    if (deviceId_ < 0) {
      useGpu_ = false;
    }
  }

  output_.deviceId = deviceId_;

  for (auto& inputConfig : config_.inputs()) {
    std::string inputName = inputConfig.input_layer_name();
    LayerPtr inputLayer;
    CHECK(mapGet(inputName, layerMap, &inputLayer))
        << "Cannot find input layer " << inputName << " for layer "
        << getName();
    this->addPrev(inputLayer);

    inputLayer->addOutputArgument(deviceId_);

    if (inputConfig.has_input_parameter_name()) {
      ParameterPtr parameter;
      CHECK(
          mapGet(inputConfig.input_parameter_name(), parameterMap, &parameter))
          << "Cannot find input parameter "
          << inputConfig.input_parameter_name() << " for layer " << getName();
      parameter->incShared();
      CHECK_EQ(parameter->getDeviceId(), getDeviceId());
      parameters_.push_back(parameter);
    } else {
      parameters_.push_back(nullptr);
    }

    if (inputConfig.has_input_layer_argument()) {
      inputArgument_.push_back(inputConfig.input_layer_argument());
    } else {
      inputArgument_.push_back("");
    }
  }

  if (config_.has_bias_parameter_name()) {
    CHECK(mapGet(config_.bias_parameter_name(), parameterMap, &biasParameter_))
        << "Cannot find bias parameter " << config_.bias_parameter_name()
        << " for layer " << getName();
    biasParameter_->incShared();
    CHECK_EQ(biasParameter_->getDeviceId(), getDeviceId());
  }

  /* specify the activation function according to the configuration */
  std::string action_type = config_.active_type();
  activation_.reset(ActivationFunction::create(action_type));
  CHECK(activation_);

  initNeedFlags();
  markInBackward_.assign(inputLayers_.size(), false);

  return true;
}

ClassRegistrar<Layer, LayerConfig> Layer::registrar_;

LayerPtr Layer::create(const LayerConfig& config) {
  std::string type = config.type();

#ifndef PADDLE_MOBILE_INFERENCE
  // NOTE: As following types have illegal character '-',
  // they can not use REGISTER_LAYER to registrar.
  // Besides, to fit with old training models,
  // they can not use '_' instead.
  if (type == "multi-class-cross-entropy")
    return LayerPtr(new MultiClassCrossEntropy(config));
  else if (type == "rank-cost")
    return LayerPtr(new RankingCost(config));
  else if (type == "auc-validation")
    return LayerPtr(new AucValidation(config));
  else if (type == "pnpair-validation")
    return LayerPtr(new PnpairValidation(config));
#endif

  return LayerPtr(registrar_.createByType(config.type(), config));
}

void Layer::resetSpecifyOutput(Argument& output,
                               size_t height,
                               size_t width,
                               bool isValueClean,
                               bool isGradClean) {
  SetDevice device(output.deviceId);

  Matrix::resizeOrCreate(
      output.value, height, width, /* trans */ false, useGpu(output.deviceId));
  if (isValueClean) {
    output.value->zeroMem();
  }

  if (passType_ != PASS_TEST && needGradient()) {
    Matrix::resizeOrCreate(
        output.grad, height, width, /* trans */ false, useGpu(output.deviceId));
    if (isGradClean) {
      output.grad->zeroMem();
    }
  }
}

void Layer::resizeOutput(size_t height, size_t width) {
  resetSpecifyOutput(output_, height, width, false, false);

  for (size_t i = 0; i != outputOtherDevice_.size(); i++) {
    resetSpecifyOutput(outputOtherDevice_[i], height, width, false, false);
  }
}

void Layer::reserveOutput(size_t height, size_t width) {
  resetSpecifyOutput(output_, height, width, false, true);

  for (size_t i = 0; i != outputOtherDevice_.size(); i++) {
    resetSpecifyOutput(outputOtherDevice_[i], height, width, false, true);
  }
}

void Layer::resetOutput(size_t height, size_t width) {
  resetSpecifyOutput(output_, height, width, true, true);

  for (size_t i = 0; i != outputOtherDevice_.size(); i++) {
    resetSpecifyOutput(outputOtherDevice_[i], height, width, true, true);
  }
}

void Layer::addOutputArgument(int deviceId) {
  if (deviceId == deviceId_) {
    output_.countIncrement();
    return;
  } else {
    for (size_t i = 0; i < outputOtherDevice_.size(); i++) {
      if (outputOtherDevice_[i].deviceId == deviceId) {
        outputOtherDevice_[i].countIncrement();
        return;
      }
    }
  }

  Argument argu;
  argu.deviceId = deviceId;
  outputOtherDevice_.push_back(argu);
  outputOtherDevice_.back().countIncrement();
}

void Layer::copyOutputToOtherDevice() {
  for (size_t i = 0; i != outputOtherDevice_.size(); i++) {
    SetDevice device(outputOtherDevice_[i].deviceId);
    // If outputOtherDevice_[i].value is a CpuMatrix,
    // the copyFrom is a synchronous interface.
    // If outputOtherDevice_[i].value is a GpuMatrix, since subsequent
    // calculations are all on HPPL_STREAM_DEFAULT,
    // copyFrom can be an asynchronous interface.
    outputOtherDevice_[i].value->copyFrom(*getOutputValue(),
                                          HPPL_STREAM_DEFAULT);
    outputOtherDevice_[i].sequenceStartPositions =
        output_.sequenceStartPositions;
    outputOtherDevice_[i].subSequenceStartPositions =
        output_.subSequenceStartPositions;
    outputOtherDevice_[i].cpuSequenceDims = output_.cpuSequenceDims;

    outputOtherDevice_[i].notifyValueReady();
  }
}

void Layer::waitInputValue() {
  for (size_t i = 0; i != inputLayers_.size(); i++) {
    if (inputLayers_[i]->getDeviceId() != deviceId_) {
      getInput(i).waitValueReady();
    }
  }
}

void Layer::waitAndMergeOutputGrad() {
  if (!output_.grad || !outputOtherDevice_.size()) {
    return;
  }

  for (size_t i = 0; i != outputOtherDevice_.size(); i++) {
    outputOtherDevice_[i].waitGradReady();
  }

  /* merge output grad */
  size_t i = 0;
  if (!output_.getAllCount()) {
    output_.grad->copyFrom(*outputOtherDevice_[0].grad, HPPL_STREAM_1);
    hl_stream_synchronize(HPPL_STREAM_1);

    i++;
    if (outputOtherDevice_.size() == 1) return;
  }

  Matrix::resizeOrCreate(tmpGrad_,
                         output_.grad->getHeight(),
                         output_.grad->getWidth(),
                         /* trans */ false,
                         useGpu(output_.deviceId));

  for (; i != outputOtherDevice_.size(); i++) {
    tmpGrad_->copyFrom(*outputOtherDevice_[i].grad, HPPL_STREAM_1);
    hl_stream_synchronize(HPPL_STREAM_1);
    output_.grad->add(*tmpGrad_);
  }
}

void Layer::markAllInputGrad() {
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    if (!markInBackward_[i]) {
      inputLayers_[i]->getOutput(deviceId_).notifyGradReady();
    }
    markInBackward_[i] = false;
  }
}

void Layer::markInputGrad(int inputIndex) {
  inputLayers_[inputIndex]->getOutput(deviceId_).notifyGradReady();
  markInBackward_[inputIndex] = true;
}

void Layer::zeroGrad() {
  CHECK(output_.grad.get() != NULL);
  output_.grad->zeroMem();
}

void Layer::initNeedFlags() {
  auto initFlag = [this](
      bool& flag, bool (Layer::*flagQueryFunc)() const, ParameterType type) {
    flag = false;
    if (biasParameter_ && biasParameter_->hasType(type)) {
      flag = true;
    }
    if (!flag) {
      for (auto& para : parameters_) {
        if (para && para->hasType(type)) {
          flag = true;
          break;
        }
      }
    }
    if (!flag) {
      for (auto& layer : inputLayers_) {
        if ((layer.get()->*flagQueryFunc)()) {
          flag = true;
        }
      }
    }
  };
  initFlag(needGradient_, &Layer::needGradient, PARAMETER_GRADIENT);
}

void Layer::showOutputStats() {
  MatrixPtr out = getOutputValue();
  if (!out) return;
  if (!out->getElementCnt()) {
    LOG(INFO) << "The number of output of " << config_.name()
              << " is 0, skip to show the statistics";
    return;
  }
  MatrixPtr outSquare;
  if (dynamic_cast<GpuSparseMatrix*>(out.get())) {
    GpuSparseMatrix* tmp = dynamic_cast<GpuSparseMatrix*>(out.get());
    outSquare = std::make_shared<CpuSparseMatrix>(tmp->getHeight(),
                                                  tmp->getWidth(),
                                                  tmp->getElementCnt(),
                                                  tmp->getValueType(),
                                                  tmp->getFormat());
  } else {
    outSquare = out->clone();
  }
  outSquare->copyFrom(*out, HPPL_STREAM_DEFAULT);
  hl_stream_synchronize(HPPL_STREAM_DEFAULT);

  real mean = outSquare->getSum() / out->getElementCnt();
  real min;
  real max;
  if (dynamic_cast<CpuSparseMatrix*>(outSquare.get())) {
    auto tmpMat = dynamic_cast<CpuSparseMatrix*>(outSquare.get());
    min = tmpMat->getMin();
    max = tmpMat->getMax();
    tmpMat->square2();
    LOG(INFO) << "show statistics of [none zero values] in sparse matrix";
  } else {
    min = outSquare->getMin();
    max = outSquare->getMax();
    outSquare->square2();
  }
  real std = (outSquare->getSum() / outSquare->getElementCnt()) - mean * mean;
  std = std > 0 ? std : 0;
  LOG(INFO) << "The output state of " << config_.name() << ": mean=" << mean
            << ", "
            << "std=" << std << ", "
            << "min=" << min << ", "
            << "max=" << max;
}

void Layer::forwardActivation() {
  /* activation */
  auto status = activation_->forward(output_);
  status.check();

  /* dropout */
  if (config_.drop_rate() > 0) {
    forwardDropOut();
    CHECK_NE(activation_->getName(), "softmax")
        << "Softmax activation cannot be used with Dropout";
  }

  if (FLAGS_show_layer_stat) {
    showOutputStats();
  }
}

void Layer::backwardActivation() {
  /* Do error clipping */
  if (config_.error_clipping_threshold() > 0.0f) {
    if (FLAGS_log_error_clipping) {
      VectorPtr outGradVec = Vector::create(
          output_.grad->getData(), output_.grad->getElementCnt(), useGpu_);
      real maxAbsGrad = outGradVec->getAbsMax();
      if (maxAbsGrad > config_.error_clipping_threshold()) {
        real avgAbsGrad = outGradVec->getAbsSum() / outGradVec->getSize();
        LOG(INFO) << " layer=" << config_.name() << " need clipping,"
                  << " max error=" << maxAbsGrad << " avg error=" << avgAbsGrad;
      }
    }
    output_.grad->clip(-config_.error_clipping_threshold(),
                       config_.error_clipping_threshold());
  }

  /* Do dropout for delta*/
  if (config_.drop_rate() > 0 && passType_ != PASS_TEST) {
    MatrixPtr oGrad = getOutputGrad();
    oGrad->dotMul(*oGrad, *dropOutMask_);
  }

  auto status = activation_->backward(output_);
  status.check();
}

void Layer::forwardDropOut() {
  auto& outV = getOutputValue();

  if (passType_ == PASS_TRAIN) {
    // new dropOutMask_ if dropOutMask_ is null ptr
    Matrix::resizeOrCreate(dropOutMask_,
                           outV->getHeight(),
                           outV->getWidth(),
                           false,
                           useGpu(deviceId_));
    dropOutMask_->randomizeUniform();  // generate a uniform random matrix
    dropOutMask_->biggerThanScalar(config_.drop_rate());  // random mask
    outV->dotMul(*outV, *dropOutMask_);                   // dropout
  } else if (passType_ == PASS_GC) {
    // only initialize once
    if (!dropOutMask_) {
      dropOutMask_ = Matrix::create(
          outV->getHeight(), outV->getWidth(), false, useGpu(deviceId_));
      // We use cpu matrix to generate mask so that the mask
      // will be same for both gpu version and cpu version.
      // This will help unittest to make sure they have same result.
      MatrixPtr tmpMask = Matrix::create(outV->getHeight(), outV->getWidth());
      tmpMask->randomizeUniform();  // generate a uniform random matrix
      tmpMask->biggerThanScalar(config_.drop_rate());  // random mask
      dropOutMask_->copyFrom(*tmpMask);
    }
    outV->dotMul(*outV, *dropOutMask_);
  } else {  // passType == PASS_TEST
    outV->mulScalar(1.0 - config_.drop_rate());
  }
}

}  // namespace paddle
