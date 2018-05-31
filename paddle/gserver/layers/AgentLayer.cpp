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

#include "AgentLayer.h"

#include "paddle/utils/Logging.h"

#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_LAYER(agent, AgentLayer);

bool AgentLayer::init(const LayerMap& layerMap,
                      const ParameterMap& parameterMap) {
  CHECK_EQ(config_.inputs_size(), 0);
  if (!Layer::init(layerMap, parameterMap)) {
    return false;
  }
  setNeedGradient(true);
  return true;
}

void AgentLayer::forward(PassType passType) {
  Layer::forward(passType);

  Argument& realOutput = realLayer_->getOutput();
  int realNumSequences = realOutput.getNumSequences();
  CHECK_LE(numSamples_, realNumSequences);

  // get Arguments from real layers
  if (numSamples_ > 0 && numSamples_ < realNumSequences) {
    if (realOutput.hasSeq()) {
      int numRows =
          realOutput.sequenceStartPositions->getData(false)[numSamples_];
      output_.subArgFrom(realOutput,
                         /* offset */ 0,
                         numRows,
                         getSize(),
                         useGpu_,
                         /* trans */ false,
                         /* seqFlag */ true,
                         /* seqStart */ 0,
                         /* seqSize */ numSamples_ + 1);
    } else {
      output_.subArgFrom(
          realOutput, /* offset */ 0, numSamples_, getSize(), useGpu_);
    }
  } else {
    output_ = realOutput;
  }
}

bool GatherAgentLayer::init(const LayerMap& layerMap,
                            const ParameterMap& parameterMap) {
  CHECK_EQ(config_.inputs_size(), 0);
  if (!Layer::init(layerMap, parameterMap)) {
    return false;
  }
  setNeedGradient(true);
  return true;
}

void GatherAgentLayer::copyIdAndSequenceInfo(
    ICpuGpuVectorPtr sequenceStartPositions,
    ICpuGpuVectorPtr subSequenceStartPositions,
    const IVectorPtr& ids,
    const std::vector<int>& idIndex) {
  output_.sequenceStartPositions = sequenceStartPositions;
  output_.subSequenceStartPositions = subSequenceStartPositions;
  allIds_ = ids;
  idIndex_ = idIndex;
}

void GatherAgentLayer::forward(PassType passType) {
  Layer::forward(passType);
  forwardIds(passType);
  forwardValue(passType);
}

void GatherAgentLayer::forwardValue(PassType passType) {
  MatrixPtr valueReal = realLayers_[0]->getOutputValue();
  if (!valueReal) return;

  int height = allIds_->getSize();
  int width = this->getSize();
  resetOutput(height, width);
  idsVec_.resize(idIndex_.size());

  const MatrixPtr& outV = getOutputValue();

  for (size_t i = 0; i < realLayers_.size(); ++i) {
    const MatrixPtr& realV = realLayers_[i]->getOutputValue();
    idsVec_[i] = IVector::create(allIds_->getData() + idIndex_[i],
                                 /* size */ realV->getHeight(),
                                 useGpu_);
    realV->addToRows(*outV, *idsVec_[i]);
  }
}

namespace {

// dest[index[i]] <- src[i] for each i
void copyElements(const IVector& srcVec,
                  const IVector& indexVec,
                  IVector& destVec) {
  const int* src = srcVec.getData();
  const int* index = indexVec.getData();
  int* dest = destVec.getData();
  int len = indexVec.getSize();
  CHECK_EQ(srcVec.getSize(), indexVec.getSize());
  for (int i = 0; i < len; ++i) {
    dest[index[i]] = src[i];
  }
}
}  // namespace

void GatherAgentLayer::forwardIds(PassType passType) {
  IVectorPtr realId = realLayers_[0]->getOutputLabel();
  if (!realId) return;

  IVector::resizeOrCreate(output_.ids, allIds_->getSize(), useGpu_);
  IVectorPtr outId = output_.ids;
  idsVec_.resize(idIndex_.size());

  for (size_t i = 0; i < realLayers_.size(); ++i) {
    const IVectorPtr& realId = realLayers_[i]->getOutputLabel();
    idsVec_[i] = IVector::create(allIds_->getData() + idIndex_[i],
                                 /* size */ realId->getSize(),
                                 useGpu_);
    execViaCpu(&copyElements, *realId, *idsVec_[i], *outId);
  }
}

void GatherAgentLayer::backward(const UpdateCallback& callback) {
  (void)callback;
  const MatrixPtr& outputGrad = getOutputGrad();

  for (size_t i = 0; i < realLayers_.size(); ++i) {
    const MatrixPtr& realG = realLayers_[i]->getOutputGrad();
    if (realG) {
      realG->selectRows(*outputGrad, *idsVec_[i]);
    }
  }
}

bool ScatterAgentLayer::init(const LayerMap& layerMap,
                             const ParameterMap& parameterMap) {
  CHECK_EQ(config_.inputs_size(), 0);
  if (!Layer::init(layerMap, parameterMap)) {
    return false;
  }
  setNeedGradient(true);
  return true;
}

void ScatterAgentLayer::forward(PassType passType) {
  Layer::forward(passType);
  CHECK_EQ(realLayer_->getDeviceId(), this->getDeviceId());

  int width = this->getSize();
  if (selectionMode_) {
    forwardWithSelection(passType);
  } else {
    if (realOutArg_.hasSeq()) {
      output_.subArgFrom(realOutArg_,
                         /* offset */ idIndex_,
                         idSize_,
                         width,
                         useGpu_,
                         /* trans */ false,
                         /* seqFlag */ true,
                         /* seqStart */ seqStartPosIndex_,
                         /* seqSize */ numSequences_);
    } else {
      output_.subArgFrom(
          realOutArg_, /* offset */ idIndex_, idSize_, width, useGpu_);
    }
  }
}

void ScatterAgentLayer::backward(const UpdateCallback& callback) {
  (void)callback;

  CHECK(!selectionMode_);

  const MatrixPtr& outputGrad = realOutArg_.grad;
  const MatrixPtr& realGrad = realLayer_->getOutputGrad();
  if (realGrad) {
    // for agent in inFrameLines and memoryFrameLines,
    // only first scatterAgentLayer should do addToRows in backward
    if (handleBackward_) {
      outputGrad->addToRows(*realGrad, *ids_);
    }
  }
}

REGISTER_LAYER(gather_agent, GatherAgentLayer);
REGISTER_LAYER(scatter_agent, ScatterAgentLayer);

void ScatterAgentLayer::forwardWithSelection(PassType passType) {
  Layer::forward(passType);
  CHECK_EQ(realLayer_->getDeviceId(), this->getDeviceId());

  const Argument& input = realLayer_->getOutput();
  CHECK_EQ(realLayer_->getSize(), this->getSize());
  int width = this->getSize();

  AsyncGpuBlock asyncGpuBlock;
  REGISTER_TIMER_INFO("SequenceAgentLayerForward", getName().c_str());

  if (!input.hasSeq()) {
    if (realLayer_->getOutput().ids) {
      IVector::resizeOrCreate(output_.ids, ids_->getSize(), useGpu_);
      output_.ids->selectFrom(*realLayer_->getOutput().ids, *ids_);
    }
    if (realLayer_->getOutput().value) {
      int height = ids_->getSize();
      resetOutput(height, width);

      const MatrixPtr& outV = getOutputValue();
      const MatrixPtr& realV = realLayer_->getOutputValue();
      outV->selectRows(*realV, *ids_);
    }
  } else {
    // Putting the generation logic here is really an ugly hack!
    // used in generation
    int height = 0;
    size_t numSequences = ids_->getSize();
    const int* starts = input.getCpuStartPositions();
    size_t size = input.hasSubseq() ? input.getNumSubSequences()
                                    : input.getNumSequences();
    const int* cpuIds = cpuIds_->getData();

    for (size_t i = 0; i < numSequences; ++i) {
      size_t seqId = cpuIds[i];
      CHECK_LT(seqId, size);
      height += starts[seqId + 1] - starts[seqId];
    }
    reserveOutput(height, width);

    const MatrixPtr& outputValue = getOutputValue();

    CHECK_NE(input.sequenceStartPositions.get(),
             output_.sequenceStartPositions.get());
    ICpuGpuVector::resizeOrCreate(
        output_.sequenceStartPositions, numSequences + 1, false);
    int* outStarts = output_.sequenceStartPositions->getMutableData(false);

    ICpuGpuVector::resizeOrCreate(inputStartPos_, height, false);
    int* inStarts = inputStartPos_->getMutableData(false);

    size_t offsetOut = 0;
    for (size_t i = 0; i < numSequences; ++i) {
      outStarts[i] = offsetOut;
      size_t seqId = cpuIds[i];
      int size = starts[seqId + 1] - starts[seqId];
      for (int j = 0; j < size; j++) {
        inStarts[offsetOut + j] = starts[seqId] + j;
      }
      offsetOut += size;
    }
    outStarts[numSequences] = offsetOut;

    outputValue->copyByRowIndex(*input.value,
                                *inputStartPos_->getVector(useGpu_));
  }
}

}  // namespace paddle
