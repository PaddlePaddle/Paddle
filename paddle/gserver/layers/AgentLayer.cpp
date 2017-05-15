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
  int realHeight = realOutput.getBatchSize();
  CHECK_LE(numSamples_, realHeight);

  // get Arguments from real layers
  if (numSamples_ > 0 && numSamples_ < realHeight) {
    if (realOutput.ids) {
      output_.ids =
          IVector::create(realOutput.ids->getData(), numSamples_, useGpu_);
    } else {
      output_.subArgFrom(
          realOutput, /* offset */ 0, numSamples_, getSize(), useGpu_);
    }
  } else {
    output_ = realOutput;
  }
}

void SequenceAgentLayer::forward(PassType passType) {
  Layer::forward(passType);

  Argument& realOutput = realLayer_->getOutput();
  int realNumSequences = realOutput.getNumSequences();
  CHECK_LE(numSamples_, realNumSequences);

  // get Arguments from real layers
  if (numSamples_ > 0 && numSamples_ < realNumSequences) {
    int numRows =
        realOutput.sequenceStartPositions->getData(false)[numSamples_];
    CHECK(!realOutput.ids) << "Not supported";
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
    output_ = realOutput;
  }
}

REGISTER_LAYER(sequence_agent, SequenceAgentLayer);

bool GatherAgentLayer::init(const LayerMap& layerMap,
                            const ParameterMap& parameterMap) {
  CHECK_EQ(config_.inputs_size(), 0);
  if (!Layer::init(layerMap, parameterMap)) {
    return false;
  }
  setNeedGradient(true);
  return true;
}

void GatherAgentLayer::copyIdAndSequenceInfo(const Argument& input,
                                             const IVectorPtr& ids,
                                             const std::vector<int>& idIndex) {
  output_.sequenceStartPositions = input.sequenceStartPositions;
  output_.subSequenceStartPositions = input.subSequenceStartPositions;
  realLayers_.clear();
  allIds_ = ids;
  idIndex_ = idIndex;
}

void GatherAgentLayer::forward(PassType passType) {
  Layer::forward(passType);

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
  if (realOutArg_.value || realOutArg_.ids) {
    output_.subArgFrom(
        realOutArg_, /* offset */ idIndex_, idSize_, width, useGpu_);
  } else {  // used in generation
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
  }
}

void ScatterAgentLayer::backward(const UpdateCallback& callback) {
  (void)callback;

  const MatrixPtr& outputGrad = realOutArg_.grad;
  const MatrixPtr& realGrad = realLayer_->getOutputGrad();
  if (realGrad) {
    // for agent in inFrameLines and memoryFrameLines,
    // only first scatterAgentLayer should do addToRows in backward
    if (idIndex_ == 0) {
      outputGrad->addToRows(*realGrad, *ids_);
    }
  }
}

REGISTER_LAYER(gather_agent, GatherAgentLayer);
REGISTER_LAYER(scatter_agent, ScatterAgentLayer);

void SequenceGatherAgentLayer::forward(PassType passType) {
  Layer::forward(passType);
  int height = 0;
  int* starts = output_.subSequenceStartPositions->getMutableData(false);
  IVectorPtr idReal = realLayers_[0]->getOutputLabel();
  if (idReal) {
    // Gather generator.idsVec
    // if is beam search generation result. Get first result.
    if (idReal->getData()[idReal->getSize() - 1] == -1) {
      for (size_t i = 0; i < realLayers_.size(); ++i) {
        // The first element stores first result size
        idReal = realLayers_[i]->getOutputLabel();
        idReal->subVecFrom(*idReal, 1, idReal->getData()[0]);
      }
    }
    for (size_t i = 0; i < realLayers_.size(); ++i) {
      CHECK(realLayers_[i]->getOutputLabel());
      starts[i] = height;
      height += realLayers_[i]->getOutputLabel()->getSize();
    }
    starts[realLayers_.size()] = height;
    output_.sequenceStartPositions->getMutableData(false)[1] = height;

    IVector::resizeOrCreate(output_.ids, height, false);
    for (size_t i = 0; i < realLayers_.size(); ++i) {
      output_.ids->subVec(starts[i], starts[i + 1] - starts[i])
          ->copyFrom(*realLayers_[i]->getOutputLabel());
    }
  } else {
    // Gather output.value, same as GatherAgentLayer
    CHECK(output_.subSequenceStartPositions);
    GatherAgentLayer::forward(passType);
  }
}

void SequenceScatterAgentLayer::forward(PassType passType) {
  Layer::forward(passType);
  CHECK_EQ(realLayer_->getDeviceId(), this->getDeviceId());

  const Argument& input = realLayer_->getOutput();
  CHECK_EQ(realLayer_->getSize(), this->getSize());
  int width = this->getSize();

  AsyncGpuBlock asyncGpuBlock;
  REGISTER_TIMER_INFO("SequenceAgentLayerForward", getName().c_str());

  if (realOutArg_.value || realOutArg_.ids) {
    CHECK(realOutArg_.sequenceStartPositions);
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

REGISTER_LAYER(sequence_gather_agent, SequenceGatherAgentLayer);
REGISTER_LAYER(sequence_scatter_agent, SequenceScatterAgentLayer);

}  // namespace paddle
