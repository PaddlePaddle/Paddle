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

#include "LstmLayer.h"
#include "paddle/math/BaseMatrix.h"
#include "paddle/math/Matrix.h"

namespace paddle {

class CoordIterator {
 public:
  std::vector<int> dims_;
  std::vector<bool> directions_;
  std::vector<int> curPos_;
  bool end_;

  void step(size_t d, bool reversed) {
    if (directions_[d] ^ reversed) {
      if (curPos_[d] == dims_[d] - 1) {
        curPos_[d] = 0;
        if (d) {
          step(d - 1, reversed);
        } else {
          end_ = true;
        }
      } else {
        curPos_[d]++;
      }
    } else {
      if (curPos_[d] == 0) {
        curPos_[d] = dims_[d] - 1;
        if (d) {
          step(d - 1, reversed);
        } else {
          end_ = true;
        }
      } else {
        curPos_[d]--;
      }
    }
  }

 public:
  CoordIterator(std::vector<int> dim, std::vector<bool> directions)
      : dims_(dim), directions_(directions), end_(false) {
    CHECK_EQ(dims_.size(), directions_.size());
    for (size_t i = 0; i < dims_.size(); i++) {
      curPos_.push_back(-1);
    }
  }
  CoordIterator& operator++() {
    step(dims_.size() - 1, false);
    return *this;
  }

  CoordIterator& operator--() {
    step(dims_.size() - 1, true);
    return *this;
  }

  std::vector<int>& curPos() { return curPos_; }

  int offset() {
    int offset = curPos_[0];
    for (size_t i = 1; i < dims_.size(); i++) {
      offset = offset * dims_[i] + curPos_[i];
    }
    return offset;
  }

  int offset(const std::vector<int>& pos) {
    int offset = pos[0];
    for (size_t i = 1; i < dims_.size(); i++) {
      offset = offset * dims_[i] + pos[i];
    }
    return offset;
  }

  std::vector<int>& begin() {
    for (size_t i = 0; i < dims_.size(); i++) {
      curPos_[i] = directions_[i] ? 0 : dims_[i] - 1;
    }
    end_ = false;
    return curPos_;
  }

  std::vector<int>& rbegin() {
    for (size_t i = 0; i < dims_.size(); i++) {
      curPos_[i] = directions_[i] ? dims_[i] - 1 : 0;
    }
    end_ = false;
    return curPos_;
  }

  bool end() { return end_; }

  bool getPrePos(const std::vector<int>& delays,
                 int idx,
                 std::vector<int>& prePos) {
    bool isAvial = true;
    prePos.clear();
    prePos.reserve(directions_.size());
    for (size_t i = 0; i < directions_.size(); i++) {
      if (int(i) == idx) {
        prePos.push_back(curPos_[i] + delays[i] * (directions_[i] ? 1 : -1));
        if (prePos[i] < 0) {
          prePos[i] = 0;
          isAvial = false;
        }
        if (prePos[i] >= dims_[i]) {
          prePos[i] = dims_[i] - 1;
          isAvial = false;
        }
      } else {
        prePos.push_back(curPos_[i]);
      }
    }
    return isAvial;
  }

  bool getNextPos(const std::vector<int>& delays,
                  int idx,
                  std::vector<int>& nextPos) {
    bool isAvial = true;
    nextPos.clear();
    nextPos.reserve(directions_.size());
    for (size_t i = 0; i < directions_.size(); i++) {
      if (int(i) == idx) {
        nextPos.push_back(curPos_[i] - delays[i] * (directions_[i] ? 1 : -1));
        if (nextPos[i] < 0) {
          nextPos[i] = 0;
          isAvial = false;
        }
        if (nextPos[i] >= dims_[i]) {
          nextPos[i] = dims_[i] - 1;
          isAvial = false;
        }
      } else {
        nextPos.push_back(curPos_[i]);
      }
    }
    return isAvial;
  }
};
/*
 * MDLstmLayer takes 1 input layer with size * (3+numDims).
 * For each sequence [start, end] it performs the following computation:
 * out_i = actState(state_i) * actGate(outputGate_i)
 *
 * For example the image with 2 dims, we take the scanning order from left-top
 * to right-bottom, then the 2 previous states of the current pixels are the
 * ones located at left and top. And each of them has a independent forget gate.
 *
 * state_i = actInput(input_i) * actGate(inputGate_i) +
 *           \sum{j}(actGate(forgetGate_i_j) * state_prev_i_j)
 *
 * inputGate = input_i * inputW + \sum{j}(output_prev_i_j * recurrInputW_j) +
 *             \sum{j}(state_prev_i_j * inputCheck_j)
 *
 * ouputGate = input_i * outputW + \sum{j}(output_prev_i_j * recurrOutputW_j) +
 *             state_i * outputCheck
 *
 * forgetGate_j = input_i * forgetW_j + \sum{j}(output_prev_i_j *
 *                recurrForgetW_j) + \sum{j}(state_prev_i_j * forgetCheck_j)
 *
 * IG Layer: (Input, InputGate, ForgetGates, OutputGate) * OutputSize
 * */

class MDLstmLayer : public LstmLayer {
 public:
  explicit MDLstmLayer(const LayerConfig& config) : LstmLayer(config) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;

  void backward(const UpdateCallback& callback) override;

 protected:
  void forwardOneSequence(int start, CoordIterator& coordIter);
  void backwardOneSequence(int start, CoordIterator& coordIter);
  void forwardGate2OutputSequence(int start, CoordIterator& coordIter);
  void backwardGate2OutputSequence(int start, CoordIterator& coordIter);

 protected:
  std::vector<Argument> frameInputGate_;
  std::vector<Argument> frameForgetGate_;
  std::vector<Argument> frameOutputGate_;
  std::vector<Argument> frameInputNode_;
  std::vector<Argument> frameGate_;
  std::vector<Argument> frameState_;
  std::vector<Argument> framePreOutput_;
  std::vector<Argument> frameOutput_;

  // Activation
  std::unique_ptr<ActivationFunction> activationGate_;
  std::unique_ptr<ActivationFunction> activationState_;

  int numDims_;
  size_t numBlocks_;
  std::vector<bool> directions_;
  std::vector<int> delays_;
  std::vector<std::vector<int>> dimsV_;
};

REGISTER_LAYER(mdlstmemory, MDLstmLayer);

bool MDLstmLayer::init(const LayerMap& layerMap,
                       const ParameterMap& parameterMap) {
  if (!Layer::init(layerMap, parameterMap)) return false;
  CHECK_EQ(1U, inputLayers_.size());
  CHECK_EQ(1U, parameters_.size());

  numBlocks_ = getSize();
  numDims_ = config_.directions_size();
  CHECK_EQ(numBlocks_ * numBlocks_ * (3 + numDims_), parameters_[0]->getSize());

  // inode(1), ig(1), fg(numDims_), og(1), peepIg(1), peepFg(numDims_),
  // peepOg(1), then size of localBias_ is 3+numDims_
  CHECK_EQ(numBlocks_ * (5 + 2 * numDims_), biasParameter_->getSize());
  weight_.reset(
      new Weight(numBlocks_, numBlocks_ * (3 + numDims_), parameters_[0]));
  if (biasParameter_.get() != NULL) {
    bias_.reset(new Weight(1, numBlocks_ * (5 + 2 * numDims_), biasParameter_));
    localBias_ = Matrix::create(nullptr,
                                /* height= */ 1,
                                numBlocks_ * (3 + numDims_),
                                /* trans= */ false,
                                useGpu_);
    checkIg_ = Matrix::create(nullptr,
                              /* height= */ 1,
                              numBlocks_,
                              /* trans= */ false,
                              useGpu_);
    checkFg_ = Matrix::create(nullptr,
                              /* height= */ numDims_,
                              numBlocks_,
                              /* trans= */ false,
                              useGpu_);
    checkOg_ = Matrix::create(nullptr,
                              /* height= */ 1,
                              numBlocks_,
                              /* trans= */ false,
                              useGpu_);
    localBiasGrad_ = Matrix::create(nullptr,
                                    /* height= */ 1,
                                    numBlocks_ * (3 + numDims_),
                                    /* trans= */ false,
                                    useGpu_);
    checkIgGrad_ = Matrix::create(nullptr,
                                  /* height= */ 1,
                                  numBlocks_,
                                  /* trans= */ false,
                                  useGpu_);
    checkFgGrad_ = Matrix::create(nullptr,
                                  /* height= */ numDims_,
                                  numBlocks_,
                                  /* trans= */ false,
                                  useGpu_);
    checkOgGrad_ = Matrix::create(nullptr,
                                  /* height= */ 1,
                                  numBlocks_,
                                  /* trans= */ false,
                                  useGpu_);

    localBias_->setData(bias_->getW()->getData());
    checkIg_->setData(bias_->getW()->getData() + numBlocks_ * (3 + numDims_));
    checkFg_->setData(bias_->getW()->getData() + numBlocks_ * (4 + numDims_));
    checkOg_->setData(bias_->getW()->getData() +
                      numBlocks_ * (4 + 2 * numDims_));

    if (bias_->getWGrad()) {
      localBiasGrad_->setData(bias_->getWGrad()->getData());
      checkIgGrad_->setData(bias_->getWGrad()->getData() +
                            numBlocks_ * (3 + numDims_));
      checkFgGrad_->setData(bias_->getWGrad()->getData() +
                            numBlocks_ * (4 + numDims_));
      checkOgGrad_->setData(bias_->getWGrad()->getData() +
                            numBlocks_ * (4 + 2 * numDims_));
    }
  } else {
    LOG(FATAL) << "Bias should be here.";
  }
  for (int i = 0; i < numDims_; i++) {
    directions_.push_back(config_.directions(i));
  }
  for (int i = 0; i < numDims_; i++) {
    delays_.push_back(-1);
  }
  activationGate_.reset(ActivationFunction::create(config_.active_gate_type()));
  activationState_.reset(
      ActivationFunction::create(config_.active_state_type()));

  return true;
}

void MDLstmLayer::forward(PassType passType) {
  Layer::forward(passType);

  const Argument& input = getInput(0);
  CHECK(input.sequenceStartPositions);
  int batchSize = input.getBatchSize();
  int numSequences = input.getNumSequences();
  resetOutput(batchSize, numBlocks_);
  CHECK_EQ(numBlocks_ * (3 + numDims_), input.value->getWidth());
  const int* starts = input.sequenceStartPositions->getData(false);
  CHECK_EQ(starts[numSequences], batchSize);

  int* dimsData = input.cpuSequenceDims->getData();
  CHECK_EQ(int(input.cpuSequenceDims->getSize()), numDims_* numSequences);

  for (int i = 0; i < numSequences; i++) {
    std::vector<int> dims;
    for (int j = 0; j < numDims_; j++) {
      dims.push_back(dimsData[i * numDims_ + j]);
    }
    dimsV_.push_back(dims);
  }

  frameInputGate_.reserve(batchSize);
  frameForgetGate_.reserve(batchSize);
  frameOutputGate_.reserve(batchSize);
  frameInputNode_.reserve(batchSize);
  frameGate_.reserve(batchSize);
  frameState_.reserve(batchSize);
  framePreOutput_.reserve(batchSize);
  frameOutput_.reserve(batchSize);

  Matrix::resizeOrCreate(gate_.value,
                         /* height= */ batchSize,
                         numBlocks_ * (3 + numDims_),
                         /* trans= */ false,
                         useGpu_);

  for (int i = frameGate_.size(); i < batchSize; i++) {
    Argument arg;
    arg.value = Matrix::create(nullptr,
                               /* height= */ 1,
                               numBlocks_ * (3 + numDims_),
                               /* trans= */ false,
                               useGpu_);
    arg.grad = Matrix::create(nullptr,
                              /* height= */ 1,
                              numBlocks_ * (3 + numDims_),
                              /* trans= */ false,
                              useGpu_);
    frameGate_.push_back(arg);
  }
  for (int i = frameInputGate_.size(); i < batchSize; i++) {
    Argument arg;
    arg.value = Matrix::create(nullptr,
                               /* height= */ 1,
                               numBlocks_,
                               /* trans= */ false,
                               useGpu_);
    arg.grad = Matrix::create(nullptr,
                              /* height= */ 1,
                              numBlocks_,
                              /* trans= */ false,
                              useGpu_);
    frameInputGate_.push_back(arg);
  }
  for (int i = frameForgetGate_.size(); i < batchSize; i++) {
    Argument arg;
    arg.value = Matrix::create(nullptr,
                               /* height= */ numDims_,
                               numBlocks_,
                               /* trans= */ false,
                               useGpu_);
    arg.grad = Matrix::create(nullptr,
                              /* height= */ numDims_,
                              numBlocks_,
                              /* trans= */ false,
                              useGpu_);
    frameForgetGate_.push_back(arg);
  }
  for (int i = frameOutputGate_.size(); i < batchSize; i++) {
    Argument arg;
    arg.value = Matrix::create(nullptr,
                               /* height= */ 1,
                               numBlocks_,
                               /* trans= */ false,
                               useGpu_);
    arg.grad = Matrix::create(nullptr,
                              /* height= */ 1,
                              numBlocks_,
                              /* trans= */ false,
                              useGpu_);
    frameOutputGate_.push_back(arg);
  }
  for (int i = frameInputNode_.size(); i < batchSize; i++) {
    Argument arg;
    arg.value = Matrix::create(nullptr,
                               /* height= */ 1,
                               numBlocks_,
                               /* trans= */ false,
                               useGpu_);
    arg.grad = Matrix::create(nullptr,
                              /* height= */ 1,
                              numBlocks_,
                              /* trans= */ false,
                              useGpu_);
    frameInputNode_.push_back(arg);
  }
  for (int i = frameState_.size(); i < batchSize; i++) {
    Argument arg;
    arg.value = Matrix::create(
        /* height= */ 1, numBlocks_, /* trans= */ false, useGpu_);
    frameState_.push_back(arg);
  }
  for (int i = framePreOutput_.size(); i < batchSize; i++) {
    Argument arg;
    arg.value = Matrix::create(
        /* height= */ 1, numBlocks_, /* trans= */ false, useGpu_);
    framePreOutput_.push_back(arg);
  }
  for (int i = frameOutput_.size(); i < batchSize; i++) {
    Argument arg;
    arg.value = Matrix::create(nullptr,
                               /* height= */ 1,
                               numBlocks_,
                               /* trans= */ false,
                               useGpu_);
    arg.grad = Matrix::create(nullptr,
                              /* height= */ 1,
                              numBlocks_,
                              /* trans= */ false,
                              useGpu_);
    frameOutput_.push_back(arg);
  }

  for (int i = 0; i < batchSize; i++) {
    frameOutput_[i].value->setData(output_.value->getData() + i * numBlocks_);
    frameGate_[i].value->setData(gate_.value->getData() +
                                 i * numBlocks_ * (3 + numDims_));
    frameInputNode_[i].value->setData(gate_.value->getData() +
                                      i * numBlocks_ * (3 + numDims_) +
                                      numBlocks_ * 0);
    frameInputGate_[i].value->setData(gate_.value->getData() +
                                      i * numBlocks_ * (3 + numDims_) +
                                      numBlocks_ * 1);
    frameForgetGate_[i].value->setData(gate_.value->getData() +
                                       i * numBlocks_ * (3 + numDims_) +
                                       numBlocks_ * 2);
    frameOutputGate_[i].value->setData(gate_.value->getData() +
                                       i * numBlocks_ * (3 + numDims_) +
                                       numBlocks_ * (2 + numDims_));
  }

  AsyncGpuBlock asyncGpuBlock;
  gate_.value->assign(*input.value);

  if (bias_) {
    gate_.value->addBias(*localBias_, 1);
  }

  for (int i = 0; i < numSequences; i++) {
    CoordIterator coordIter(dimsV_[i], directions_);
    forwardOneSequence(starts[i], coordIter);
  }
}

void MDLstmLayer::forwardGate2OutputSequence(int start,
                                             CoordIterator& coordIter) {
  int idxCurr = start + coordIter.offset();
  std::vector<int> preOffsetV;
  preOffsetV.reserve(numDims_);
  for (int i = 0; i < numDims_; i++) {
    std::vector<int> prePos;
    if (coordIter.getPrePos(delays_, i, prePos)) {
      preOffsetV[i] = coordIter.offset(prePos);
    } else {
      preOffsetV[i] = -1;
    }
  }

  for (int i = 0; i < numDims_; i++) {
    if (preOffsetV[i] >= 0) {
      frameInputGate_[idxCurr].value->addDotMul(
          *frameState_[start + preOffsetV[i]].value, *checkIg_, 1.0, 1.0);

      MatrixPtr fgGateOneDim = Matrix::create(
          frameForgetGate_[idxCurr].value->getData() + i * numBlocks_,
          1,
          numBlocks_,
          false,
          useGpu_);
      MatrixPtr checkFgOneDim =
          Matrix::create(checkFg_->getData() + i * numBlocks_,
                         1.0,
                         numBlocks_,
                         false,
                         useGpu_);
      fgGateOneDim->addDotMul(
          *frameState_[start + preOffsetV[i]].value, *checkFgOneDim, 1.0, 1.0);
    }
  }
  auto status = activationGate_->forward(frameInputGate_[idxCurr]);
  status.check();
  status = activationGate_->forward(frameForgetGate_[idxCurr]);
  status.check();
  status = activation_->forward(frameInputNode_[idxCurr]);
  status.check();

  frameState_[idxCurr].value->zeroMem();
  for (int i = 0; i < numDims_; i++) {
    if (preOffsetV[i] >= 0) {
      MatrixPtr fgGateOneDim = Matrix::create(
          frameForgetGate_[idxCurr].value->getData() + i * numBlocks_,
          1,
          numBlocks_,
          false,
          useGpu_);
      frameState_[idxCurr].value->addDotMul(
          *frameState_[start + preOffsetV[i]].value, *fgGateOneDim, 1.0, 1.0);
    }
  }
  frameState_[idxCurr].value->addDotMul(*frameInputNode_[idxCurr].value,
                                        *frameInputGate_[idxCurr].value,
                                        1.0,
                                        1.0);

  frameOutputGate_[idxCurr].value->addDotMul(
      *frameState_[idxCurr].value, *checkOg_, 1.0, 1.0);
  status = activationGate_->forward(frameOutputGate_[idxCurr]);
  status.check();

  framePreOutput_[idxCurr].value->copyFrom(*(frameState_[idxCurr].value));
  status = activationState_->forward(framePreOutput_[idxCurr]);
  status.check();

  frameOutput_[idxCurr].value->dotMul(*framePreOutput_[idxCurr].value,
                                      *frameOutputGate_[idxCurr].value);
}

void MDLstmLayer::forwardOneSequence(int start, CoordIterator& coordIter) {
  for (coordIter.begin(); !coordIter.end(); ++coordIter) {
    int offset = coordIter.offset();
    for (int i = 0; i < numDims_; i++) {
      std::vector<int> prePos;
      if (coordIter.getPrePos(delays_, i, prePos)) {
        int preOffset = coordIter.offset(prePos);
        frameGate_[start + offset].value->mul(
            *frameOutput_[start + preOffset].value, *weight_->getW(), 1.0, 1.0);
      }
    }
    forwardGate2OutputSequence(start, coordIter);
  }
}

void MDLstmLayer::backward(const UpdateCallback& callback) {
  const Argument& input = getInput(0);
  CHECK(input.sequenceStartPositions);
  int batchSize = input.getBatchSize();
  const int* starts = input.sequenceStartPositions->getData(false);
  size_t numSequences = input.getNumSequences();

  Matrix::resizeOrCreate(gate_.grad,
                         /* height= */ batchSize,
                         numBlocks_ * (3 + numDims_),
                         /* trans= */ false,
                         useGpu_);

  for (int i = 0; i < batchSize; i++) {
    if (frameState_[i].grad == NULL)
      frameState_[i].grad = Matrix::create(
          /* height= */ 1, numBlocks_, /* trans= */ false, useGpu_);
  }
  for (int i = 0; i < batchSize; i++) {
    if (framePreOutput_[i].grad == NULL)
      framePreOutput_[i].grad = Matrix::create(
          /* height= */ 1, numBlocks_, /* trans= */ false, useGpu_);
  }

  for (int i = 0; i < batchSize; i++) {
    frameOutput_[i].grad->setData(output_.grad->getData() + i * numBlocks_);
    frameGate_[i].grad->setData(gate_.grad->getData() +
                                i * numBlocks_ * (3 + numDims_));
    frameInputNode_[i].grad->setData(gate_.grad->getData() +
                                     i * numBlocks_ * (3 + numDims_) +
                                     numBlocks_ * 0);
    frameInputGate_[i].grad->setData(gate_.grad->getData() +
                                     i * numBlocks_ * (3 + numDims_) +
                                     numBlocks_ * 1);
    frameForgetGate_[i].grad->setData(gate_.grad->getData() +
                                      i * numBlocks_ * (3 + numDims_) +
                                      numBlocks_ * 2);
    frameOutputGate_[i].grad->setData(gate_.grad->getData() +
                                      i * numBlocks_ * (3 + numDims_) +
                                      numBlocks_ * (2 + numDims_));
  }

  {
    AsyncGpuBlock asyncGpuBlock;

    for (size_t i = 0; i < numSequences; i++) {
      CoordIterator coordIter(dimsV_[i], directions_);
      backwardOneSequence(starts[i], coordIter);
    }
  }

  if (input.grad) {
    input.grad->add(*gate_.grad);
  }
  if (bias_ && bias_->getWGrad()) {
    localBiasGrad_->collectBias(*gate_.grad, 1);
    bias_->getParameterPtr()->incUpdate(callback);
  }

  weight_->getParameterPtr()->incUpdate(callback);
}

void MDLstmLayer::backwardGate2OutputSequence(int start,
                                              CoordIterator& coordIter) {
  int idxCurr = start + coordIter.offset();
  std::vector<int> preOffsetV;
  std::vector<int> nextOffsetV;
  preOffsetV.reserve(numDims_);
  nextOffsetV.reserve(numDims_);
  for (int i = 0; i < numDims_; i++) {
    std::vector<int> prePos;
    if (coordIter.getPrePos(delays_, i, prePos)) {
      preOffsetV[i] = coordIter.offset(prePos);
    } else {
      preOffsetV[i] = -1;
    }
    std::vector<int> nextPos;
    if (coordIter.getNextPos(delays_, i, nextPos)) {
      nextOffsetV[i] = coordIter.offset(nextPos);
    } else {
      nextOffsetV[i] = -1;
    }
  }

  framePreOutput_[idxCurr].grad->dotMul(*frameOutput_[idxCurr].grad,
                                        *frameOutputGate_[idxCurr].value);
  activationState_->backward(framePreOutput_[idxCurr]).check();
  frameState_[idxCurr].grad->copyFrom(*(framePreOutput_[idxCurr].grad));

  frameOutputGate_[idxCurr].grad->dotMul(*frameOutput_[idxCurr].grad,
                                         *framePreOutput_[idxCurr].value);
  activationGate_->backward(frameOutputGate_[idxCurr]).check();

  frameState_[idxCurr].grad->addDotMul(
      *frameOutputGate_[idxCurr].grad, *checkOg_, 1.0, 1.0);
  for (int i = 0; i < numDims_; i++) {
    if (nextOffsetV[i] >= 0) {
      frameState_[idxCurr].grad->addDotMul(
          *frameInputGate_[start + nextOffsetV[i]].grad, *checkIg_, 1.0, 1.0);

      MatrixPtr fgGateOneDimGrad = Matrix::create(
          frameForgetGate_[start + nextOffsetV[i]].grad->getData() +
              i * numBlocks_,
          1,
          numBlocks_,
          false,
          useGpu_);
      MatrixPtr fgGateOneDimVal = Matrix::create(
          frameForgetGate_[start + nextOffsetV[i]].value->getData() +
              i * numBlocks_,
          1,
          numBlocks_,
          false,
          useGpu_);
      MatrixPtr checkFgOneDim = Matrix::create(
          checkFg_->getData() + i * numBlocks_, 1, numBlocks_, false, useGpu_);

      frameState_[idxCurr].grad->addDotMul(
          *fgGateOneDimGrad, *checkFgOneDim, 1.0, 1.0);
      frameState_[idxCurr].grad->addDotMul(
          *frameState_[start + nextOffsetV[i]].grad,
          *fgGateOneDimVal,
          1.0,
          1.0);
    }
  }

  frameInputNode_[idxCurr].grad->dotMul(*frameState_[idxCurr].grad,
                                        *frameInputGate_[idxCurr].value);
  frameInputGate_[idxCurr].grad->dotMul(*frameState_[idxCurr].grad,
                                        *frameInputNode_[idxCurr].value);

  frameForgetGate_[idxCurr].grad->zeroMem();
  for (int i = 0; i < numDims_; i++) {
    if (preOffsetV[i] >= 0) {
      MatrixPtr fgGateOneDimGrad = Matrix::create(
          frameForgetGate_[idxCurr].grad->getData() + i * numBlocks_,
          1,
          numBlocks_,
          false,
          useGpu_);
      fgGateOneDimGrad->addDotMul(*frameState_[idxCurr].grad,
                                  *frameState_[start + preOffsetV[i]].value,
                                  1.0,
                                  1.0);
    }
  }

  activationGate_->backward(frameInputGate_[idxCurr]).check();
  activationGate_->backward(frameForgetGate_[idxCurr]).check();
  activation_->backward(frameInputNode_[idxCurr]).check();

  if (bias_->getWGrad()) {
    for (int i = 0; i < numDims_; i++) {
      if (preOffsetV[i] >= 0) {
        checkIgGrad_->addDotMul(*frameInputGate_[idxCurr].grad,
                                *frameState_[start + preOffsetV[i]].value,
                                1.0,
                                1.0);

        MatrixPtr fgGateOneDimGrad = Matrix::create(
            frameForgetGate_[idxCurr].grad->getData() + i * numBlocks_,
            1,
            numBlocks_,
            false,
            useGpu_);
        MatrixPtr checkFgOneDimGrad =
            Matrix::create(checkFgGrad_->getData() + i * numBlocks_,
                           1,
                           numBlocks_,
                           false,
                           useGpu_);
        checkFgOneDimGrad->addDotMul(*fgGateOneDimGrad,
                                     *frameState_[start + preOffsetV[i]].value,
                                     1.0,
                                     1.0);
      }
    }
    checkOgGrad_->addDotMul(
        *frameOutputGate_[idxCurr].grad, *frameState_[idxCurr].value, 1.0, 1.0);
  }
}

void MDLstmLayer::backwardOneSequence(int start, CoordIterator& coordIter) {
  MatrixPtr weightT = weight_->getW()->getTranspose();
  for (coordIter.rbegin(); !coordIter.end(); --coordIter) {
    int offset = coordIter.offset();
    backwardGate2OutputSequence(start, coordIter);
    for (int i = 0; i < numDims_; i++) {
      std::vector<int> prePos;
      if (coordIter.getPrePos(delays_, i, prePos)) {
        int preOffset = coordIter.offset(prePos);
        frameOutput_[start + preOffset].grad->mul(
            *frameGate_[start + offset].grad, *weightT, 1.0, 1.0);
        if (weight_->getWGrad()) {
          weight_->getWGrad()->mul(
              *frameOutput_[start + preOffset].value->getTranspose(),
              *frameGate_[start + offset].grad,
              1.0,
              1.0);
        }
      }
    }
  }
}

}  // namespace paddle
