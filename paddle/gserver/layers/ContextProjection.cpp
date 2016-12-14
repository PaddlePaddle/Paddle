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

#include "ContextProjection.h"
#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_PROJECTION(context, ContextProjection);

ContextProjection::ContextProjection(const ProjectionConfig& config,
                                     ParameterPtr parameter,
                                     bool useGpu)
    : Projection(config, parameter, useGpu) {
  CHECK(config.has_context_start());
  CHECK(config.has_context_length());
  if (config.context_start() == 0 && config.context_length() == 1) {
    config_.set_trainable_padding(false);
  }
  if (config_.trainable_padding()) {
    CHECK(parameter);
    beginPad_ = std::max(0, -config.context_start());
    endPad_ = std::max(0, config.context_start() + config.context_length() - 1);
    size_t totalPad = beginPad_ + endPad_;
    size_t inputDim = parameter->getSize() / totalPad;
    CHECK_EQ(config.input_size(), inputDim);
    CHECK_EQ(inputDim * totalPad, parameter->getSize());
    weight_.reset(new Weight(totalPad, inputDim, parameter));
  }
}

void ContextProjection::resetState() {
  CHECK_LE(config_.context_start() + config_.context_length(), 1)
      << "state is not allowed for future context";
  if (config_.context_start() >= 0) return;
  Matrix::resizeOrCreate(state_,
                         -config_.context_start(),
                         config_.input_size(),
                         false,  // trans
                         useGpu_);
  Matrix::resizeOrCreate(state2_,
                         -config_.context_start(),
                         config_.input_size(),
                         false,  // trans
                         useGpu_);
  if (config_.trainable_padding()) {
    state_->assign(*weight_->getW()->subMatrix(0, -config_.context_start()));
  } else {
    state_->zeroMem();
  }
}

void ContextProjection::setState(LayerStatePtr state) {
  CHECK(state->value.size() == 1)
      << "one matrix is expected for ContextProjection state";
  state_->copyFrom(*(state->value[0]));
}

LayerStatePtr ContextProjection::getState() {
  if (state_ == nullptr) {
    return nullptr;
  }
  LayerStatePtr res = std::make_shared<LayerState>();
  res->value.push_back(state_->clone(0, 0, false));
  res->value[0]->copyFrom(*state_);
  return res;
}

void ContextProjection::forward() {
  CHECK(in_->value);
  CHECK(in_->sequenceStartPositions);

  auto startPositions = in_->sequenceStartPositions->getVector(useGpu_);

  int64_t inputDim = in_->value->getWidth();
  int64_t dim = out_->value->getWidth();
  CHECK_EQ(dim, inputDim * config_.context_length());

  REGISTER_TIMER_INFO("ContextProjectionForward", getName().c_str());
  bool isPadding = config_.trainable_padding();
  out_->value->contextProjectionForward(
      in_->value,
      state_ ? state_ : isPadding ? weight_->getW() : nullptr,
      *startPositions,
      config_.context_length(),
      config_.context_start(),
      beginPad_,
      state_ ? true : isPadding);

  if (state_ && config_.context_start() < 0) {
    CHECK_EQ(1, in_->getNumSequences());
    const int* starts = in_->sequenceStartPositions->getData(false);
    int length = starts[1] - starts[0];
    if (-config_.context_start() <= length) {
      MatrixPtr sub = in_->value->subMatrix(starts[1] + config_.context_start(),
                                            -config_.context_start());
      state_->copyFrom(*sub);
    } else {
      int prevLength = -config_.context_start() - length;
      state2_->subMatrix(0, prevLength)
          ->copyFrom(*state_->subMatrix(length, prevLength));
      state2_->subMatrix(prevLength, length)
          ->copyFrom(*in_->value->subMatrix(starts[0], length));
      std::swap(state_, state2_);
    }
  }
}

void ContextProjection::backward(const UpdateCallback& callback) {
  CHECK(in_->value);
  int64_t inputDim = in_->value->getWidth();
  int64_t dim = out_->value->getWidth();
  CHECK_EQ(dim, inputDim * config_.context_length());
  auto startPositions = in_->sequenceStartPositions->getVector(useGpu_);

  REGISTER_TIMER_INFO("ContextProjectionBackward", getName().c_str());
  bool isPadding = config_.trainable_padding();
  if (!out_->grad->useGpu()) {
    out_->grad->contextProjectionBackward(
        in_->grad,
        isPadding ? weight_->getWGrad() : nullptr,
        *startPositions,
        config_.context_length(),
        config_.context_start(),
        beginPad_,
        isPadding);
  } else {
    if (in_->grad) {
      out_->grad->contextProjectionBackwardData(in_->grad,
                                                *startPositions,
                                                config_.context_length(),
                                                config_.context_start());
    }

    if (isPadding && weight_->getWGrad()) {
      out_->grad->contextProjectionBackwardWeight(
          weight_->getWGrad(),
          *startPositions,
          config_.context_length(),
          config_.context_start(),
          weight_->getWGrad()->getHeight(),
          beginPad_);
    }
  }

  if (config_.trainable_padding()) {
    weight_->getParameterPtr()->incUpdate(callback);
  }
}

}  // namespace paddle
