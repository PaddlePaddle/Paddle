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
  // init forward_ and backward_ functions
  init();
}

bool ContextProjection::init() {
  size_t context_length = config_.context_length();
  int context_start = config_.context_start();
  bool is_padding = config_.trainable_padding();
  size_t total_pad = is_padding ? beginPad_ + endPad_ : 0;

  createFunction(forward_,
                 "ContextProjectionForward",
                 FuncConfig()
                     .set("context_length", context_length)
                     .set("context_start", context_start)
                     .set("begin_pad", beginPad_));
  createFunction(backward_,
                 "ContextProjectionBackward",
                 FuncConfig()
                     .set("context_length", context_length)
                     .set("context_start", context_start)
                     .set("begin_pad", beginPad_)
                     .set("is_padding", is_padding)
                     .set("total_pad", total_pad));

  return true;
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
  CHECK(in_->value && out_->value);
  CHECK(in_->sequenceStartPositions);

  size_t input_dim = in_->value->getWidth();
  size_t dim = out_->value->getWidth();
  CHECK_EQ(dim, input_dim * config_.context_length());
  // size_t batch_size = in_->value->getHeight();
  CHECK_EQ(forward_.size(), (size_t)1) << "Only one forward function here";

  REGISTER_TIMER_INFO("ContextProjectionForward", getName().c_str());
  bool is_padding = config_.trainable_padding();
  /// first use state_, otherwise use weight_(padding false === w nullptr)
  auto w_ptr =
      state_ ? state_.get() : is_padding ? weight_->getW().get() : nullptr;
  const auto start_pos = in_->sequenceStartPositions->getVector(useGpu_);
  BufferArgs inputs;
  BufferArgs outputs;
  inputs.addArg(*in_->value, *start_pos);
  if (w_ptr) {
    inputs.addArg(CpuMatrix(w_ptr->getData(), w_ptr->getHeight(), input_dim),
                  *start_pos);
  }
  outputs.addArg(*out_->value, *start_pos, ADD_TO);
  forward_[0]->calc(inputs, outputs);

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
  CHECK(in_->value && out_->value && out_->grad);
  size_t input_dim = in_->value->getWidth();
  size_t dim = out_->value->getWidth();
  CHECK_EQ(dim, input_dim * config_.context_length());
  size_t batch_size = in_->value->getHeight();
  CHECK_EQ(batch_size, out_->value->getHeight());
  CHECK_EQ(static_cast<int>(backward_.size()), 1)
      << "Only one backward function here";

  REGISTER_TIMER_INFO("ContextProjectionBackward", getName().c_str());
  bool is_padding = config_.trainable_padding();
  auto start_pos = in_->sequenceStartPositions;
  auto w_ptr = is_padding ? weight_->getWGrad() : nullptr;

  BufferArgs inputs;
  BufferArgs outputs;
  inputs.addArg(*out_->grad, *in_->sequenceStartPositions->getVector(useGpu_));
  outputs.addArg(
      CpuMatrix(
          in_->grad ? in_->grad->getData() : nullptr, batch_size, input_dim),
      *in_->sequenceStartPositions->getVector(useGpu_),
      ADD_TO);
  outputs.addArg(CpuMatrix(w_ptr ? w_ptr->getData() : nullptr,
                           w_ptr ? w_ptr->getHeight() : 0,
                           input_dim),
                 ADD_TO);
  backward_[0]->calc(inputs, outputs);

  if (config_.trainable_padding()) {
    weight_->getParameterPtr()->incUpdate(callback);
  }
}

}  // namespace paddle
