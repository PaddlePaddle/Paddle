/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "ActivationFunction.h"
#include "mkldnn.hpp"
#include "paddle/gserver/layers/MKLDNNBase.h"
#include "paddle/math/MKLDNNMatrix.h"
#include "paddle/parameter/Argument.h"

namespace paddle {

/**
 * @brief Base class of MKLDNN Activation.
 * Common activation function are provieded,
 * including mkldnn_relu, mkldnn_elu, mkldnn_tanh, mkldnn_softmax
 */
class MKLDNNActivation : public ActivationFunction {
protected:
  // input value element count
  size_t cnt_;
  // mkldnn matrix, primitive, stream and pipeline
  MKLDNNMatrixPtr val_;
  MKLDNNMatrixPtr grad_;
  std::shared_ptr<MKLDNNStream> stream_;
  std::shared_ptr<mkldnn::primitive> fwd_;
  std::shared_ptr<mkldnn::primitive> bwd_;
  std::vector<mkldnn::primitive> pipelineFwd_;
  std::vector<mkldnn::primitive> pipelineBwd_;

public:
  MKLDNNActivation() : cnt_(0) {}
  ~MKLDNNActivation() {}
  static ActivationFunction* create(const std::string& type);
  static std::vector<std::string> getAllRegisteredTypes();
  virtual const std::string& getName() const = 0;
  virtual Error __must_check forward(Argument& act) = 0;
  virtual Error __must_check backward(Argument& act) = 0;
};

/**
 * @brief Base class of MKLDNN Eltwise Activation,
 * includes mkldnn_relu, mkldnn_elu and mkldnn_tanh.
 */
class MKLDNNEltwiseActivation : public MKLDNNActivation {
  typedef mkldnn::eltwise_forward eltwise_fwd;
  typedef mkldnn::eltwise_backward eltwise_bwd;

public:
  MKLDNNEltwiseActivation() {}

  ~MKLDNNEltwiseActivation() {}

  virtual const std::string& getName() const = 0;
  virtual float getAlpha() const = 0;
  virtual float getBeta() const { return 0.f; }

  /**
   * reshape and reset the forward and backward primitives
   */
  void resetPrimitives(Argument& act) {
    if (cnt_ == act.value->getElementCnt()) {
      return;
    }
    cnt_ = act.value->getElementCnt();
    stream_.reset(new MKLDNNStream());
    auto eng = CPUEngine::Instance().getEngine();

    // get algo setting
    mkldnn::algorithm algo;
    if (this->getName() == "mkldnn_relu") {
      algo = mkldnn::algorithm::eltwise_relu;
    } else if (this->getName() == "mkldnn_tanh") {
      algo = mkldnn::algorithm::eltwise_tanh;
    } else if (this->getName() == "mkldnn_elu") {
      algo = mkldnn::algorithm::eltwise_elu;
    } else {
      LOG(FATAL) << "Unkown eltwise activation type: " << this->getName();
    }
    // note: alpha represents the NegativeSlope when used in relu.
    float alpha = getAlpha();
    float beta = getBeta();

    /// forward
    val_ = std::dynamic_pointer_cast<MKLDNNMatrix>(act.value);
    if (val_ == nullptr) {
      int bs = act.getBatchSize();
      int ih = act.getFrameHeight() > 0 ? act.getFrameHeight() : 1;
      int iw = act.getFrameWidth() > 0 ? act.getFrameWidth() : 1;
      int ic = cnt_ / bs / ih / iw;
      CHECK_EQ(cnt_, (size_t)bs * ic * ih * iw);
      val_ = MKLDNNMatrix::create(
          act.value, {bs, ic, ih, iw}, mkldnn::memory::format::nchw, eng);
      CHECK(val_);
    }
    auto fwdDesc = eltwise_fwd::desc(mkldnn::prop_kind::forward_training,
                                     algo,
                                     val_->getMemoryDesc(),
                                     alpha,
                                     beta);
    auto fwdPD = eltwise_fwd::primitive_desc(fwdDesc, eng);
    // inplace buffer, dst = src
    fwd_.reset(new eltwise_fwd(fwdPD, *val_, *val_));
    pipelineFwd_.clear();
    pipelineFwd_.push_back(*fwd_);

    /// backward
    if (act.grad == nullptr) {
      grad_ = nullptr;
      return;
    }
    grad_ = MKLDNNMatrix::create(act.grad, val_->getPrimitiveDesc());
    auto bwdDesc = eltwise_bwd::desc(
        algo, grad_->getMemoryDesc(), val_->getMemoryDesc(), alpha, beta);
    auto bwdPD = eltwise_bwd::primitive_desc(bwdDesc, eng, fwdPD);
    bwd_.reset(new eltwise_bwd(bwdPD, *val_, *grad_, *grad_));
    pipelineBwd_.clear();
    pipelineBwd_.push_back(*bwd_);
  }

  Error __must_check forward(Argument& act) {
    resetPrimitives(act);
    stream_->submit(pipelineFwd_);
    return Error();
  }

  Error __must_check backward(Argument& act) {
    stream_->submit(pipelineBwd_);
    return Error();
  }
};

}  // namespace paddle
