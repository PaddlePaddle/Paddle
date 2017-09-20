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
  // should not merge the resetBwd into resetFwd,
  // because the grad data would be changing before backward.
  bool needResetBwd_;
  // mkldnn matrix, primitive, stream and pipeline
  MKLDNNMatrixPtr val_;
  MKLDNNMatrixPtr grad_;
  std::shared_ptr<MKLDNNStream> stream_;
  std::shared_ptr<mkldnn::primitive> fwd_;
  std::shared_ptr<mkldnn::primitive> bwd_;
  std::vector<mkldnn::primitive> pipelineFwd_;
  std::vector<mkldnn::primitive> pipelineBwd_;

public:
  MKLDNNActivation() : cnt_(0), needResetBwd_(true) {}
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

protected:
  // save the forward primitive desc, which can be used backward
  std::shared_ptr<eltwise_fwd::primitive_desc> fwdPD_;
  // eltwise_bwd need src input value
  MKLDNNMatrixPtr inVal_;
  // use for copy data
  std::shared_ptr<mkldnn::reorder> copyInVal_;

public:
  MKLDNNEltwiseActivation() {}

  ~MKLDNNEltwiseActivation() {}

  virtual const std::string& getName() const = 0;

  // in common, the alpha of forward and backward should be equal.
  // but for relu, to avoid negative value, they should be opposite
  virtual float getAlpha() const = 0;
  virtual float getBwdAlpha() const = 0;
  virtual float getBeta() const { return 0.f; }
  virtual mkldnn::algorithm getAlgo(const std::string& type) const {
    if (type == "mkldnn_relu") {
      return mkldnn::algorithm::eltwise_relu;
    } else if (type == "mkldnn_tanh") {
      return mkldnn::algorithm::eltwise_tanh;
    } else if (type == "mkldnn_elu") {
      return mkldnn::algorithm::eltwise_elu;
    } else {
      LOG(FATAL) << "Unkown eltwise activation type: " << type;
    }
    return (mkldnn::algorithm)0;
  }

  /**
   * reshape and reset the forward primitives
   */
  void resetFwd(Argument& act) {
    if (cnt_ == act.value->getElementCnt()) {
      return;
    }
    cnt_ = act.value->getElementCnt();
    stream_.reset(new MKLDNNStream());
    auto eng = CPUEngine::Instance().getEngine();

    // get algo setting
    mkldnn::algorithm algo = getAlgo(this->getName());
    // note: alpha represents the NegativeSlope when used in relu.
    float alpha = getAlpha();
    float beta = getBeta();

    /// forward
    pipelineFwd_.clear();
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
    fwdPD_.reset(new eltwise_fwd::primitive_desc(fwdDesc, eng));
    // use inplace for forward but save input value before submit
    inVal_ = val_;
    copyInVal_ = nullptr;
    if (act.grad && algo == mkldnn::algorithm::eltwise_tanh) {
      // tanh need save src input for backward
      inVal_ = MKLDNNMatrix::create(nullptr, val_->getPrimitiveDesc());
      copyInVal_ = std::make_shared<mkldnn::reorder>(*val_, *inVal_);
      CHECK(copyInVal_) << "should not be emptry";
      pipelineFwd_.push_back(*copyInVal_);
    }
    fwd_.reset(new eltwise_fwd(*fwdPD_, *val_, *val_));
    pipelineFwd_.push_back(*fwd_);
    needResetBwd_ = true;
  }

  /**
   * reset the backward primitives, can not merge into resetFwd as the grad data
   * would be changing before backward.
   */
  void resetBwd(Argument& act) {
    if (!needResetBwd_) {
      return;
    }
    needResetBwd_ = false;
    mkldnn::algorithm algo = getAlgo(this->getName());
    float alpha = getBwdAlpha();
    float beta = getBeta();
    grad_ = MKLDNNMatrix::create(act.grad, val_->getPrimitiveDesc());
    auto eng = CPUEngine::Instance().getEngine();
    auto bwdDesc = eltwise_bwd::desc(
        algo, grad_->getMemoryDesc(), val_->getMemoryDesc(), alpha, beta);
    auto bwdPD = eltwise_bwd::primitive_desc(bwdDesc, eng, *fwdPD_);
    CHECK(inVal_);
    bwd_.reset(new eltwise_bwd(bwdPD, *inVal_, *grad_, *grad_));
    pipelineBwd_.clear();
    pipelineBwd_.push_back(*bwd_);
  }

  Error __must_check forward(Argument& act) {
    resetFwd(act);
    stream_->submit(pipelineFwd_);
    return Error();
  }

  Error __must_check backward(Argument& act) {
    resetBwd(act);
    stream_->submit(pipelineBwd_);
    return Error();
  }
};

}  // namespace paddle
