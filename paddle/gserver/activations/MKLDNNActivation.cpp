/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "MKLDNNActivation.h"
#include "mkldnn.hpp"
#include "paddle/utils/ClassRegistrar.h"

namespace paddle {

static ClassRegistrar<ActivationFunction> gMKLDNNActivationRegistrar;
/**
 * @def MKLDNN_ACTIVATION_CLASS_NAME
 * @note MKLDNN_ACTIVATION_CLASS_NAME(relu) relu_;
 * means mkldnn_reluActivation relu_;
 */
#define MKLDNN_ACTIVATION_CLASS_NAME(ACT_TYPE) mkldnn_##ACT_TYPE##Activation

/**
 * @def BEGIN_MKLDNN_ACTIVATION
 */
#define BEGIN_MKLDNN_ACTIVATION(ACT_TYPE, BASE_CLASS) \
  class MKLDNN_ACTIVATION_CLASS_NAME(ACT_TYPE) : public BASE_CLASS {
/**
 * @def END_MKLDNN_ACTIVATION
 */
#define END_MKLDNN_ACTIVATION(ACT_TYPE)                            \
 private:                                                          \
  static const std::string name;                                   \
                                                                   \
 public:                                                           \
  const std::string& getName() const { return name; }              \
  }                                                                \
  ;                                                                \
  const std::string MKLDNN_ACTIVATION_CLASS_NAME(ACT_TYPE)::name = \
      "mkldnn_" #ACT_TYPE;                                         \
  static InitFunction __reg_activation__mkldnn_##ACT_TYPE([] {     \
    gMKLDNNActivationRegistrar                                     \
        .registerClass<MKLDNN_ACTIVATION_CLASS_NAME(ACT_TYPE)>(    \
            "mkldnn_" #ACT_TYPE);                                  \
  });

/**
 * @def DEFINE_MKLDNN_ACTIVATION
 */
#define DEFINE_MKLDNN_ACTIVATION(ACT_TYPE, BASE_CLASS) \
  BEGIN_MKLDNN_ACTIVATION(ACT_TYPE, BASE_CLASS)        \
  END_MKLDNN_ACTIVATION(ACT_TYPE)

/**
 * @def DEFINE_MKLDNN_ELTWISE_ACTIVATION
 */
#define DEFINE_MKLDNN_ELTWISE_ACTIVATION(                            \
    ACT_TYPE, BASE_CLASS, ALPHA, BWD_ALPHA)                          \
  BEGIN_MKLDNN_ACTIVATION(ACT_TYPE, BASE_CLASS)                      \
 private:                                                            \
  static const float alpha;                                          \
  static const float bwdAlpha;                                       \
                                                                     \
 public:                                                             \
  float getAlpha() const { return alpha; }                           \
  float getBwdAlpha() const { return bwdAlpha; }                     \
  END_MKLDNN_ACTIVATION(ACT_TYPE)                                    \
  const float MKLDNN_ACTIVATION_CLASS_NAME(ACT_TYPE)::alpha = ALPHA; \
  const float MKLDNN_ACTIVATION_CLASS_NAME(ACT_TYPE)::bwdAlpha = BWD_ALPHA;

/**
 * @brief MKLDNN Relu Activation.
 * Actually mkldnn_relu is Leaky Relu.
 *  f(x) = x                   (x >= 0)
 *  f(x) = negative_slope * x  (x <  0)
 * @note the negative_slope should be -0.f in forward
 */
DEFINE_MKLDNN_ELTWISE_ACTIVATION(relu, MKLDNNEltwiseActivation, -0.f, 0.f)

/**
 * @brief MKLDNN Tanh Activation.
 */
DEFINE_MKLDNN_ELTWISE_ACTIVATION(tanh, MKLDNNEltwiseActivation, 0.f, 0.f)

/**
 * @brief MKLDNN ELU(Exponential Linear Unit) Activation.
 *  f(x) = x                              (x >= 0)
 *  f(x) = negative_slope * (exp(x) - 1)  (x <  0)
 */
DEFINE_MKLDNN_ELTWISE_ACTIVATION(elu, MKLDNNEltwiseActivation, 0.f, 0.f)

mkldnn::algorithm MKLDNNEltwiseActivation::getAlgo(std::string type) const {
  const std::map<std::string, mkldnn::algorithm> algoMap = {
      {"relu", algorithm::eltwise_relu},
      {"tanh", algorithm::eltwise_tanh},
      {"elu", algorithm::eltwise_elu}};
  type.erase(0, 7);  // remove mkldnn_
  algorithm algo = (algorithm)0;
  mapGet(type, algoMap, &algo);
  return algo;
}

void MKLDNNEltwiseActivation::resetFwd(Argument& act) {
  if (cnt_ == act.value->getElementCnt()) {
    return;
  }
  MKLDNNActivation::resetFwd(act);
  // note: alpha represents the NegativeSlope when used in relu.
  float alpha = getAlpha();
  float beta = getBeta();
  algorithm algo = getAlgo(this->getName());
  auto fwdDesc = eltwise_fwd::desc(mkldnn::prop_kind::forward_training,
                                   algo,
                                   val_->getMemoryDesc(),
                                   alpha,
                                   beta);
  fwdPD_.reset(new eltwise_fwd::primitive_desc(fwdDesc, *engine_));
  // use inplace for forward but save input value before submit
  inVal_ = val_;
  copyInVal_ = nullptr;
  if (act.grad && algo == algorithm::eltwise_tanh) {
    // tanh need save src input for backward
    inVal_ = MKLDNNMatrix::create(val_->getPrimitiveDesc());
    copyInVal_ = std::make_shared<mkldnn::reorder>(*val_, *inVal_);
    CHECK(copyInVal_) << "should not be emptry";
    pipelineFwd_.push_back(*copyInVal_);
  }
  fwd_.reset(new eltwise_fwd(*fwdPD_, *val_, *val_));
  pipelineFwd_.push_back(*fwd_);
  needResetBwd_ = true;
}

void MKLDNNEltwiseActivation::resetBwd(Argument& act) {
  if (!needResetBwd_) {
    return;
  }
  VLOG(MKLDNN_BASE) << getName() << " reset mkldnn backward";
  needResetBwd_ = false;
  algorithm algo = getAlgo(this->getName());
  float alpha = getBwdAlpha();
  float beta = getBeta();
  grad_ = MKLDNNMatrix::create(val_->getPrimitiveDesc(), act.grad);
  auto eng = CPUEngine::Instance().getEngine();
  auto bwdDesc = eltwise_bwd::desc(
      algo, grad_->getMemoryDesc(), val_->getMemoryDesc(), alpha, beta);
  auto bwdPD = eltwise_bwd::primitive_desc(bwdDesc, eng, *fwdPD_);
  CHECK(inVal_);
  bwd_.reset(new eltwise_bwd(bwdPD, *inVal_, *grad_, *grad_));
  pipelineBwd_.clear();
  pipelineBwd_.push_back(*bwd_);
}

/**
 * @brief MKLDNN Softmax Activation
 */
DEFINE_MKLDNN_ACTIVATION(softmax, MKLDNNSoftmaxActivation)

void MKLDNNSoftmaxActivation::resetFwd(Argument& act) {
  if (cnt_ == act.value->getElementCnt()) {
    return;
  }
  MKLDNNActivation::resetFwd(act);
  int axis = 1;
  auto fwdDesc = softmax_fwd::desc(
      mkldnn::prop_kind::forward_scoring, val_->getMemoryDesc(), axis);
  auto fwdPD = softmax_fwd::primitive_desc(fwdDesc, *engine_);
  fwd_.reset(new softmax_fwd(fwdPD, *val_, *val_));
  pipelineFwd_.push_back(*fwd_);
}

Error __must_check MKLDNNSoftmaxActivation::forward(Argument& act) {
  resetFwd(act);
  stream_->submit(pipelineFwd_);
  real* v = act.value->getData();
  real threshold = exp(-64);
#pragma omp parallel for
  for (size_t i = 0; i < act.value->getElementCnt(); ++i) {
    v[i] = v[i] < threshold ? threshold : v[i];
  }
  return Error();
}

Error __must_check MKLDNNSoftmaxActivation::backward(Argument& act) {
  MatrixPtr outputV = act.value;
  MatrixPtr outputG = act.grad;
  Matrix::resizeOrCreate(sftMaxDot_,
                         outputG->getHeight(),
                         outputG->getWidth(),
                         /* trans */ false,
                         /* useGpu */ false);
  Matrix::resizeOrCreate(sftMaxSum_,
                         outputG->getHeight(),
                         1,
                         /* trans */ false,
                         /* useGpu */ false);
  sftMaxDot_->dotMul(*outputG, *outputV);
  sftMaxSum_->colMerge(*sftMaxDot_);
  act.grad->softmaxDerivative(*act.value, *sftMaxSum_);
  return Error();
}

ActivationFunction* MKLDNNActivation::create(const std::string& type) {
  return gMKLDNNActivationRegistrar.createByType(type);
}

std::vector<std::string> MKLDNNActivation::getAllRegisteredTypes() {
  std::vector<std::string> types;
  gMKLDNNActivationRegistrar.forEachType(
      [&](const std::string& type) { types.push_back(type); });
  return types;
}

void MKLDNNActivation::resetFwd(Argument& act) {
  VLOG(MKLDNN_BASE) << getName() << " reset mkldnn forward";
  cnt_ = act.value->getElementCnt();
  pipelineFwd_.clear();
  stream_.reset(new MKLDNNStream());
  engine_.reset(new mkldnn::engine(mkldnn::engine::cpu, 0));
  val_ = std::dynamic_pointer_cast<MKLDNNMatrix>(act.value);
  if (val_ == nullptr) {
    int bs = act.getBatchSize();
    int ih = act.getFrameHeight() > 0 ? act.getFrameHeight() : 1;
    int iw = act.getFrameWidth() > 0 ? act.getFrameWidth() : 1;
    int ic = cnt_ / bs / ih / iw;
    CHECK_EQ(cnt_, (size_t)bs * ic * ih * iw);
    val_ = MKLDNNMatrix::create(
        {bs, ic, ih, iw}, mkldnn::memory::format::nchw, *engine_, act.value);
    CHECK(val_);
    val_->downSpatial();
  }
}

Error __must_check MKLDNNActivation::forward(Argument& act) {
  resetFwd(act);
  stream_->submit(pipelineFwd_);
  return Error();
}
Error __must_check MKLDNNActivation::backward(Argument& act) {
  resetBwd(act);
  stream_->submit(pipelineBwd_);
  return Error();
}
}  // namespace paddle
