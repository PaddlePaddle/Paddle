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

#include "Layer.h"
#include "paddle/math/Matrix.h"
#include <vector>
#include "mkldnn.hpp"
#include "MkldnnBase.h"
#include "MkldnnMemory.h"

#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

namespace paddle {

class MkldnnLayer;
typedef std::shared_ptr<MkldnnLayer> MkldnnLayerPtr;

/**
 * @brief Base class of Mkldnnlayer.
 *
 */
class MkldnnLayer : public Layer {
public:
  /// bottom data and diff buffers
  MkldnnBufferPtr botData_, botDiff_;
  /// top data and diff buffers
  MkldnnBufferPtr topData_, topDiff_;

  /// dims and format for user buffer
  mkldnn::memory::dims botDims_, wgtDims_, biasDims_, topDims_;
  mkldnn::memory::format botFmt_, wgtFmt_, biasFmt_, topFmt_;
  mkldnn::engine engine_;

  // input element cnt
  size_t inputElmCnt_;
  // the input matrix height and width
  size_t iMatH_, iMatW_;

  // the output matrix height and width
  // height: mklSeqLen * bs
  // width : layer size == oc * oh * ow
  size_t oMatH_, oMatW_;

  // MKLDNN aligned seqLen
  // The lengths of sequences in one Batch should be equal
  int seqLen_;
  // batchsize
  int bs_;
  // input image channel, height and width
  int ic_, ih_, iw_;
  // output image channel, height and width
  int oc_, oh_, ow_;

  bool needResetBwd_;

  /// for support mixed with cpu layers
  // flags whether to set memory format of top data or bots diff
  // only one top data but may have several bot diff
  bool nextIsDnn_;
  std::vector<bool> prevIsDnn_;

  /// support with original cpu weights format for only scoring phase
  bool scoreWithPaddleWgt_;

  /// some functions need prepare only once, but not suitable at init
  bool preparedOnce_;

public:
  explicit MkldnnLayer(const LayerConfig& config)
    : Layer(config),
      botData_(nullptr),
      botDiff_(nullptr),
      topData_(nullptr),
      topDiff_(nullptr),
      engine_(mkldnn::engine::cpu, 0),
      inputElmCnt_(0),
      iMatH_(0), iMatW_(0),
      oMatH_(0), oMatW_(0),
      seqLen_(0), bs_(0),
      ic_(0), ih_(0), iw_(0),
      oc_(0), oh_(0), ow_(0),
      needResetBwd_(true),
      nextIsDnn_(false),
      scoreWithPaddleWgt_(false),
      preparedOnce_(false)
    {}

  ~MkldnnLayer() {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap) {
    /* Initialize the basic parent class */
    if (!Layer::init(layerMap, parameterMap)) {
      return false;
    }

    if (hasActivation()) {
      VLOG(DNN_BASE) << "Layer name: " << getName() << ", type: " << getType()
        << ", act: " << activation_->getName();
    } else {
      VLOG(DNN_BASE) << "Layer name: " << getName() << ", type: " << getType();
    }

    // buffers dims
    botDims_ = {};
    wgtDims_ = {};
    biasDims_ = {};
    topDims_ = {};
    botFmt_ = mkldnn::memory::format::nchw;
    wgtFmt_ = mkldnn::memory::format::format_undef;
    biasFmt_ = mkldnn::memory::format::x;
    topFmt_ = mkldnn::memory::format::nchw;

    // load from proto setting
    loadConfig();

    return initWgt(layerMap, parameterMap);
  }

  /**
   * @brief Forward propagation of MKLDNN
   */
  void forward(PassType passType) {
    passType_ = passType;
    if (inputElmCnt_ != getInputValue(0)->getElementCnt()) {
      VLOG(DNN_BASE) << "reshape mkldnn forward of layer: " << getName();
      inputElmCnt_ = getInputValue(0)->getElementCnt();
      iMatW_ = getInputValue(0)->getWidth();
      iMatH_ = getInputValue(0)->getHeight();
      CHECK_EQ(inputElmCnt_, iMatW_ * iMatH_);

      prepareOnce();

      reshape();
      printSizeInfo();

      CHECK_GT(oMatH_, 0);
      CHECK_EQ(oMatW_, getSize())
        << "maybe forget to set new layersize when reshape output info";
      resetOutput(oMatH_, oMatW_);

      resetFwd(passType);

      resetFwdAct();

      printDataFlow();

      if (scoreWithPaddleWgt_ && passType != PASS_TEST) {
        LOG(WARNING) << "scoreWithPaddleWgt_ is invalid in training";
      }
      needResetBwd_ = true;
    }
    {
      REGISTER_TIMER_INFO("mkldnn_FwdTimer", getName().c_str());
      Layer::forward(passType);
      submitFwd();
    }
  }

  /**
   * @brief Backward propagation of MKLDNN
   */
  void backward(const UpdateCallback& callback) {
    if (needResetBwd_) {
      needResetBwd_ = false;
      VLOG(DNN_BASE) << "reshape mkldnn backward of layer: " << getName();

      resetSumTopDiffs();

      resetBwd();

      resetBwdAct();

      printDiffFlow();
    }
    {
      REGISTER_TIMER_INFO("mkldnn_BwdTimer", getName().c_str());
      submitSumTopDiffs();
      submitBwd(callback);
    }
  }

  /**
   * Load settings from proto
   */
  virtual void loadConfig() {
    CHECK_EQ(config_.inputs_size(), 1) << "Only support one input config!";
    if (config_.has_score_with_paddle_wgt()) {
      scoreWithPaddleWgt_ = config_.score_with_paddle_wgt();
    }
  }

  /**
   * Initial weight of this layer
   */
  virtual bool initWgt(const LayerMap& layerMap,
                           const ParameterMap& parameterMap) = 0;

  /**
   * Reshape all the sizes needed including input, output and image sizes
   * Should take iMatH_ and iMatW_ as input
   * and cal the oMatH_ and oMatW_
   */
  virtual void reshape() = 0;

  /** 
   * Reset the primitive and buffer needed in Forward
   */
  virtual void resetFwd(PassType passType) = 0;

  /** 
   * Submit the forward primitive
   */
  virtual void submitFwd() = 0;

  /** 
   * each dnn layer should have function
   * to init or reset backward mkldnn
   */
  virtual void resetBwd() = 0;

  /** 
   * Submit the forward primitive
   */
  virtual void submitBwd(const UpdateCallback& callback) = 0;

  /** 
   * Reset the primitive and buffer needed in forward activation
   */
  virtual void resetFwdAct() {
    if (!hasMkldnnAct()) {
      return;
    }
    std::shared_ptr<mkldnn::memory::desc> md(
      new mkldnn::memory::desc(topData_->getUserMD()));
    CHECK(md);
    activation_->resetFwd(output_, std::static_pointer_cast<void>(md));
  }

  /** 
   * Reset the primitive and buffer needed in backward activation
   */
  virtual void resetBwdAct() {
    if (!hasMkldnnAct()) {
      return;
    }
    std::shared_ptr<mkldnn::memory::desc> md(
      new mkldnn::memory::desc(topDiff_->getUserMD()));
    CHECK(md);
    activation_->resetBwd(output_, std::static_pointer_cast<void>(md));
  }

  /**
   * Forward the activation
   * It will auto call mkldnn activation if have
   */
  virtual void forwardDnnAct() {
    forwardActivation();
  }

  /**
   * Backward the activation
   * It will auto call mkldnn activation if have
   */
  virtual void BackwardDnnAct() {
    backwardActivation();
  }

  /**
   * Reset primitive to sum the top diffs if have several branches
   * Used in GoogleNet, ResNet, etc.
   */
  virtual void resetSumTopDiffs() {
    // TODO(TJ): enable me
  }

  /**
   * Submit the primitive of summing the top diffs if have
   * Used in GoogleNet, ResNet, etc.
   */
  virtual void submitSumTopDiffs() {
    // TODO(TJ): enable me
  }


public:
  /**
   * Get mkldnn top data buffer
   */
  std::shared_ptr<void> getMkldnnTopData() override {
    return topData_;
  }

  /**
   * Get mkldnn bottom diff buffer
   */
  std::shared_ptr<void> getMkldnnBotDiff() override {
    return botDiff_;
  }

  /**
   * This function is to reorder the mkldnn weights format to
   * the format in original cpu layers
   */
  void reorderWeights() override {
    // TODO(TJ): enable me
  }

protected:
  /**
   * Functions that should be called only once after initial
   * should be placed in this functoin
   */
  void prepareOnce() {
    if (preparedOnce_) {
      return;
    }
    /*dnnOutGrads_.resize(nextLayers_.size(), nullptr);
    for (size_t i = 0; i < nextLayers_.size(); ++i) {
      topDiffMDs_.push_back(nullptr);
      dnnOutIdxMap_[nextLayers_[i]->getName()] = i;
    //  LOG(INFO)<<"next name:" << nextLayers_[i]->getName();
    }
    if (nextLayers_.size() > 0 && topDiffMDs_.size() > nextLayers_.size()) {
      // in base layer init will add one nullptr for PASS_grad check
      // so remove the redundant one
      topDiffMDs_.pop_back();
      CHECK_EQ(topDiffMDs_.size(), nextLayers_.size());
    } else {
      CHECK_EQ(topDiffMDs_.size() - 1, nextLayers_.size());
    }
    */
    initDnnflags();
    preparedOnce_ = true;
  }

  /**
   * init the flags whether to set memory desc
   * of top data or bot diff.
   * each layer can have its own implements.
   * Caution: Do not call it at init function
   *          this function can work only after all layers init have done
   */
  void initDnnflags() {
    // set topdata internal only if all next layers are MKLDNN layers
    nextIsDnn_ = areNextAllDnn();
    for (size_t i = 0; i != inputLayers_.size(); ++i) {
      prevIsDnn_.push_back(isPrevDnn(i));
    }
  }

  /**
   * Print some size info like input, output or image sizes
   */
  virtual void printSizeInfo() {
    VLOG(DNN_SIZES) << "bs: " << bs_
      << ", ic: " << ic_ << ", ih: " << ih_ << ", iw: " << iw_
      << ", oc: " << oc_ << ", oh: " << oh_ << ", ow: " << ow_;
  }

  /**
   * Print the mkldnn memory format flow of data
   */
  void printDataFlow() {
    if (botData_ && topData_) {
      VLOG(DNN_FMTS) << "data format flow --- "
        << botData_->getUserFmt() << " >>> ("
        << botData_->getIntlFmt() << " >>> "
        << topData_->getIntlFmt() << ") >>> "
        << topData_->getUserFmt();
    }
  }

  /**
   * Print the mkldnn memory format flow of diff
   */
  void printDiffFlow() {
    if (botDiff_ && topDiff_) {
      VLOG(DNN_FMTS) << "diff format flow --- "
        << botDiff_->getUserFmt() << " <<< ("
        << botDiff_->getIntlFmt()<< " <<< "
        << topDiff_->getIntlFmt() << ") <<< "
        << topDiff_->getUserFmt();
    }
  }

  // get the aligned seq length from paddle sequence info
  // and the length among batchsize should be the same
  int getPaddleAlignedSeqLen(const Argument& arg) {
    CHECK(arg.sequenceStartPositions);
    int sampleSize = arg.getBatchSize();  // bs*seqlen
    size_t numSequences = arg.getNumSequences();
    const int* starts = arg.sequenceStartPositions->getData(false);
    CHECK_EQ(starts[numSequences], sampleSize);
    int len = 0;
    for (size_t i = 0; i < numSequences; ++i) {
      int tmp = starts[i + 1] - starts[i];
      CHECK(len == 0 || len == tmp)
        << "all seq length should be equal," << len << " vs " << tmp;
      len = tmp;
    }
    return len;
  }

  /**
   * Calculate output size based on caffeMode_.
   * - input(+padding): 0123456789
   * - imageSize(+padding) = 10;
   * - filterSize = 3;
   * - stride = 2;
   * - caffeMode_ is true:
       - output: (012), (234), (456), (678)
       - outputSize = 4;
   * - caffeMode_ is false:
   *   - output: (012), (234), (456), (678), (9)
   *   - outputSize = 5;
   *** for conv only support caffe mode by now
   */
  int calOutputSize(int imageSize, int filterSize, int padding, int stride,
                       bool caffeMode = true) {
    int outputSize;
    if (!caffeMode) {
      outputSize =
          (imageSize - filterSize + 2 * padding + stride - 1) / stride + 1;
    } else {
      outputSize = (imageSize - filterSize + 2 * padding) / stride + 1;
    }
    CHECK_GE(outputSize, 1);
    return outputSize;
  }
};

}  // namespace paddle
