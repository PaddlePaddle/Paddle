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
 * @brief Base class of Dnnlayer.
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
  size_t inputMatH_, inputMatW_;

  // the output matrix height and width
  // height: mklSeqLen * bs
  // width : layer size == oc * oh * ow
  size_t outputMatH_, outputMatW_;

  // MKLDNN aligned seqLen
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
      needResetBwd_(true),
      nextIsDnn_(false),
      scoreWithPaddleWgt_(false),
      preparedOnce_(false)
    {}

  ~MkldnnLayer() {}

  /**
   * Get mkldnn top data memory
   */
  virtual std::shared_ptr<void> getMkldnnTopData() {
    return topData_;
  }

  /**
   * Get mkldnn bottom diff memory
   */
  virtual std::shared_ptr<void> getMkldnnBotDiff() {
    return botDiff_;
  }

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap) {
    /* Initialize the basic parent class */
    if (!Layer::init(layerMap, parameterMap)) {
      return false;
    }

    if (hasActivation()) {
      VLOG(1) << "Layer name: " << getName() << ", type: " << getType()
        << ", act: " << activation_->getName();
    } else {
      VLOG(1) << "Layer name: " << getName() << ", type: " << getType();
    }

    inputElmCnt_ = 0;
    seqLen_ = 0;

    // buffers
    botDims_ = {};
    wgtDims_ = {};
    biasDims_ = {};
    topDims_ = {};
    botFmt_ = mkldnn::memory::format::nchw;
    wgtFmt_ = mkldnn::memory::format::format_undef;
    biasFmt_ = mkldnn::memory::format::x;
    topFmt_ = mkldnn::memory::format::nchw;
    bs_ = 0; 
    oc_ = 0; ih_ = 0; iw_ = 0;
    ic_ = 0; oh_ = 0; ow_ = 0;

    // load from proto setting
    loadConfig();

    return initWgt(layerMap, parameterMap);
  }

  void forward(PassType passType) {
    passType_ = passType;
    if (inputElmCnt_ != getInputValue(0)->getElementCnt()) {
      VLOG(1) << "reshape mkldnn fwd of layer: " << getName();

      prepareOnce();
      
      updateInputShape();

      reshapeOutputInfo();

      reshapeOutputBuffer();

      resetFwd(passType);

      resetFwdAct();

      printDataFlow();

      needResetBwd_ = true;

      printInfo();
      
      if (scoreWithPaddleWgt_ && passType != PASS_TEST) {
        LOG(WARNING) << "scoreWithPaddleWgt_ is invalid when training";
      }
    }

    {
      //REGISTER_TIMER_DYNAMIC("Fwd_" + getName());
      REGISTER_TIMER_INFO("mkldnn_FwdTimer", getName().c_str());
      Layer::forward(passType);
      submitFwd();
    }
  }

  void backward(const UpdateCallback& callback) {
    if (needResetBwd_) {
      needResetBwd_ = false;
      VLOG(1) << "reshape mkldnn bwd of layer: " << getName();

      gatherTopDiffs();

      resetBwd();

      resetBwdAct();

      printDiffFlow();      
    }
    {
      //REGISTER_TIMER_DYNAMIC("Bwd_" + getName());
      REGISTER_TIMER_INFO("mkldnn_BwdTimer", getName().c_str());
      /*
      sumbitSumTopDiffs();
      if (nullptr != sumTopDiffs_) {
        std::vector<primitive> sum;
        sum.push_back(*sumTopDiffs_);
        mkldnn::stream(mkldnn::stream::kind::eager).submit(sum).wait();
      }
      */
      submitBwd(callback);
    }
  }

  // reset activation fwd
  virtual void resetFwdAct() {
    if (!hasMkldnnAct()) {
      return;
    }
    std::shared_ptr<mkldnn::memory::desc> md(
      new mkldnn::memory::desc(topData_->getUserMD()));
    CHECK(md);
    activation_->resetFwd(output_, std::static_pointer_cast<void>(md));
  }

  // reset activation bwd
  virtual void resetBwdAct() {
    if (!hasMkldnnAct()) {
      return;
    }
    std::shared_ptr<mkldnn::memory::desc> md(
      new mkldnn::memory::desc(topDiff_->getUserMD()));
    CHECK(md);
    activation_->resetBwd(output_, std::static_pointer_cast<void>(md));
  }

  // the activation will auto call dnn act if have
  virtual void forwardDnnAct() {
    forwardActivation();
  }

  // the activation will auto call dnn act if have
  virtual void BackwardDnnAct() {
    backwardActivation();
  }

  // reshape the buffer of output
  virtual void reshapeOutputBuffer() {
    CHECK_EQ(outputMatW_, getSize())
      << "maybe forget to set new layersize when reshape output info";
    resetOutput(outputMatH_, outputMatW_);
  }

  // reload the settings from proto
  virtual void loadConfig() = 0;

  /**
   * each dnn layer should have function 
   * to init weight
   */
  virtual bool initWgt(const LayerMap& layerMap,
                           const ParameterMap& parameterMap) = 0;

  /** 
   * each dnn layer should have function
   * to init or reset forward mkldnn
   */
  virtual void resetFwd(PassType passType) = 0;

  /** 
   * each dnn layer should have function
   * to init or reset backward mkldnn
   */
  virtual void resetBwd() = 0;

  virtual void submitFwd() = 0;

  virtual void submitBwd(const UpdateCallback& callback) = 0;

  // reshape output info:
  // output matrix height and width 
  // bs and sometimes seqlen
  virtual void reshapeOutputInfo() = 0;





  

protected:
  // TODO: add comment or rename
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

  void updateInputShape() {
    inputElmCnt_ = getInputValue(0)->getElementCnt();
    inputMatW_ = getInputValue(0)->getWidth();
    inputMatH_ = getInputValue(0)->getHeight();
    CHECK_EQ(inputElmCnt_, inputMatW_ * inputMatH_);
  }

  void printDataFlow() {
    if (botData_ && topData_) {
      VLOG(1) << "data format flow --- "
        << botData_->getUserFmt() << " >>> ("
        << botData_->getIntlFmt() << " >>> "
        << topData_->getIntlFmt() << ") >>> "
        << topData_->getUserFmt();
    }
  }

  void printDiffFlow() {
    // print the diff flow
    if (botDiff_ && topDiff_) {
      VLOG(1) << "diff format flow --- "
        << botDiff_->getUserFmt() << " <<< ("
        << botDiff_->getIntlFmt()<< " <<< "
        << topDiff_->getIntlFmt() << ") <<< "
        << topDiff_->getUserFmt();
    }
  }

  /**
   * if have several input topdiffs
   * then create a handle to sum them
   */
  virtual void gatherTopDiffs() {
    /*sumTopDiffs_ = nullptr;
    if (nextLayers_.size() <= 1)
      return;
    engine eg = CpuEngine::Instance().getEngine();
    std::vector<memory::primitive_desc> srcPDs;
    std::vector<std::shared_ptr<memory::desc>> prvMDs;
    std::vector<primitive::at> srcMems;
    std::vector<double> scales;
    CHECK_EQ(nextLayers_.size(), topDiffBuffers_.size());
    for (size_t i = 0; i < topDiffBuffers_.size(); ++i) {
      // 1. create buffers and init user
      real* diff = dnnOutGrads_[i]->getData();
      topDiffBuffers_[i].reset(new MkldnnBuffer());
      topDiffBuffers_[i]->initUser(diff, topDims_, topFmt_, eg);
      // 2. use private MD when init user if has
      // TODO(TJ): any improvment if prvs are different format?
      const std::shared_ptr<memory::desc>& prvMD = getTopDiffMD(i);
      if (prvMD) {
        topDiffBuffers_[i]->resetUser(diff, *prvMD, eg);
        prvMDs.push_back(prvMD);
      }
      // 3. init Intl with empty cvt
      topDiffBuffers_[i]->initCvt();
      CHECK(i == 0 || (topDiffBuffers_[i-1]->getIntlSize()
        == topDiffBuffers_[i]->getIntlSize())) << "All size should be equal";
      VLOG(1) << "TopDiff format: " << DNN_FMTS[topDiffBuffers_[i]->getIntlFmt()];
      // 4. save buffers
      scales.push_back(1.0);  // no scale
      srcPDs.push_back(topDiffBuffers_[i]->getIntlPD());
      srcMems.push_back(*(topDiffBuffers_[i]->getIntlMem()));
    }
    if (prvMDs.size() > 0 && prvMDs.size() != nextLayers_.size()) {
      LOG(INFO) << "prvMDs.size() != nextLayers_.size(): " << prvMDs.size()
        << " vs " << nextLayers_.size();
      LOG(INFO) << "Next layers: ";
      for (size_t i = 0; i < nextLayers_.size(); ++i) {
        LOG(INFO) << nextLayers_[i]->getName()
          << ", type: " << nextLayers_[i]->getType();
      }
      LOG(FATAL)  << "Do not support mixed layer type inside branch";
    }
    // 5. create sum PD
    std::shared_ptr<sum::primitive_desc> sumPD;
    sumPD.reset(new sum::primitive_desc(
      MkldnnBuffer::getMD(topDims_), scales, srcPDs));
    // 6. init the buffer of result
    tmpDiff_.reset(new MkldnnBuffer());
    real *topDiffData = getDnnOutputGrad()->getData();
    tmpDiff_->initUser(topDiffData, sumPD->dst_primitive_desc());
    tmpDiff_->initCvt();
    // change the first intl MD
    topDiffMDs_[0].reset(new memory::desc(tmpDiff_->getIntlMD()));
    // 7. create sum handle
    sumTopDiffs_.reset(new sum(*sumPD, srcMems, *(tmpDiff_->getIntlMem())));
    */
  };

  /**
   * print some info like input or output size
   */
  virtual void printInfo() {
    
    VLOG(2) << "bs: " << bs_
      << ", ic: " << ic_ << ", ih: " << ih_ << ", iw: " << iw_
      << ", oc: " << oc_ << ", oh: " << oh_ << ", ow: " << ow_;
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
  int getOutputSize(int imageSize, int filterSize, int padding, int stride,
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
