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

#include <vector>
#include "Layer.h"
#include "MKLDNNBase.h"
#include "mkldnn.hpp"
#include "paddle/math/MKLDNNMatrix.h"

DECLARE_bool(use_mkldnn);
DECLARE_bool(use_mkldnn_wgt);

namespace paddle {

class MKLDNNLayer;
typedef std::shared_ptr<MKLDNNLayer> MKLDNNLayerPtr;

/**
 * @brief Base class of MKLDNNlayer.
 *
 */
class MKLDNNLayer : public Layer {
protected:
  // batch size
  int bs_;
  // input image channel, height and width
  int ic_, ih_, iw_;
  // output image channel, height and width
  int oc_, oh_, ow_;

  // backward also need reset after reset forward handle
  bool needResetBwd_;

  // mkldnn engine, stream and primivtives
  mkldnn::engine engine_;
  std::shared_ptr<MKLDNNStream> stream_;
  std::shared_ptr<mkldnn::primitive> fwd_;
  std::shared_ptr<mkldnn::primitive> bwdWgt_;
  std::shared_ptr<mkldnn::primitive> bwdData_;
  std::vector<mkldnn::primitive> pipelineFwd_;
  std::vector<mkldnn::primitive> pipelineBwd_;

  // TODO(TJ): change below memory as MKLDNNMatrixPtr type
  std::shared_ptr<mkldnn::memory> inVal_;
  std::shared_ptr<mkldnn::memory> inGrad_;
  std::shared_ptr<mkldnn::memory> outVal_;
  std::shared_ptr<mkldnn::memory> outGrad_;
  std::shared_ptr<mkldnn::memory> wgtVal_;
  std::shared_ptr<mkldnn::memory> wgtGrad_;
  std::shared_ptr<mkldnn::memory> biasVal_;
  std::shared_ptr<mkldnn::memory> biasGrad_;

public:
  explicit MKLDNNLayer(const LayerConfig& config)
      : Layer(config),
        bs_(0),
        ic_(0),
        ih_(0),
        iw_(0),
        oc_(0),
        oh_(0),
        ow_(0),
        needResetBwd_(true),
        engine_(mkldnn::engine::cpu, 0),
        stream_(nullptr),
        fwd_(nullptr),
        bwdWgt_(nullptr),
        bwdData_(nullptr) {}

  ~MKLDNNLayer() {}

  virtual bool init(const LayerMap& layerMap,
                    const ParameterMap& parameterMap) {
    if (!Layer::init(layerMap, parameterMap)) {
      return false;
    }

    CHECK(FLAGS_use_mkldnn) << "MkldnnLayers only support use_mkldnn."
                            << "Please set WITH_MKLDNN=ON "
                            << "and set use_mkldnn=True";
    stream_.reset(new MKLDNNStream());
    engine_ = CPUEngine::Instance().getEngine();

    // TODO(TJ): deivecId
    return true;
  }

  /**
   * convert weight from paddle format to mkldnn format
   * weight_ will be override
   */
  virtual void convertWeightsFromPaddle() {}

  /**
   * convert mkldnn weight to paddle format
   * weight_ will be override
   */
  virtual void convertWeightsToPaddle() {}

  /**
   * print info about sizes
   */
  virtual void printSizeInfo() {
    VLOG(MKLDNN_SIZES) << getName() << ": bs: " << bs_ << ", ic: " << ic_
                       << ", ih: " << ih_ << ", iw: " << iw_ << ", oc: " << oc_
                       << ", oh: " << oh_ << ", ow: " << ow_;
  }

  // TODO(TJ): move to MkldnnMatrix
  // create memory desc
  inline mkldnn::memory::desc createMD(
      mkldnn::memory::dims dims,
      mkldnn::memory::format fmt,
      mkldnn::memory::data_type type = mkldnn::memory::data_type::f32) {
    // TODO(TJ): isFmtSuppoted(fmt)
    return mkldnn::memory::desc(dims, type, fmt);
  }
};

}  // namespace paddle
