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
#include "MkldnnBase.h"
#include "mkldnn.hpp"

namespace paddle {

class MkldnnLayer;
typedef std::shared_ptr<MkldnnLayer> MkldnnLayerPtr;

/**
 * @brief Base class of Mkldnnlayer.
 *
 */
class MkldnnLayer : public Layer {
protected:
  // batch size
  int bs_;
  // input image channel, height and width
  int ic_, ih_, iw_;
  // output image channel, height and width
  int oc_, oh_, ow_;

  // mkldnn engine, stream and primivtives
  mkldnn::engine engine_;
  std::shared_ptr<MkldnnStream> stream_;

  std::shared_ptr<mkldnn::primitive> fwd_;
  std::shared_ptr<mkldnn::primitive> bwdWgt_;
  std::shared_ptr<mkldnn::primitive> bwdData_;
  std::vector<mkldnn::primitive> pipelineFwd_;
  std::vector<mkldnn::primitive> pipelineBwd_;

public:
  explicit MkldnnLayer(const LayerConfig& config)
      : Layer(config),
        bs_(0),
        ic_(0),
        ih_(0),
        iw_(0),
        oc_(0),
        oh_(0),
        ow_(0),
        engine_(mkldnn::engine::cpu, 0),
        stream_(nullptr),
        fwd_(nullptr),
        bwdWgt_(nullptr),
        bwdData_(nullptr) {}

  ~MkldnnLayer() {}

  virtual bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  void resetForwardFC(int bs,
                      int ic,
                      int ih,
                      int iw,
                      real* botData,
                      int oc,
                      real* topData,
                      real* wgtData,
                      real* biasData);

  void mkldnnForwardFC(int bs,
                       int ic,
                       int ih,
                       int iw,
                       real* botData,
                       int oc,
                       real* topData,
                       real* wgtData,
                       real* biasData);

  void resetBackwardFC(int bs,
                       int ic,
                       int ih,
                       int iw,
                       real* botDiff,
                       real* botData,
                       int oc,
                       real* topDiff,
                       real* wgtDiff,
                       real* wgtData,
                       real* biasDiff);

  void mkldnnBackwardFC(int bs,
                        int ic,
                        int ih,
                        int iw,
                        real* botDiff,
                        real* botData,
                        int oc,
                        real* topDiff,
                        real* wgtDiff,
                        real* wgtData,
                        real* biasDiff);

  // TODO(TJ): move to MkldnnMatrix
  // create memory desc
  inline mkldnn::memory::desc createMD(
      mkldnn::memory::dims dims,
      mkldnn::memory::format fmt,
      mkldnn::memory::data_type type = mkldnn::memory::data_type::f32);
};

}  // namespace paddle
