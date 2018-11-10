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

#pragma once

#include <memory>

#include "CRFLayer.h"
#include "LinearChainCRF.h"

namespace paddle {

/**
 * A layer for calculating the decoding sequence of sequential conditional
 * random field model.
 * The decoding sequence is stored in output_.ids
 * It also calculate error, output_.value[i] is 1 for incorrect decoding
 * or 0 for correct decoding)
 * See LinearChainCRF.h for the detail of the CRF formulation.
 */
class CRFDecodingLayer : public CRFLayer {
 public:
  explicit CRFDecodingLayer(const LayerConfig& config) : CRFLayer(config) {}
  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;
  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback) override;

 protected:
  std::unique_ptr<LinearChainCRF> crf_;
};

}  // namespace paddle
