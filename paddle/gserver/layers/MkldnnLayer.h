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
#include "mkldnn.hpp"

namespace paddle {

class MkldnnLayer;
typedef std::shared_ptr<MkldnnLayer> MkldnnLayerPtr;

/**
 * @brief Base class of Mkldnnlayer.
 *
 */
class MkldnnLayer : public Layer {
public:
  explicit MkldnnLayer(const LayerConfig& config) : Layer(config) {}

  ~MkldnnLayer() {}

  virtual bool init(const LayerMap& layerMap,
                    const ParameterMap& parameterMap) {
    return Layer::init(layerMap, parameterMap);
    // TODO(TJ): deivecId
  }

  void resetOutput(size_t height, size_t width) { ; }
};

}  // namespace paddle
