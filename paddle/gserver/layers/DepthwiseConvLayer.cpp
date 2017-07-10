/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "DepthwiseConvLayer.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_LAYER(depthwise_conv, DepthwiseConvLayer);

bool DepthwiseConvLayer::init(const LayerMap &layerMap,
                              const ParameterMap &parameterMap) {
  /* Initialize the basic convolutional parent class */
  ExpandConvBaseLayer::init(layerMap, parameterMap);

  size_t numInputs = config_.inputs_size();
  inputShape_.resize(numInputs);
  filterShape_.resize(numInputs);
  outputShape_.resize(numInputs);

  for (int i = 0; i < config_.inputs_size(); i++) {
    std::vector<size_t> paddings = {(size_t)paddingY_[i], (size_t)padding_[i]};
    std::vector<size_t> strides = {(size_t)strideY_[i], (size_t)stride_[i]};
    createFunction(forward_,
                   "DepthwiseConv",
                   FuncConfig()
                       .set("paddings", paddings)
                       .set("strides", strides)
                       .set("groups", (size_t)groups_[i]));

    createFunction(backward_,
                   "DepthwiseConvGradInput",
                   FuncConfig()
                       .set("paddings", paddings)
                       .set("strides", strides)
                       .set("groups", (size_t)groups_[i]));

    createFunction(backward_,
                   "DepthwiseConvGradFilter",
                   FuncConfig()
                       .set("paddings", paddings)
                       .set("strides", strides)
                       .set("groups", (size_t)groups_[i]));
  }
  return true;
}

}  // namespace paddle
