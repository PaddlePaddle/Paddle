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

#include "CrossEntropyOverBeam.h"

namespace paddle {

REGISTER_LAYER(cross_entropy_over_beam, CrossEntropyOverBeam);

bool CrossEntropyOverBeam::init(const LayerMap& layerMap,
                                const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  setNeedSequenceInfo(false);

  return true;
}

void CrossEntropyOverBeam::forward(PassType passType) {}

void CrossEntropyOverBeam::backward(const UpdateCallback& callback) {}

}  // namespace paddle
