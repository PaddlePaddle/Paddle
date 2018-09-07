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

#include "TransLayer.h"
#include "paddle/utils/Logging.h"
namespace paddle {

REGISTER_LAYER(trans, TransLayer);

bool TransLayer::init(const LayerMap& layerMap,
                      const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  /* the size of inputs for trans-layer is 1 */
  CHECK_EQ(config_.inputs_size(), 1);

  return true;
}

void TransLayer::forward(PassType passType) {
  Layer::forward(passType);

  /* malloc memory for the output_ if necessary */
  MatrixPtr input = getInputValue(0);
  int height = input->getHeight();
  int width = input->getWidth();

  resizeOutput(width, height);

  MatrixPtr outV = getOutputValue();

  /* outV's memory has been allocated, so memAlloc = false */
  input->transpose(outV, false);
  if (getInputGrad(0)) {
    zeroGrad();
  }
}

void TransLayer::backward(const UpdateCallback& callback) {
  (void)callback;

  MatrixPtr outputGrad = getOutputGrad();
  if (outputGrad == NULL) {
    return;
  }
  MatrixPtr preGrad = getInputGrad(0);
  if (preGrad) {
    MatrixPtr transGrad = Matrix::create(preGrad->getHeight(),
                                         preGrad->getWidth(),
                                         /* trans= */ false,
                                         preGrad->useGpu());
    outputGrad->transpose(transGrad, false);
    preGrad->add(*transGrad);
  }
}

}  // namespace paddle
