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

#include "TensorLayer.h"

#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_LAYER(tensor, TensorLayer);

bool TensorLayer::init(const LayerMap& layerMap,
                       const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  /* initialize the weightList */
  CHECK_EQ(inputLayers_.size(), 2LU);
  CHECK(parameters_[0]);
  CHECK(!parameters_[1]);

  // Option the parameters
  size_t height = inputLayers_[0]->getSize();
  size_t width = inputLayers_[1]->getSize();
  CHECK_EQ(width * height * getSize(), parameters_[0]->getSize());

  for (size_t i = 0; i < getSize(); ++i) {
    // create a new weight
    Weight* w = new Weight(height, width, parameters_[0], i * width * height);

    // append the new weight to the list
    weights_.emplace_back(w);
  }

  /* initialize biases_ */
  if (biasParameter_.get() != NULL) {
    biases_ = std::unique_ptr<Weight>(new Weight(1, getSize(), biasParameter_));
  }

  return true;
}

void TensorLayer::forward(PassType passType) {
  Layer::forward(passType);

  /* malloc memory for the output_ if necessary */
  int batchSize = getInputValue(0)->getHeight();
  int size = getSize();

  { resetOutput(batchSize, size); }

  MatrixPtr outV = getOutputValue();
  /* add the bias-vector */
  if (biases_.get() != NULL) {
    outV->addBias(*(biases_->getW()), 1);
  }

  /* e1 * W * trans(e2) */ {
    MatrixPtr input1 = getInputValue(0);
    MatrixPtr input2 = getInputValue(1);
    MatrixPtr tmpMat = Matrix::create(input2->getHeight(),
                                      input2->getWidth(),
                                      /* trans= */ false,
                                      input2->useGpu());
    REGISTER_TIMER_INFO("TensorFwMulTimer", getName().c_str());
    for (size_t i = 0; i < getSize(); ++i) {
      MatrixPtr weights = weights_[i]->getW();
      tmpMat->mul(*input1, *weights, 1, 0);
      outV->rowDotMul(i, *tmpMat, *input2);
    }
  }

  /* activation */ { forwardActivation(); }
}

void TensorLayer::backward(const UpdateCallback& callback) {
  /* Do derivation */ { backwardActivation(); }

  if (biases_ && biases_->getWGrad()) {
    biases_->getWGrad()->collectBias(*getOutputGrad(), 1);

    /* Increasing the number of gradient */
    biases_->getParameterPtr()->incUpdate(callback);
  }

  bool syncFlag = hl_get_sync_flag();

  /* Calculate the W-gradient for the current layer */
  MatrixPtr input1 = getInputValue(0);
  MatrixPtr input2 = getInputValue(1);
  MatrixPtr oGrad = getOutputGrad();
  MatrixPtr tmpMat = Matrix::create(input1->getHeight(),
                                    input1->getWidth(),
                                    /* trans= */ false,
                                    input1->useGpu());

  /* trans(grad * e1) * e2 */ {
    REGISTER_TIMER_INFO("TensorGradMulTimer", getName().c_str());
    for (size_t i = 0; i < getSize(); ++i) {
      if (weights_[i]->getWGrad()) {
        tmpMat->rowScale(i, *input1, *oGrad);
        MatrixPtr input1_T = tmpMat->getTranspose();
        weights_[i]->getWGrad()->mul(*input1_T, *input2, 1, 1);
      }
    }
  }

  hl_set_sync_flag(false);

  /* Calculate the input layers error */ {
    MatrixPtr preGrad1 = getInputGrad(0);
    MatrixPtr preGrad2 = getInputGrad(1);

    REGISTER_TIMER_INFO("TensorBpMulTimer", getName().c_str());
    for (size_t i = 0; i < getSize(); ++i) {
      MatrixPtr weights = weights_[i]->getW();

      if (NULL != preGrad1) { /* (grad * e2) * trans(W) */
        tmpMat->rowScale(i, *input2, *oGrad);
        MatrixPtr weights_T = weights->getTranspose();
        preGrad1->mul(*tmpMat, *weights_T, 1, 1);
      }
      if (NULL != preGrad2) { /* (grad * e1) * W */
        tmpMat->rowScale(i, *input1, *oGrad);
        preGrad2->mul(*tmpMat, *weights, 1, 1);
      }
    }
  }
  hl_set_sync_flag(syncFlag);
  parameters_[0]->incUpdate(callback);
}

}  // namespace paddle
