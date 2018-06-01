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

#include "Layer.h"
#include "paddle/math/Matrix.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

namespace paddle {

/**
 * @brief A layer for weighted sum of vectors,
 * which is used in NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND
 * TRANSLATE
 * - Input: the the size of the first input is weightDim,
 *          and the size of the second input is weightdim * dataDim.
 * - Output: the sizeof the output is dataDim
 * \f[
 *   out(j) = \sum_{i}(in0(i) * in1(i,j + i * dataDim)),
 *               i = 0,1,...,(weightDim-1); j = 0, 1,...,(dataDim-1)
 * \f]
 * Note that the above computation is for one sample. Multiple samples are
 * processed in one batch.
 *
 * The config file api is linear_comb_layer.
 */
class ConvexCombinationLayer : public Layer {
 protected:
  /// A matrix pointer pointing to second input.
  MatrixPtr tmpMtx0;
  /// A matrix pointer pointing to first input.
  MatrixPtr tmpRow0;
  /// A matrix pointer pointing to output.
  MatrixPtr tmpRow1;

 public:
  explicit ConvexCombinationLayer(const LayerConfig& config) : Layer(config) {}

  ~ConvexCombinationLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};

REGISTER_LAYER(convex_comb, ConvexCombinationLayer);

bool ConvexCombinationLayer::init(const LayerMap& layerMap,
                                  const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  CHECK_EQ(2U, inputLayers_.size());
  size_t dataDim = getSize();
  size_t weightDim = inputLayers_[0]->getSize();

  CHECK_EQ(weightDim * dataDim, inputLayers_[1]->getSize())
      << "Dimension mismatch";

  tmpRow0 = Matrix::create(nullptr,
                           /* height= */ 1,
                           weightDim,
                           /* trans= */ false,
                           useGpu_);
  tmpRow1 = Matrix::create(nullptr,
                           /* height= */ 1,
                           dataDim,
                           /* trans= */ false,
                           useGpu_);
  tmpMtx0 = Matrix::create(nullptr,
                           /* height= */ weightDim,
                           dataDim,
                           /* trans= */ false,
                           useGpu_);

  return true;
}

void ConvexCombinationLayer::forward(PassType passType) {
  Layer::forward(passType);

  MatrixPtr inV0 = getInputValue(0);
  MatrixPtr inV1 = getInputValue(1);

  size_t batchSize = inV0->getHeight();
  size_t weightDim = inV0->getWidth();
  size_t dataDim = getSize();

  CHECK_EQ(batchSize, inV1->getHeight());

  {
    REGISTER_TIMER_INFO("FwResetTimer", getName().c_str());
    reserveOutput(batchSize, dataDim);
  }

  MatrixPtr outV = getOutputValue();

  REGISTER_TIMER_INFO("FwCvxCombTimer", getName().c_str());
  for (size_t i = 0; i < batchSize; i++) {
    tmpMtx0->setData(inV1->getData() + i * weightDim * dataDim);
    tmpRow0->setData(inV0->getData() + i * weightDim);
    tmpRow1->setData(outV->getData() + i * dataDim);

    tmpRow1->mul(*tmpRow0, *tmpMtx0, 1, 0);
  }
}

void ConvexCombinationLayer::backward(const UpdateCallback& callback) {
  MatrixPtr outG = getOutputGrad();
  MatrixPtr inV0 = getInputValue(0);
  MatrixPtr inV1 = getInputValue(1);
  MatrixPtr inG0 = getInputGrad(0);
  MatrixPtr inG1 = getInputGrad(1);

  size_t batchSize = inV0->getHeight();
  size_t weightDim = inV0->getWidth();
  size_t dataDim = getSize();

  REGISTER_TIMER_INFO("BwCvxCombTimer", getName().c_str());

  if (inG0) {
    for (size_t i = 0; i < batchSize; i++) {
      tmpRow0->setData(inG0->getData() + i * weightDim);
      tmpRow1->setData(outG->getData() + i * dataDim);
      tmpMtx0->setData(inV1->getData() + i * weightDim * dataDim);

      tmpRow0->mul(*tmpRow1, *(tmpMtx0->getTranspose()), 1, 1);
    }
  }

  if (inG1) {
    for (size_t i = 0; i < batchSize; i++) {
      tmpRow0->setData(inV0->getData() + i * weightDim);
      tmpRow1->setData(outG->getData() + i * dataDim);
      tmpMtx0->setData(inG1->getData() + i * weightDim * dataDim);

      tmpMtx0->mul(*(tmpRow0->getTranspose()), *tmpRow1, 1, 1);
    }
  }
}

}  // namespace paddle
