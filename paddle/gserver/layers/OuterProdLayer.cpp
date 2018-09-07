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
 * @brief A layer for computing the outer product of two vectors
 * @note used in NEURAL TURING MACHINE
 * Input1: vector (batchSize * dim1)
 * Input2: vector (batchSize * dim2)
 * Output: a matrix: (batchSize * (dim1*dim2))
 */

class OuterProdLayer : public Layer {
 protected:
  MatrixPtr tmpMtx0;
  MatrixPtr tmpRow0;
  MatrixPtr tmpRow1;

 public:
  explicit OuterProdLayer(const LayerConfig& config) : Layer(config) {}

  ~OuterProdLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};

REGISTER_LAYER(out_prod, OuterProdLayer);

bool OuterProdLayer::init(const LayerMap& layerMap,
                          const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);

  CHECK_EQ(inputLayers_.size(), 2U);

  size_t dim0 = inputLayers_[0]->getSize();
  size_t dim1 = inputLayers_[1]->getSize();

  CHECK_EQ(dim0 * dim1, getSize()) << "Dimension mismatch";

  tmpRow0 = Matrix::create(
      nullptr, /* height= */ 1, dim0, /* trans= */ false, useGpu_);
  tmpRow1 = Matrix::create(
      nullptr, /* height= */ 1, dim1, /* trans= */ false, useGpu_);
  tmpMtx0 = Matrix::create(nullptr,
                           /* height= */ dim0,
                           dim1,
                           /* trans= */ false,
                           useGpu_);
  return true;
}

void OuterProdLayer::forward(PassType passType) {
  Layer::forward(passType);

  MatrixPtr inV0 = getInputValue(0);
  MatrixPtr inV1 = getInputValue(1);

  size_t batchSize = inV0->getHeight();
  size_t dim0 = inV0->getWidth();
  size_t dim1 = inV1->getWidth();

  CHECK_EQ(dim0 * dim1, getSize());
  CHECK_EQ(inV1->getHeight(), batchSize);

  {
    REGISTER_TIMER_INFO("FwResetTimer", getName().c_str());
    reserveOutput(batchSize, dim0 * dim1);
  }

  MatrixPtr outV = getOutputValue();

  {
    REGISTER_TIMER_INFO("FwOutProdTimer", getName().c_str());
    for (size_t i = 0; i < batchSize; i++) {
      tmpMtx0->setData(outV->getData() + i * dim0 * dim1);
      tmpRow0->setData(inV0->getData() + i * dim0);
      tmpRow1->setData(inV1->getData() + i * dim1);

      tmpMtx0->mul(*tmpRow0->getTranspose(), *tmpRow1);
    }
  }
}

void OuterProdLayer::backward(const UpdateCallback& callback) {
  MatrixPtr inV0 = getInputValue(0);
  MatrixPtr inV1 = getInputValue(1);
  MatrixPtr outG = getOutputGrad();
  MatrixPtr inG0 = getInputGrad(0);
  MatrixPtr inG1 = getInputGrad(1);

  size_t batchSize = inV0->getHeight();
  size_t dim0 = inV0->getWidth();
  size_t dim1 = inV1->getWidth();

  {
    REGISTER_TIMER_INFO("BwOutProdTimer", getName().c_str());

    if (inG0) {
      for (size_t i = 0; i < batchSize; i++) {
        tmpMtx0->setData(outG->getData() + i * dim0 * dim1);
        tmpRow0->setData(inG0->getData() + i * dim0);
        tmpRow1->setData(inV1->getData() + i * dim1);

        tmpRow0->mul(*tmpRow1, *tmpMtx0->getTranspose(), 1, 1);
      }
    }

    if (inG1) {
      for (size_t i = 0; i < batchSize; i++) {
        tmpMtx0->setData(outG->getData() + i * dim0 * dim1);
        tmpRow0->setData(inV0->getData() + i * dim0);
        tmpRow1->setData(inG1->getData() + i * dim1);

        tmpRow1->mul(*tmpRow0, *tmpMtx0, 1, 1);
      }
    }
  }
}

}  // namespace paddle
