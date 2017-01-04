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

#include "CosSimVecMatLayer.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

namespace paddle {

/**
 * @brief A layer for computing cosine similarity between a vector
 * and each row of a matrix
 * out[i] = cos_scale * cos(in1, in2(i,:));
 * @note used in NEURAL TURING MACHINE
 *
 * Input1: a vector (batchSize * dataDim)
 *
 * Input2: a matrix in vector form (batchSize * (weightDim*dataDim))
 *
 * Output: a vector (batchSize * weightDim)
 */

class CosSimVecMatLayer : public Layer {
protected:
  MatrixPtr tmpMtx0;
  MatrixPtr tmpMtx1;
  MatrixPtr tmpRow0;
  MatrixPtr tmpRow1;
  MatrixPtr tmpRow2;
  MatrixPtr tmpRow3;

public:
  explicit CosSimVecMatLayer(const LayerConfig& config) : Layer(config) {}

  ~CosSimVecMatLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};

REGISTER_LAYER(cos_vm, CosSimVecMatLayer);

bool CosSimVecMatLayer::init(const LayerMap& layerMap,
                             const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);

  CHECK_EQ(inputLayers_.size(), 2U);

  size_t dataDim = inputLayers_[0]->getSize();
  size_t numKeys = getSize();
  size_t memoryDim = inputLayers_[1]->getSize();

  CHECK_EQ(dataDim * numKeys, memoryDim) << "Dimension mismatch";

  tmpRow0 = Matrix::create(nullptr,
                           /* height= */ 1,
                           dataDim,
                           /* trans= */ false,
                           useGpu_);
  tmpRow1 = Matrix::create(nullptr,
                           /* height= */ 1,
                           dataDim,
                           /* trans= */ false,
                           useGpu_);
  tmpRow2 = Matrix::create(nullptr,
                           /* height= */ numKeys,
                           1,
                           /* trans= */ false,
                           useGpu_);
  tmpRow3 = Matrix::create(nullptr,
                           /* height= */ numKeys,
                           1,
                           /* trans= */ false,
                           useGpu_);

  tmpMtx0 = Matrix::create(nullptr,
                           /* height= */ numKeys,
                           dataDim,
                           /* trans= */ false,
                           useGpu_);
  tmpMtx1 = Matrix::create(nullptr,
                           /* height= */ numKeys,
                           dataDim,
                           /* trans= */ false,
                           useGpu_);

  /// todo(tianbing), do we really need to check these shared pointers?
  CHECK(tmpRow0 && tmpRow1 && tmpRow2 && tmpRow3 && tmpMtx0 && tmpMtx1);

  createFunction(forward_,
                 "CosSimForward",
                 FuncConfig().set("scale", (real)config_.cos_scale()));
  createFunction(backward_,
                 "CosSimBackward",
                 FuncConfig().set("scale", (real)config_.cos_scale()));

  return true;
}

void CosSimVecMatLayer::forward(PassType passType) {
  Layer::forward(passType);
  CHECK_EQ(forward_.size(), 1) << "Only one forward function needed";

  MatrixPtr inV0 = getInputValue(0);
  MatrixPtr inV1 = getInputValue(1);

  size_t batchSize = inV0->getHeight();
  size_t numKeys = getSize();

  CHECK_EQ(batchSize, inV1->getHeight());

  {
    REGISTER_TIMER_INFO("FwResetTimer", getName().c_str());
    reserveOutput(batchSize, numKeys);
  }

  MatrixPtr outV = getOutputValue();
  CHECK(outV && inV0 && inV1);
  REGISTER_TIMER_INFO("FwCosVMTimer", getName().c_str());
  for (size_t i = 0; i < batchSize; i++) {
    tmpRow0->setData(inV0->rowBuf(i));
    tmpMtx0->setData(inV1->rowBuf(i));
    tmpRow2->setData(outV->rowBuf(i));

    forward_[0]->calc({Tensor(tmpMtx0->getData(),
                              Dims{tmpMtx0->getHeight(), tmpMtx0->getWidth()}),
                       Tensor(tmpRow0->getData(),
                              Dims{tmpRow0->getHeight(), tmpRow0->getWidth()})},
                      {Tensor(tmpRow2->getData(),
                              Dims{tmpRow2->getHeight(), tmpRow2->getWidth()})},
                      {});
  }
}

void CosSimVecMatLayer::backward(const UpdateCallback& callback) {
  CHECK_EQ(backward_.size(), 1) << "Only one forward function needed";

  MatrixPtr inV0 = getInputValue(0);
  MatrixPtr inV1 = getInputValue(1);
  MatrixPtr inG0 = getInputGrad(0);
  MatrixPtr inG1 = getInputGrad(1);
  MatrixPtr outV = getOutputValue();
  MatrixPtr outG = getOutputGrad();

  size_t batchSize = inV0->getHeight();
  CHECK(inV0 && inV1 && inG0 && inG1 && outV && outG);
  REGISTER_TIMER_INFO("BwCosVMTimer", getName().c_str());

  for (size_t i = 0; i < batchSize; i++) {
    tmpRow0->setData(inV0->rowBuf(i));
    tmpRow1->setData(inG0->rowBuf(i));
    tmpMtx0->setData(inV1->rowBuf(i));
    tmpMtx1->setData(inG1->rowBuf(i));
    tmpRow2->setData(outV->rowBuf(i));
    tmpRow3->setData(outG->rowBuf(i));

    backward_[0]->calc(
        {Tensor(tmpRow2->getData(),
                Dims{tmpRow2->getHeight(), tmpRow2->getWidth()}),
         Tensor(tmpMtx0->getData(),
                Dims{tmpMtx0->getHeight(), tmpMtx0->getWidth()}),
         Tensor(tmpRow0->getData(),
                Dims{tmpRow0->getHeight(), tmpRow0->getWidth()}),
         Tensor(tmpMtx1->getData(),
                Dims{tmpMtx1->getHeight(), tmpMtx1->getWidth()}),
         Tensor(tmpRow1->getData(),
                Dims{tmpRow1->getHeight(), tmpRow1->getWidth()})},
        {Tensor(tmpRow3->getData(),
                Dims{tmpRow3->getHeight(), tmpRow3->getWidth()})},
        {});
  }
}

}  // namespace paddle
