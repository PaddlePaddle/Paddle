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

#include "FactorizationMachineLayer.h"
#include <algorithm>
#include <vector>
#include "paddle/math/SparseMatrix.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_LAYER(factorization_machine, FactorizationMachineLayer);

bool FactorizationMachineLayer::init(const LayerMap& layerMap,
                                     const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  factorSize_ = config_.factor_size();

  /* initialize the latentVectors_ */
  CHECK_EQ(inputLayers_.size(), 1UL);
  size_t inputSize = inputLayers_[0]->getSize();
  CHECK_EQ(parameters_[0]->getSize(), inputSize * factorSize_);
  latentVectors_ = std::unique_ptr<Weight>(
      new Weight(inputSize, factorSize_, parameters_[0]));

  return true;
}

void FactorizationMachineLayer::forward(PassType passType) {
  Layer::forward(passType);

  const MatrixPtr& inputV = getInputValue(0);

  size_t batchSize = inputV->getHeight();
  size_t outputSize = getSize();
  size_t inputSize = inputLayers_[0]->getSize();
  reserveOutput(batchSize, outputSize);

  MatrixPtr outV = getOutputValue();

  Matrix::resizeOrCreate(
      latentVectorsSquare_, inputSize, factorSize_, false, useGpu_);
  Matrix::resizeOrCreate(
      inputMulFactor_, batchSize, factorSize_, false, useGpu_);
  Matrix::resizeOrCreate(tmpOut_, batchSize, factorSize_, false, useGpu_);

  REGISTER_TIMER_INFO("FmInputMulFactorTimer", getName().c_str());
  inputMulFactor_->mul(*inputV, *latentVectors_->getW());
  inputMulFactor_->square2(*tmpOut_);
  outV->sumRows(*tmpOut_, 0.5, 0);

  if (dynamic_cast<CpuSparseMatrix*>(inputV.get())) {
    Matrix::resizeOrCreateSparseMatrix(inputSquare_,
                                       inputV->getHeight(),
                                       inputV->getWidth(),
                                       inputV->getElementCnt(),
                                       inputV->getValueType());
    inputSquare_->copyFrom(*inputV);
    (dynamic_cast<CpuSparseMatrix*>(inputSquare_.get()))->square2();
  } else {
    Matrix::resizeOrCreate(
        inputSquare_, inputV->getHeight(), inputV->getWidth(), false, useGpu_);
    inputV->square2(*inputSquare_);
  }
  latentVectors_->getW()->square2(*latentVectorsSquare_);
  tmpOut_->mul(*inputSquare_, *latentVectorsSquare_);
  outV->sumRows(*tmpOut_, -0.5, 1.0);

  /* activation */ {
    REGISTER_TIMER_INFO("FmFwAtvTimer", getName().c_str());
    forwardActivation();
  }
}

void FactorizationMachineLayer::backward(const UpdateCallback& callback) {
  /* Do derivation */ { backwardActivation(); }

  const MatrixPtr& inputV = getInputValue(0);
  const MatrixPtr& oGrad = getOutputGrad();

  Matrix::resizeOrCreate(
      tmpSum_, 1, latentVectors_->getW()->getHeight(), false, useGpu_);
  MatrixPtr tmpSumTrans = Matrix::create(tmpSum_->getRowBuf(0),
                                         latentVectors_->getW()->getHeight(),
                                         1,
                                         false,
                                         useGpu_);

  /* Calculate the gradients of the latentVectors_ matrix */
  if (latentVectors_->getWGrad()) {
    if (dynamic_cast<CpuSparseMatrix*>(inputV.get())) {
      Matrix::resizeOrCreateSparseMatrix(tmpInput_,
                                         inputV->getHeight(),
                                         inputV->getWidth(),
                                         inputV->getElementCnt());

      CpuSparseMatrix* sparseInputV =
          dynamic_cast<CpuSparseMatrix*>(inputV.get());
      CpuSparseMatrix* sparseInputSquare =
          dynamic_cast<CpuSparseMatrix*>(inputSquare_.get());
      CpuSparseMatrix* sparseTmpInput =
          dynamic_cast<CpuSparseMatrix*>(tmpInput_.get());
      sparseTmpInput->copyFrom(*sparseInputV);

      sparseTmpInput->rowScale(0, *sparseInputV, *oGrad);
      latentVectors_->getWGrad()->mul(
          *sparseTmpInput->getTranspose(), *inputMulFactor_, 1, 1);
      sparseTmpInput->rowScale(0, *sparseInputSquare, *oGrad);

      Matrix::resizeOrCreate(negOnes_, 1, inputV->getHeight(), false, useGpu_);
      negOnes_->zeroMem();
      negOnes_->add(-1);
      tmpSum_->mul(*negOnes_, *sparseTmpInput, 1, 0);
    } else {
      Matrix::resizeOrCreate(
          tmpInput_, inputV->getHeight(), inputV->getWidth(), false, useGpu_);

      tmpInput_->rowScale(0, *inputV, *oGrad);
      latentVectors_->getWGrad()->mul(
          *tmpInput_->getTranspose(), *inputMulFactor_, 1, 1);
      tmpInput_->rowScale(0, *inputSquare_, *oGrad);

      tmpSum_->sumCols(*tmpInput_, -1, 0);
    }

    latentVectors_->getWGrad()->addRowScale(
        0, *latentVectors_->getW(), *tmpSumTrans);

    /* Increasing the number of gradient */
    latentVectors_->getParameterPtr()->incUpdate(callback);
  }

  /* Calculate the input layers gradient */
  MatrixPtr inGrad = getInputGrad(0);
  if (inGrad != NULL) {
    inGrad->mul(
        *inputMulFactor_, *latentVectors_->getW()->getTranspose(), 1, 1);
    tmpSumTrans->sumRows(*latentVectorsSquare_, -1, 0);
    inGrad->addColScale(0, *inputV, *tmpSum_);
    inGrad->rowScale(0, *inGrad, *oGrad);
  }
}

}  // namespace paddle
