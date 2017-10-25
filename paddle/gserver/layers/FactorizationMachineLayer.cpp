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
  size_t height = inputLayers_[0]->getSize();
  CHECK_EQ(parameters_[0]->getSize(), height * factorSize_);
  latentVectors_ =
      std::unique_ptr<Weight>(new Weight(height, factorSize_, parameters_[0]));

  v2_ = Matrix::create(height, factorSize_, false, useGpu_);

  return true;
}

void FactorizationMachineLayer::forward(PassType passType) {
  Layer::forward(passType);

  const MatrixPtr& inputV = getInputValue(0);

  size_t batchSize = inputV->getHeight();
  size_t size = getSize();
  reserveOutput(batchSize, size);

  MatrixPtr outV = getOutputValue();

  Matrix::resizeOrCreate(tmpMul_, batchSize, factorSize_, false, useGpu_);
  Matrix::resizeOrCreate(tmpOut_, batchSize, factorSize_, false, useGpu_);

  REGISTER_TIMER_INFO("FwMulTimer", getName().c_str());
  tmpMul_->mul(*inputV, *latentVectors_->getW());
  tmpMul_->square2(*tmpOut_);
  outV->sumRows(*tmpOut_, 0.5, 0);

  x2_ = inputV->clone(0, 0, useGpu_);
  if (dynamic_cast<CpuSparseMatrix*>(x2_.get())) {
    x2_->copyFrom(*inputV);
    (dynamic_cast<CpuSparseMatrix*>(x2_.get()))->square2();
  } else {
    inputV->square2(*x2_);
  }
  latentVectors_->getW()->square2(*v2_);
  tmpOut_->mul(*x2_, *v2_);
  outV->sumRows(*tmpOut_, -0.5, 1.0);

  /* activation */ {
    REGISTER_TIMER_INFO("FwAtvTimer", getName().c_str());
    forwardActivation();
  }
}

void FactorizationMachineLayer::backward(const UpdateCallback& callback) {
  /* Do derivation */ {
    REGISTER_TIMER_INFO("BpAvtTimer", getName().c_str());
    backwardActivation();
  }

  const MatrixPtr& inputV = getInputValue(0);
  const MatrixPtr& oGrad = getOutputGrad();

  MatrixPtr tmpSum =
      Matrix::create(1, latentVectors_->getW()->getHeight(), false, useGpu_);
  MatrixPtr tmpSum_T = Matrix::create(tmpSum->getRowBuf(0),
                                      latentVectors_->getW()->getHeight(),
                                      1,
                                      false,
                                      useGpu_);

  /* Calculate the gradients of the latentVectors_ matrix */
  if (latentVectors_->getWGrad()) {
    MatrixPtr tmpIn = inputV->clone(0, 0, useGpu_);
    if (dynamic_cast<CpuSparseMatrix*>(inputV.get())) {
      CpuSparseMatrix* inputV_s = dynamic_cast<CpuSparseMatrix*>(inputV.get());
      CpuSparseMatrix* x2_s = dynamic_cast<CpuSparseMatrix*>(x2_.get());
      CpuSparseMatrix* tmpIn_s = dynamic_cast<CpuSparseMatrix*>(tmpIn.get());
      tmpIn_s->copyFrom(*inputV_s);
      tmpIn_s->rowScale(0, *inputV_s, *oGrad);
      latentVectors_->getWGrad()->mul(*tmpIn->getTranspose(), *tmpMul_, 1, 1);
      tmpIn_s->rowScale(0, *x2_s, *oGrad);
    } else {
      tmpIn->rowScale(0, *inputV, *oGrad);
      latentVectors_->getWGrad()->mul(*tmpIn->getTranspose(), *tmpMul_, 1, 1);
      tmpIn->rowScale(0, *x2_, *oGrad);
    }

    tmpSum->sumCols(*tmpIn, -1, 0);
    latentVectors_->getWGrad()->addRowScale(
        0, *latentVectors_->getW(), *tmpSum_T);

    /* Increasing the number of gradient */
    latentVectors_->getParameterPtr()->incUpdate(callback);
  }

  /* Calculate the input layers gradient */
  MatrixPtr inGrad = getInputGrad(0);
  if (inGrad != NULL) {
    MatrixPtr latentVectors_T = latentVectors_->getW()->getTranspose();
    inGrad->mul(*tmpMul_, *latentVectors_T, 1, 1);
    tmpSum_T->sumRows(*v2_, -1, 0);
    inGrad->addColScale(0, *inputV, *tmpSum);
    inGrad->rowScale(0, *inGrad, *oGrad);
  }
}

}  // namespace paddle
