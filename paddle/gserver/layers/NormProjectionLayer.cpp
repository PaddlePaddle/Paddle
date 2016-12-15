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

#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"
#include "paddle/math/cross_map_normal_op.h"
#include "NormProjectionLayer.h"

namespace paddle {
size_t CMRProjectionNormLayer::getSize() {
  CHECK_EQ(inputLayers_.size(), 1UL);
  size_t layerSize = 0;
  imgSizeH_ = inputLayers_[0]->getOutput().getFrameHeight();
  imgSizeW_ = inputLayers_[0]->getOutput().getFrameWidth();
  if (imgSizeH_ == 0) {
    imgSizeH_ = imgSizeY_;
  }
  if (imgSizeW_ == 0) {
    imgSizeW_ = imgSize_;
  }
  outputH_ = imgSizeH_;
  outputW_ = imgSizeW_;
  layerSize = outputH_ * outputW_ * channels_;

  getOutput().setFrameHeight(outputH_);
  getOutput().setFrameWidth(outputW_);
  return layerSize;
}

bool CMRProjectionNormLayer::init(const LayerMap& layerMap,
                                  const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  ResponseNormLayer::init(layerMap, parameterMap);

  /* the size of inputs for norm-layer is 1 */
  CHECK_EQ(config_.inputs_size(), 1);

  if (useGpu_) {
    normal_ = FunctionBase::funcRegistrar_.createByType(
        FUNC_NAME(CrossMapNormal, GPU));
  } else {
    normal_ = FunctionBase::funcRegistrar_.createByType(
        FUNC_NAME(CrossMapNormal, CPU));
  }
  normal_->init(
      FuncConfig().set("size", size_).set("scale", scale_).set("pow", pow_));

  return true;
}

void CMRProjectionNormLayer::forward(PassType passType) {
  Layer::forward(passType);

  /* malloc memory for the output_ if necessary */
  /* note: one sample correspond to one row */
  MatrixPtr input = inputLayers_[0]->getOutputValue();
  int batchSize = input->getHeight();
  int size = getSize();
  resetOutput(batchSize, size);

  MatrixPtr outV = getOutputValue();

  Matrix::resizeOrCreate(denoms_, batchSize, size, /* trans */ false, useGpu_);

  Dims dims{(size_t)batchSize,
            (size_t)channels_,
            (size_t)imgSizeH_,
            (size_t)imgSizeW_};
  normal_->calc(
      {Tensor(input->getData(), dims)},
      {Tensor(outV->getData(), dims), Tensor(denoms_->getData(), dims)},
      {});
}

void CMRProjectionNormLayer::backward(const UpdateCallback& callback) {
  (void)callback;

  if (NULL == inputLayers_[0]->getOutputGrad()) {
    return;
  }
  /* Do derivation */
  MatrixPtr preOutGrad = inputLayers_[0]->getOutputGrad();
  MatrixPtr localGrad = getOutputGrad();
  MatrixPtr localOutV = getOutputValue();
  MatrixPtr preOutV = inputLayers_[0]->getOutputValue();

  if (useGpu_) {
    CrossMapNormalGrad<DEVICE_TYPE_GPU> crossGrad;
    crossGrad(dynamic_cast<GpuMatrix&>(*preOutGrad),
              dynamic_cast<GpuMatrix&>(*preOutV),
              dynamic_cast<GpuMatrix&>(*localGrad),
              dynamic_cast<GpuMatrix&>(*localOutV),
              dynamic_cast<GpuMatrix&>(*denoms_),
              channels_,
              imgSizeH_,
              imgSizeW_,
              size_,
              scale_,
              pow_);
  } else {
    CrossMapNormalGrad<DEVICE_TYPE_CPU> crossGrad;
    crossGrad(dynamic_cast<CpuMatrix&>(*preOutGrad),
              dynamic_cast<CpuMatrix&>(*preOutV),
              dynamic_cast<CpuMatrix&>(*localGrad),
              dynamic_cast<CpuMatrix&>(*localOutV),
              dynamic_cast<CpuMatrix&>(*denoms_),
              channels_,
              imgSizeH_,
              imgSizeW_,
              size_,
              scale_,
              pow_);
  }
}
}  // namespace paddle
