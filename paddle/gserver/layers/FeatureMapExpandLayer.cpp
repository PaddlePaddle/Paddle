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
#include "paddle/utils/Stat.h"

namespace paddle {

/**
 * @brief A layer for expanding a batch of images to feature maps.
 * Each data of the input is a 2 dimensional matrix. Each element of the matrix
 * is replicated num_filters times to create a feature map with num_filters
 * channels.
 * - Input: Input one should be dense image data.
 * - Output: expanded fature maps.
 * \f[
 *  y.row[i] = x.row[i \mod x.width], i = 0,1,..., (x.width * num\_filters - 1)
 * \f]
 * For example, num_filters = 4:
 * @code
 *   x = [a1,a2;
 *        b1,b2]
 *   y = [a1, a2, a1, a2, a1, a2, a1, a2;
 *        b1, b2, b1, b2, b1, b2, b1, b2;]
 * @endcode
 */

class FeatureMapExpandLayer : public Layer {
 private:
  int numFilters_;
  bool asRowVector_;

 public:
  explicit FeatureMapExpandLayer(const LayerConfig& config) : Layer(config) {}

  ~FeatureMapExpandLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};

REGISTER_LAYER(featmap_expand, FeatureMapExpandLayer);

bool FeatureMapExpandLayer::init(const LayerMap& layerMap,
                                 const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  CHECK_EQ(inputLayers_.size(), 1UL);
  numFilters_ = config_.num_filters();
  asRowVector_ = config_.user_arg() != "as_col_vec";
  return true;
}

void FeatureMapExpandLayer::forward(PassType passType) {
  Layer::forward(passType);
  MatrixPtr inputV = getInputValue(0);
  size_t batchSize = getInput(0).getBatchSize();
  int imgSize = inputV->getWidth();
  resetOutput(batchSize, imgSize * numFilters_);

  MatrixPtr outputV = getOutputValue();

  {
    AsyncGpuBlock asyncGpuBlock;
    if (asRowVector_) {
      for (size_t i = 0; i < batchSize; i++) {
        MatrixPtr outVTmp =
            Matrix::create(outputV->getData() + i * imgSize * numFilters_,
                           numFilters_,
                           imgSize,
                           false,
                           useGpu_);
        MatrixPtr inVTmp = Matrix::create(
            inputV->getData() + i * imgSize, 1, imgSize, false, useGpu_);
        outVTmp->addRowVector(*inVTmp);
      }
    } else {
      for (size_t i = 0; i < batchSize; i++) {
        MatrixPtr outVTmp =
            Matrix::create(outputV->getData() + i * imgSize * numFilters_,
                           imgSize,
                           numFilters_,
                           false,
                           useGpu_);
        MatrixPtr inVTmp = Matrix::create(
            inputV->getData() + i * imgSize, imgSize, 1, false, useGpu_);
        outVTmp->addColVector(*inVTmp);
      }
    }
  }
  /* activation */ {
    REGISTER_TIMER_INFO("FwAtvTimer", getName().c_str());
    forwardActivation();
  }
}

void FeatureMapExpandLayer::backward(const UpdateCallback& callback) {
  MatrixPtr inGrad = getInputGrad(0);
  if (NULL == inGrad) {
    return;
  }
  MatrixPtr outGrad = getOutputGrad();
  size_t batchSize = getInput(0).getBatchSize();
  int imgSize = inGrad->getWidth();
  /* Do activation */ {
    REGISTER_TIMER_INFO("BpAvtTimer", getName().c_str());
    backwardActivation();
  }
  {
    AsyncGpuBlock asyncGpuBlock;
    if (asRowVector_) {
      for (size_t i = 0; i < batchSize; i++) {
        MatrixPtr outGradTmp =
            Matrix::create(outGrad->getData() + i * imgSize * numFilters_,
                           numFilters_,
                           imgSize,
                           false,
                           useGpu_);
        MatrixPtr inGradTmp = Matrix::create(
            inGrad->getData() + i * imgSize, 1, imgSize, false, useGpu_);
        inGradTmp->collectBias(*outGradTmp, 1);
      }
    } else {
      for (size_t i = 0; i < batchSize; i++) {
        MatrixPtr outGradTmp =
            Matrix::create(outGrad->getData() + i * imgSize * numFilters_,
                           imgSize,
                           numFilters_,
                           false,
                           useGpu_);
        MatrixPtr inGradTmp = Matrix::create(
            inGrad->getData() + i * imgSize, imgSize, 1, false, useGpu_);
        inGradTmp->sumRows(*outGradTmp, 1, 1);
      }
    }
  }
}

}  // namespace paddle.
