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

#include "DataNormLayer.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_LAYER(data_norm, DataNormLayer);

bool DataNormLayer::init(const LayerMap& layerMap,
                         const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  /* initialize the weight */
  CHECK(!biasParameter_) << "DataNormLayer does not need bias";
  CHECK(inputLayers_.size() == 1 && inputLayers_[0]->getType() == "data")
      << "DataNormLayer accepts one and only one DataLayer as its input layer";
  CHECK_EQ(inputLayers_.size(), parameters_.size());
  CHECK_EQ(inputLayers_[0]->getSize(), getSize());
  CHECK_EQ(parameters_[0]->getSize(), 5 * getSize());
  CHECK(parameters_[0]->isStatic())
      << "The parameter of DataNormLayer must be static";

  weight_ = std::unique_ptr<Weight>(new Weight(5, getSize(), parameters_[0]));
  min_ = Matrix::create(
      nullptr, /* height= */ 1, getSize(), /* trans= */ false, useGpu_);
  rangeReciprocal_ = Matrix::create(nullptr,
                                    /* height= */ 1,
                                    getSize(),
                                    /* trans= */ false,
                                    useGpu_);
  mean_ = Matrix::create(nullptr,
                         /* height= */ 1,
                         getSize(),
                         /* trans= */ false,
                         useGpu_);
  stdReciprocal_ = Matrix::create(nullptr,
                                  /* height= */ 1,
                                  getSize(),
                                  /* trans= */ false,
                                  useGpu_);
  decimalReciprocal_ = Matrix::create(nullptr,
                                      /* height= */ 1,
                                      getSize(),
                                      /* trans= */ false,
                                      useGpu_);

  min_->setData(weight_->getW()->getData());
  rangeReciprocal_->setData(weight_->getW()->getData() + getSize());
  mean_->setData(weight_->getW()->getData() + 2 * getSize());
  stdReciprocal_->setData(weight_->getW()->getData() + 3 * getSize());
  decimalReciprocal_->setData(weight_->getW()->getData() + 4 * getSize());

  /* normalization strategy */
  if (config_.data_norm_strategy() == "z-score") {
    mode_ = kZScore;
  } else if (config_.data_norm_strategy() == "min-max") {
    mode_ = kMinMax;
  } else if (config_.data_norm_strategy() == "decimal-scaling") {
    mode_ = kDecimalScaling;
  } else {
    LOG(FATAL) << "Unknown data normalization strategy: "
               << config_.data_norm_strategy();
  }

  return true;
}

void DataNormLayer::forward(PassType passType) {
  Layer::forward(passType);

  /* malloc memory for the output_ if necessary */
  int batchSize = getInput(0).getBatchSize();
  int size = getSize();
  reserveOutput(batchSize, size);

  const MatrixPtr inValue = getInputValue(0);
  MatrixPtr outValue = getOutputValue();
  outValue->copyFrom(*inValue);
  switch (mode_) {
    case kZScore: {
      outValue->addBias(*mean_, -1.0);
      outValue->colScale(0, *outValue, *stdReciprocal_);
      break;
    }
    case kMinMax: {
      outValue->addBias(*min_, -1.0);
      outValue->colScale(0, *outValue, *rangeReciprocal_);
      break;
    }
    case kDecimalScaling: {
      outValue->colScale(0, *outValue, *decimalReciprocal_);
      break;
    }
    default:
      LOG(FATAL) << "should not reach here";
  }
}

void DataNormLayer::backward(const UpdateCallback& callback) {
  // The parameter for DataNormLayer is static, and does not need to be updated
  (void)callback;

  /* Calculate the input layers error */
  const MatrixPtr& outGrad = getOutputGrad();
  MatrixPtr inGrad = getInputGrad(0);
  if (inGrad) {
    switch (mode_) {
      case kZScore: {
        inGrad->addColScale(0, *outGrad, *stdReciprocal_);
        break;
      }
      case kMinMax: {
        inGrad->addColScale(0, *outGrad, *rangeReciprocal_);
        break;
      }
      case kDecimalScaling: {
        inGrad->addColScale(0, *outGrad, *decimalReciprocal_);
        break;
      }
      default: { LOG(FATAL) << "should not reach here"; }
    }
  }
}

}  // namespace paddle
