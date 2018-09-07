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
 *@brief This layer multiplex multiple layers according to the index,
 * which is provided by the first input layer.
 * - Input[0]: the index of the layer to output of size batchSize.
 * - Input[1:N]; the candidate output data.
 * For each index i from 0 to batchSize -1, the output is the i-th row of the
 * (index[i] + 1)-th layer.
 *
 * For each i-th row of output:
 *
 * \f[
 *   y[i][j] = x_{x_{0}[i] + 1}[i][j], j = 0,1, ... , (x_{1}.width - 1)
 * \f]
 * where, y is output. \f$x_{k}\f$ is the k-th input layer and
 * \f$k = x_{0}[i] + 1\f$.
 */

class MultiplexLayer : public Layer {
 protected:
  /**
   * @brief A struct is used to save the copy information, includes input
   * layer index and copy size.
   */
  struct CopyInfo {
    CopyInfo(int inStartIdx, int inLength, int inCopyIdx)
        : startIdx(inStartIdx), length(inLength), copyIdx(inCopyIdx) {}

    /// The start row of input.
    int startIdx;
    /// Number of rows. If the layer index in Input[0] is not consecutive,
    /// the length is one. Otherwise, the length is > 1 and copy multi rows
    /// once.
    int length;
    /// The copied layer index, which needs to add 1.
    int copyIdx;
  };

  /// A list of CopyInfo used to save copy information.
  std::vector<CopyInfo> copySchedule_;

  /// Temporary matrix pointer to point to input data.
  MatrixPtr tmpSrc_;
  /// Temporary matrix pointer to point to output data.
  MatrixPtr tmpDest_;

 public:
  explicit MultiplexLayer(const LayerConfig& config) : Layer(config) {}

  ~MultiplexLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;

 private:
  /**
   * @brief Calculate copy info for input layers.
   */
  void calculateCopySchedule(const IVectorPtr& copyIds, size_t numIns);
};

REGISTER_LAYER(multiplex, MultiplexLayer);

void MultiplexLayer::calculateCopySchedule(const IVectorPtr& copyIds,
                                           size_t numIns) {
  copySchedule_.clear();
  CopyInfo prevCopyInfo(0, 0, -1);
  for (size_t i = 0; i < copyIds->getSize(); i++) {
    int copyId = copyIds->getElement(i);
    CHECK_GE(copyId, 0);
    CHECK_LT(copyId, int(numIns));
    // copy same input layer with prevous and will copy consecutive.
    if (copyId == prevCopyInfo.copyIdx) {
      ++prevCopyInfo.length;
    } else {
      if (prevCopyInfo.copyIdx != -1) {
        copySchedule_.emplace_back(prevCopyInfo);
      }
      prevCopyInfo.startIdx = i;
      prevCopyInfo.length = 1;
      prevCopyInfo.copyIdx = copyId;
    }
  }
  if (prevCopyInfo.copyIdx != -1) {
    copySchedule_.emplace_back(prevCopyInfo);
  }
}

bool MultiplexLayer::init(const LayerMap& layerMap,
                          const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);

  CHECK_GE(inputLayers_.size(), 2U);

  tmpSrc_ =
      Matrix::create(nullptr, /* height= */ 1, 1, /* trans= */ false, useGpu_);
  tmpDest_ =
      Matrix::create(nullptr, /* height= */ 1, 1, /* trans= */ false, useGpu_);
  return true;
}

void MultiplexLayer::forward(PassType passType) {
  Layer::forward(passType);

  IVectorPtr copyIds = getInput(0).ids;
  MatrixPtr inV1 = getInputValue(1);
  CHECK_EQ(copyIds->getSize(), inV1->getHeight());
  for (size_t i = 2; i < inputLayers_.size(); i++) {
    CHECK_EQ(inV1->getHeight(), getInputValue(i)->getHeight());
    CHECK_EQ(inV1->getWidth(), getInputValue(i)->getWidth());
  }

  calculateCopySchedule(copyIds, inputLayers_.size() - 1);
  {
    REGISTER_TIMER_INFO("FwResetTimer", getName().c_str());
    reserveOutput(inV1->getHeight(), inV1->getWidth());
  }

  MatrixPtr outV = getOutputValue();
  {
    REGISTER_TIMER_INFO("FwLMultplexingTimer", getName().c_str());
    AsyncGpuBlock block;
    for (const CopyInfo& info : copySchedule_) {
      outV->subMatrix(info.startIdx, info.length, tmpDest_)
          ->copyFrom(*getInputValue(info.copyIdx + 1)
                          ->subMatrix(info.startIdx, info.length, tmpSrc_));
    }
  }

  /* activation */ {
    REGISTER_TIMER_INFO("FwAtvTimer", getName().c_str());
    forwardActivation();
  }
}

void MultiplexLayer::backward(const UpdateCallback& callback) {
  /* Do derivation */ {
    REGISTER_TIMER_INFO("BpAvtTimer", getName().c_str());
    backwardActivation();
  }

  MatrixPtr outG = getOutputGrad();

  {
    REGISTER_TIMER_INFO("BwLMultiplexTimer", getName().c_str());
    AsyncGpuBlock block;
    for (const CopyInfo& info : copySchedule_) {
      if (getInputGrad(info.copyIdx + 1)) {
        getInputGrad(info.copyIdx + 1)
            ->subMatrix(info.startIdx, info.length, tmpDest_)
            ->add(*outG->subMatrix(info.startIdx, info.length, tmpSrc_));
      }
    }
  }
}

}  // namespace paddle
