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

#include "Layer.h"
#include "paddle/math/Matrix.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

namespace paddle {

/**
 * @brief A layer for circular convluation of two vectors,
 * which is used in NEURAL TURING MACHINE.
 * - Input: two vectors, the first is data (batchSize x dataDim)
 * the second is shift weights (batchSize x shiftDim)
 * - Output: a vector (batchSize x dataDim)
 * Assumed that:
 * - a[in]: contains M elements.
 * - b[in]: contains N elements (N should be odd).
 * - c[out]: contains M elements.
 *
 * \f[
 *     c[i] = \sum_{j=-(N-1)/2}^{(N-1)/2}a_{i+j} * b_{j}
 * \f]
 *
 * In this formula:
 *  - a's index is computed modulo M.
 *  - b's index is comupted modulo N.
 *
 * The config file api is conv_shift_layer.
 */

class ConvShiftLayer : public Layer {
public:
  explicit ConvShiftLayer(const LayerConfig& config) : Layer(config) {}

  ~ConvShiftLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
  bool isSeqType();
  void circularConvSeq();
  void circularConvSeqDerivative();
};

REGISTER_LAYER(conv_shift, ConvShiftLayer);

bool ConvShiftLayer::init(const LayerMap& layerMap,
                          const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  CHECK_EQ(inputLayers_.size(), 2U);

  return true;
}

bool ConvShiftLayer::isSeqType() {
  const Argument& inLayer0 = getInput(0);
  if (nullptr == inLayer0.sequenceStartPositions)
    return false;
  else
    return true;
}

void ConvShiftLayer::circularConvSeq() {
  const Argument& inLayer0 = getInput(0);
  MatrixPtr in0 = inLayer0.value;
  MatrixPtr in1 = getInputValue(1);
  MatrixPtr out = getOutputValue();
  const ICpuGpuVectorPtr& sequenceStartPositions =
      inLayer0.sequenceStartPositions;

  size_t width0 = in0->getWidth();
  size_t numSeqs = sequenceStartPositions->getSize() - 1;
  size_t height0 = in0->getHeight();
  size_t width1 = in1->getWidth();
  size_t height1 = in1->getHeight();

  CHECK_EQ(numSeqs, height1);
  CHECK_EQ(width0, out->getWidth());
  CHECK_EQ(height0, out->getHeight());

  CHECK_EQ(width1 % 2, 1U);

  real* inV0 = in0->getData();
  const int* startPosIntPtr = sequenceStartPositions->getData(false);
  real* inV1 = in1->getData();
  real* outV = out->getData();

  int leftCtxLen = (width1 - 1) / 2;
  for (size_t x = 0; x < numSeqs - 1; x++) {
    int curSeqLen = startPosIntPtr[x + 1];
    size_t curSeqWidth = curSeqLen * width0;
    for (size_t i = 0; i < curSeqWidth; i++) {
      for (size_t j = 0; j < width1; ++j) {
        int index = i + j - leftCtxLen;
        index = (index + curSeqWidth) % curSeqWidth;
        int outVRowOffset = i / width0;
        int outVColOffset = i % width0;
        int inV0RowOffset = index / width0;
        int inV0ColOffset = index % width0;
        (outV + outVRowOffset)[outVColOffset] +=
            (inV0 + inV0RowOffset)[inV0ColOffset] * inV1[j];
      }
    }
    outV += curSeqWidth;
    inV0 += curSeqWidth;
    inV1 += width1;
  }
}

void ConvShiftLayer::circularConvSeqDerivative() {
  const Argument& inLayer0 = getInput(0);
  MatrixPtr in0 = inLayer0.value;
  MatrixPtr in1 = getInputValue(1);
  MatrixPtr inG0 = getInputGrad(0);
  MatrixPtr inG1 = getInputGrad(1);
  MatrixPtr outG = getOutputGrad();
  const ICpuGpuVectorPtr& sequenceStartPositions =
      inLayer0.sequenceStartPositions;

  size_t height0 = in0->getHeight();
  size_t height1 = in1->getHeight();
  size_t numSeqs = sequenceStartPositions->getSize() - 1;
  size_t width0 = in0->getWidth();
  size_t width1 = in1->getWidth();

  CHECK_EQ(height1, numSeqs);
  CHECK_EQ(height0, inG0->getHeight());
  CHECK_EQ(width0, inG0->getWidth());
  CHECK_EQ(height1, inG1->getHeight());
  CHECK_EQ(width1, inG1->getWidth());
  CHECK_EQ(height0, outG->getHeight());
  CHECK_EQ(width0, outG->getWidth());

  const int* startPosIntPtr = sequenceStartPositions->getData(false);
  real* outGV = outG->getData();
  real* inV0 = in0->getData();
  real* inV1 = in1->getData();
  real* inGV0 = inG0->getData();
  real* inGV1 = inG1->getData();

  int leftCtxLen = (width1 - 1) / 2;
  for (size_t x = 0; x < numSeqs - 1; x++) {
    int curSeqLen = startPosIntPtr[x + 1];
    size_t curSeqWidth = curSeqLen * width0;
    for (size_t j = 0; j < width1; j++) {
      for (size_t i = 0; i < curSeqWidth; i++) {
        int index = i + j - leftCtxLen;
        index = (index + curSeqWidth) % curSeqWidth;
        int inGV0RowOffset = index / width0;
        int inGV0ColOffset = index % width0;
        int outGVRowOffset = i / width0;
        int outGVColOffset = i % width0;
        (inGV0 + inGV0RowOffset)[inGV0ColOffset] +=
            (outGV + outGVRowOffset)[outGVColOffset] * inV1[j];
        inGV1[j] += (outGV + outGVRowOffset)[outGVColOffset] *
                    (inGV0 + inGV0RowOffset)[inGV0ColOffset];
      }
    }
    outGV += curSeqWidth;
    inV0 += curSeqWidth;
    inV1 += width1;
    inGV0 += curSeqWidth;
    inGV1 += width1;
  }
}

void ConvShiftLayer::forward(PassType passType) {
  Layer::forward(passType);

  MatrixPtr inV0 = getInputValue(0);

  size_t batchSize = inV0->getHeight();
  size_t dataDim = inV0->getWidth();

  CHECK_EQ(dataDim, getSize());

  {
    REGISTER_TIMER_INFO("FwResetTimer", getName().c_str());
    resetOutput(batchSize, dataDim);
  }

  REGISTER_TIMER_INFO("FwConvShiftTimer", getName().c_str());
  if (!isSeqType()) {
    MatrixPtr inV1 = getInputValue(1);
    CHECK_EQ(batchSize, inV1->getHeight());
    MatrixPtr outV = getOutputValue();
    outV->circularConv(*inV0, *inV1);
  } else {
    circularConvSeq();
  }
}

void ConvShiftLayer::backward(const UpdateCallback& callback) {
  MatrixPtr inG0 = getInputGrad(0);
  MatrixPtr inG1 = getInputGrad(1);

  REGISTER_TIMER_INFO("BwConvShiftTimer", getName().c_str());

  if (!(inG0 && inG1)) {
    CHECK(!inG0 || !inG1) << "Not supported";
  }

  if (!isSeqType()) {
    MatrixPtr inV0 = getInputValue(0);
    MatrixPtr inV1 = getInputValue(1);
    MatrixPtr outG = getOutputGrad();
    outG->circularConvDerivative(*outG, *inV0, *inV1, *inG0, *inG1);
  } else {
    circularConvSeqDerivative();
  }
}

}  // namespace paddle
