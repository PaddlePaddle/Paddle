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

#include "ProposalTargetLayer.h"
#include <vector>
#include "paddle/utils/ThreadLocal.h"

namespace paddle {

REGISTER_LAYER(proposal_target, ProposalTargetLayer);

bool ProposalTargetLayer::init(const LayerMap& layerMap,
                               const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);

  auto layerConf = config_.inputs(0).proposal_target_conf();
  posOverlapThreshold_ = layerConf.pos_overlap_threshold();
  negOverlapThreshold_ = layerConf.neg_overlap_threshold();
  boxBatchSize_ = layerConf.box_batch_size();
  boxFgRatio_ = layerConf.box_fg_ratio();
  CHECK_LT(boxFgRatio_, 1.);
  numClasses_ = layerConf.num_classes();
  backgroundId_ = layerConf.background_id();
  return true;
}

void ProposalTargetLayer::bboxOverlaps(
    const std::vector<std::vector<real>>& priorBBoxes,
    const std::vector<std::vector<real>>& gtBBoxes,
    std::vector<real>& overlaps) {
  for (size_t i = 0; i < priorBBoxes.size(); ++i) {
    for (size_t j = 0; j < gtBBoxes.size(); ++j) {
      real width = std::min(priorBBoxes[i][2], gtBBoxes[j][2]) -
                   std::max(priorBBoxes[i][0], gtBBoxes[j][0]) + 1;
      real height = std::min(priorBBoxes[i][3], gtBBoxes[j][3]) -
                    std::max(priorBBoxes[i][1], gtBBoxes[j][1]) + 1;
      if (width > 0 && height > 0) {
        real gtboxArea = (gtBBoxes[j][2] - gtBBoxes[j][0] + 1) *
                         (gtBBoxes[j][3] - gtBBoxes[j][1] + 1);
        real priorboxArea = (priorBBoxes[i][2] - priorBBoxes[i][0] + 1) *
                            (priorBBoxes[i][3] - priorBBoxes[i][1] + 1);
        real overlapArea = width * height;
        overlaps[i * gtBBoxes.size() + j] =
            overlapArea / (gtboxArea + priorboxArea - overlapArea);
      }
    }
  }
}

std::pair<size_t, size_t> ProposalTargetLayer::labelBBoxes(
    const std::vector<std::vector<real>>& priorBBoxes,
    const std::vector<std::vector<real>>& gtBBoxes,
    const std::vector<real>& overlaps,
    const real posOverlapThreshold,
    const real negOverlapThreshold,
    std::vector<int>& matchIndices,
    std::vector<int>& labels) {
  size_t numPos = 0;
  size_t numNeg = 0;
  std::vector<int> gtBBoxMaxIdxs(gtBBoxes.size(), -1);
  for (size_t n = 0; n < overlaps.size(); ++n) {
    size_t priorBBoxIdx = n / gtBBoxes.size();
    size_t gtBBoxIdx = n % gtBBoxes.size();
    if (matchIndices[priorBBoxIdx] == -1 ||
        overlaps[n] > overlaps[priorBBoxIdx * gtBBoxes.size() +
                               matchIndices[priorBBoxIdx]]) {
      matchIndices[priorBBoxIdx] = gtBBoxIdx;  // overlaps.argmax(axis=1)
    }
  }

  for (size_t n = 0; n < priorBBoxes.size();
       ++n) {  // fg/bg label: above/below threshold IOU
    if (overlaps[n * gtBBoxes.size() + matchIndices[n]] >=
        posOverlapThreshold) {
      labels[n] = 1;
      ++numPos;
    } else if (overlaps[n * gtBBoxes.size() + matchIndices[n]] <=
               negOverlapThreshold) {
      labels[n] = 0;
      ++numNeg;
    } else {
      labels[n] = -1;
    }
    if (priorBBoxes[n][4] == -1) {  // disabled label from prior data
      labels[n] = -1;
    }
  }
  return std::make_pair(numPos, numNeg);
}

template <typename T>
void ProposalTargetLayer::sampleBBoxes(
    std::vector<T>& allLabels, T label, T disabledLable, size_t m, size_t n) {
  auto& randEngine = ThreadLocalRandomEngine::get();
  for (size_t i = 0; i < allLabels.size(); ++i) {
    if (allLabels[i] == label) {
      if (rand_(randEngine) * n < m) {
        --m;
      } else {
        allLabels[i] = disabledLable;
      }
      --n;
    }
  }
}

std::pair<size_t, size_t> ProposalTargetLayer::generateMatchIndices(
    const Matrix& priorValue,
    const Matrix& gtValue,
    const int* gtStartPosPtr,
    const size_t gtBBoxDim,
    const size_t seqNum,
    const real posOverlapThreshold,
    const real negOverlapThreshold,
    const size_t boxBatchSize,
    const real boxFgRatio,
    std::vector<std::vector<int>>* priorBBoxIdxsVecPtr,
    std::vector<std::vector<int>>* matchIndicesVecPtr) {
  size_t batchSize = seqNum;
  size_t totalPos = 0;
  size_t totalNeg = 0;
  std::vector<real> allLabels;
  std::vector<real> allTargets;
  const real* priorData = priorValue.getData();
  const size_t priorBBoxDim = priorValue.getWidth();

  priorBBoxIdxsVecPtr->resize(batchSize);
  for (size_t i = 0; i < priorValue.getHeight(); ++i) {
    size_t batchIdx = *(priorData + i * priorBBoxDim + 5);
    CHECK_LT(batchIdx, seqNum);
    (*priorBBoxIdxsVecPtr)[batchIdx].push_back(i);
  }

  for (size_t n = 0; n < batchSize; ++n) {
    std::vector<std::vector<real>> priorBBoxes;
    for (size_t i = 0; i < (*priorBBoxIdxsVecPtr)[n].size(); ++i) {
      int priorBBoxIdx = (*priorBBoxIdxsVecPtr)[n][i];
      std::vector<real> priorBBox;
      priorBBox.push_back(*(priorData + priorBBoxIdx * priorBBoxDim + 0));
      priorBBox.push_back(*(priorData + priorBBoxIdx * priorBBoxDim + 1));
      priorBBox.push_back(*(priorData + priorBBoxIdx * priorBBoxDim + 2));
      priorBBox.push_back(*(priorData + priorBBoxIdx * priorBBoxDim + 3));
      priorBBox.push_back(*(priorData + priorBBoxIdx * priorBBoxDim +
                            4));  // disabled box label
      priorBBoxes.push_back(priorBBox);
    }
    std::vector<int> matchIndices(priorBBoxes.size(), -1);
    size_t numGTBBoxes = gtStartPosPtr[n + 1] - gtStartPosPtr[n];
    if (!numGTBBoxes) {
      matchIndicesVecPtr->push_back(matchIndices);
      continue;
    }
    std::vector<std::vector<real>> gtBBoxes;
    auto startPos = gtValue.getData() + gtStartPosPtr[n] * gtBBoxDim;
    for (size_t i = 0; i < numGTBBoxes; ++i) {
      std::vector<real> gtBBox;
      gtBBox.push_back(*(startPos + i * gtBBoxDim + 0));
      gtBBox.push_back(*(startPos + i * gtBBoxDim + 1));
      gtBBox.push_back(*(startPos + i * gtBBoxDim + 2));
      gtBBox.push_back(*(startPos + i * gtBBoxDim + 3));
      gtBBoxes.push_back(gtBBox);
    }

    std::vector<real> overlaps(priorBBoxes.size() * gtBBoxes.size(), -1);
    bboxOverlaps(
        priorBBoxes,
        gtBBoxes,
        overlaps);  // calculate the overlaps of priorboxes and gtBBoxes

    std::vector<int> labels(priorBBoxes.size(),
                            -1);  // init with -1 to label disabled priorboxes
    std::pair<size_t, size_t> numLabels =
        labelBBoxes(priorBBoxes,
                    gtBBoxes,
                    overlaps,
                    posOverlapThreshold,
                    negOverlapThreshold,
                    matchIndices,
                    labels);  // lable the priorboxes
    totalPos += numLabels.first;
    totalNeg += numLabels.second;
    matchIndicesVecPtr->push_back(matchIndices);
    std::copy(labels.begin(), labels.end(), std::back_inserter(allLabels));
  }

  size_t numPos = boxBatchSize * boxFgRatio;
  if (totalPos > numPos) {  // subsample positive labels if we have too many
    sampleBBoxes<real>(allLabels, 1, -1, numPos, totalPos);
  } else {
    numPos = totalPos;
  }
  size_t numNeg = boxBatchSize - numPos;
  if (totalNeg > numNeg) {  // subsample negative labels if we have too many
    sampleBBoxes<real>(allLabels, 0, -1, numNeg, totalNeg);
  } else {
    numNeg = totalNeg;
  }
  size_t idx = 0;
  for (size_t n = 0; n < batchSize; ++n) {
    for (size_t i = 0; i < (*matchIndicesVecPtr)[n].size(); ++i) {
      if (allLabels[idx] != 1) {
        (*matchIndicesVecPtr)[n][i] =
            allLabels[idx++]--;  // -1 for bg, -2 for disabled
      }
    }
  }
  return std::make_pair(numPos, numNeg);
}

void ProposalTargetLayer::encodeTarget(const std::vector<real>& priorBBox,
                                       const std::vector<real>& gtBBox,
                                       std::vector<real>& target) {
  real priorBBoxWidth = priorBBox[2] - priorBBox[0] + 1;
  real priorBBoxHeight = priorBBox[3] - priorBBox[1] + 1;
  real priorBBoxCenterX = (priorBBox[2] + priorBBox[0]) / 2;
  real priorBBoxCenterY = (priorBBox[3] + priorBBox[1]) / 2;

  real gtBBoxWidth = gtBBox[2] - gtBBox[0] + 1;
  real gtBBoxHeight = gtBBox[3] - gtBBox[1] + 1;
  real gtBBoxCenterX = (gtBBox[2] + gtBBox[0]) / 2;
  real gtBBoxCenterY = (gtBBox[3] + gtBBox[1]) / 2;

  target[0] = (gtBBoxCenterX - priorBBoxCenterX) / priorBBoxWidth;
  target[1] = (gtBBoxCenterY - priorBBoxCenterY) / priorBBoxHeight;
  target[2] = std::log(gtBBoxWidth / priorBBoxWidth);
  target[3] = std::log(gtBBoxHeight / priorBBoxHeight);
}

void ProposalTargetLayer::forward(PassType passType) {
  Layer::forward(passType);
  // priorValue layout:
  // | xmin1 | ymin1 | xmax1 | ymax1 | disableFlag1 | batchIdx1 |
  // | xmin2 | ......
  MatrixPtr priorValue = getInputValue(0);
  // labelValue layout(sequence data):
  // | xmin1_1 | ymin1_1 | xmax1_1 | ymax1_1  | class1 | ......
  MatrixPtr labelValue = getInputValue(1);

  if (useGpu_) {
    MatrixPtr priorCpuBuffer;
    Matrix::resizeOrCreate(priorCpuBuffer,
                           priorValue->getHeight(),
                           priorValue->getWidth(),
                           false,
                           false);
    MatrixPtr labelCpuBuffer;
    Matrix::resizeOrCreate(labelCpuBuffer,
                           labelValue->getHeight(),
                           labelValue->getWidth(),
                           false,
                           false);
    priorCpuBuffer->copyFrom(*priorValue);
    labelCpuBuffer->copyFrom(*labelValue);
    priorValue = priorCpuBuffer;
    labelValue = labelCpuBuffer;
  }

  // Match prior bbox to groundtruth bbox
  Argument label = getInput(1);
  const int* labelIndex = label.sequenceStartPositions->getData(false);
  const size_t gtBBoxDim = 5;
  const size_t seqNum = label.getNumSequences();
  allPriorBBoxIdxs_.clear();
  allMatchIndices_.clear();
  std::pair<size_t, size_t> retPair = generateMatchIndices(*priorValue,
                                                           *labelValue,
                                                           labelIndex,
                                                           gtBBoxDim,
                                                           seqNum,
                                                           posOverlapThreshold_,
                                                           negOverlapThreshold_,
                                                           boxBatchSize_,
                                                           boxFgRatio_,
                                                           &allPriorBBoxIdxs_,
                                                           &allMatchIndices_);
  size_t numROIs = retPair.first + retPair.second;

  resetOutput(numROIs, 10);
  MatrixPtr outputValue = getOutputValue();
  if (useGpu_) {
    MatrixPtr outputCpuBuffer;
    Matrix::resizeOrCreate(outputCpuBuffer,
                           outputValue->getHeight(),
                           outputValue->getWidth(),
                           false,
                           false);
    outputValue = outputCpuBuffer;
  }

  // | batchIdx1 | xmin1 | ymin1 | xmax1 | ymax1 | class1 | target1_1 |
  // target2_1 | target3_1 | target4_1 | | batchIdx2 | ......
  real* outData = outputValue->getData();
  for (size_t n = 0; n < seqNum; ++n) {
    for (size_t i = 0; i < allMatchIndices_[n].size(); ++i) {
      if (allMatchIndices_[n][i] == -2) continue;  // disabled priorbox
      const int priorBBoxIdx = allPriorBBoxIdxs_[n][i];
      const int gtBBoxIdx = allMatchIndices_[n][i];
      auto* priorOffset =
          priorValue->getData() + priorBBoxIdx * priorValue->getWidth();
      // ROIs' data for ROIPooling
      *(outData++) = n;
      *(outData++) = *(priorOffset + 0);
      *(outData++) = *(priorOffset + 1);
      *(outData++) = *(priorOffset + 2);
      *(outData++) = *(priorOffset + 3);
      if (gtBBoxIdx == -1) {  // bg priorbox
        // class data for classfication loss
        *(outData++) = backgroundId_;
        // trivial target data for bg priorbox
        *(outData++) = 0.;
        *(outData++) = 0.;
        *(outData++) = 0.;
        *(outData++) = 0.;
      } else {  // fg priorbox
        std::vector<real> priorBBox{
            *(priorOffset + 0),
            *(priorOffset + 1),
            *(priorOffset + 2),
            *(priorOffset + 3),
        };
        auto* labelOffset =
            labelValue->getData() + (labelIndex[n] + gtBBoxIdx) * gtBBoxDim;
        // class data for classfication loss
        *(outData++) = *(labelOffset + 4);
        std::vector<real> gtBBox{
            *(labelOffset + 0),
            *(labelOffset + 1),
            *(labelOffset + 2),
            *(labelOffset + 3),
        };
        std::vector<real> gtEncode(4);
        encodeTarget(priorBBox, gtBBox, gtEncode);
        // target data for bbox regression loss
        *(outData++) = gtEncode[0];
        *(outData++) = gtEncode[1];
        *(outData++) = gtEncode[2];
        *(outData++) = gtEncode[3];
      }
    }
  }

  if (useGpu_) {
    getOutputValue()->copyFrom(*outputValue);
  }
}

}  // namespace paddle
