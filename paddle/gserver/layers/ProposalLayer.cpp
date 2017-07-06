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

#include "ProposalLayer.h"

namespace paddle {

REGISTER_LAYER(proposal, ProposalLayer);

bool ProposalLayer::init(const LayerMap& layerMap,
                         const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);
  auto& layerConf = config_.inputs(0).proposal_conf();
  nmsThreshold_ = layerConf.nms_threshold();
  confidenceThreshold_ = layerConf.confidence_threshold();
  nmsTopK_ = layerConf.nms_top_k();
  keepTopK_ = layerConf.keep_top_k();
  minWidth_ = layerConf.min_width();
  minHeight_ = layerConf.min_height();
  numClasses_ = 2;
  inputNum_ = 1;
  backgroundId_ = 0;
  return true;
}

real ProposalLayer::jaccardOverlap(const UnnormalizedBBox& bbox1,
                                   const UnnormalizedBBox& bbox2) {
  if (bbox2.xMin > bbox1.xMax || bbox2.xMax < bbox1.xMin ||
      bbox2.yMin > bbox1.yMax || bbox2.yMax < bbox1.yMin) {
    return 0.0;
  } else {
    real interXMin = std::max(bbox1.xMin, bbox2.xMin);
    real interYMin = std::max(bbox1.yMin, bbox2.yMin);
    real interXMax = std::min(bbox1.xMax, bbox2.xMax);
    real interYMax = std::min(bbox1.yMax, bbox2.yMax);

    real interWidth = interXMax - interXMin + 1;
    real interHeight = interYMax - interYMin + 1;
    real interArea = interWidth * interHeight;

    real bboxArea1 = bbox1.getArea();
    real bboxArea2 = bbox2.getArea();

    return interArea / (bboxArea1 + bboxArea2 - interArea);
  }
}

void ProposalLayer::applyNMSFast(const vector<UnnormalizedBBox>& bboxes,
                                 const real* confScoreData,
                                 size_t classIdx,
                                 size_t topK,
                                 real confThreshold,
                                 real nmsThreshold,
                                 real minWidth,
                                 real minHeight,
                                 size_t numPriorBBoxes,
                                 size_t numClasses,
                                 vector<size_t>* indices) {
  vector<pair<real, size_t>> scores;
  for (size_t i = 0; i < numPriorBBoxes; ++i) {
    if (bboxes[i].getWidth() < minWidth || bboxes[i].getHeight() < minHeight) {
      continue;  // remove predicted boxes with either height or width <
                 // threshold
    }
    size_t confOffset = i * numClasses + classIdx;
    if (confScoreData[confOffset] > confThreshold)
      scores.push_back(std::make_pair(confScoreData[confOffset], i));
  }
  std::stable_sort(scores.begin(), scores.end(), sortScorePairDescend<size_t>);
  if (topK > 0 && topK < scores.size()) scores.resize(topK);
  while (scores.size() > 0) {
    const size_t idx = scores.front().second;
    bool keep = true;
    for (size_t i = 0; i < indices->size(); ++i) {
      if (keep) {
        const size_t savedIdx = (*indices)[i];
        real overlap = jaccardOverlap(bboxes[idx], bboxes[savedIdx]);
        keep = overlap <= nmsThreshold;
      } else {
        break;
      }
    }
    if (keep) indices->push_back(idx);
    scores.erase(scores.begin());
  }
}

size_t ProposalLayer::getDetectionIndices(
    const real* confData,
    const size_t numPriorBBoxes,
    const size_t numClasses,
    const size_t backgroundId,
    const size_t batchSize,
    const size_t confThreshold,
    const size_t nmsTopK,
    const real nmsThreshold,
    const size_t keepTopK,
    const real minWidth,
    const real minHeight,
    const vector<vector<UnnormalizedBBox>>& allDecodedBBoxes,
    vector<map<size_t, vector<size_t>>>* allDetectionIndices) {
  size_t totalKeepNum = 0;
  for (size_t n = 0; n < batchSize; ++n) {
    const vector<UnnormalizedBBox>& decodedBBoxes = allDecodedBBoxes[n];
    size_t numDetected = 0;
    map<size_t, vector<size_t>> indices;
    size_t confOffset = n * numPriorBBoxes * numClasses;
    for (size_t c = 0; c < numClasses; ++c) {
      if (c == backgroundId) continue;
      applyNMSFast(decodedBBoxes,
                   confData + confOffset,
                   c,
                   nmsTopK,
                   confThreshold,
                   nmsThreshold,
                   minWidth,
                   minHeight,
                   numPriorBBoxes,
                   numClasses,
                   &(indices[c]));
      numDetected += indices[c].size();
    }
    if (keepTopK > 0 && numDetected > keepTopK) {
      vector<pair<real, pair<size_t, size_t>>> scoreIndexPairs;
      for (size_t c = 0; c < numClasses; ++c) {
        const vector<size_t>& labelIndices = indices[c];
        for (size_t i = 0; i < labelIndices.size(); ++i) {
          size_t idx = labelIndices[i];
          scoreIndexPairs.push_back(
              std::make_pair((confData + confOffset)[idx * numClasses + c],
                             std::make_pair(c, idx)));
        }
      }
      std::sort(scoreIndexPairs.begin(),
                scoreIndexPairs.end(),
                sortScorePairDescend<pair<size_t, size_t>>);
      scoreIndexPairs.resize(keepTopK);
      map<size_t, vector<size_t>> newIndices;
      for (size_t i = 0; i < scoreIndexPairs.size(); ++i) {
        size_t label = scoreIndexPairs[i].second.first;
        size_t idx = scoreIndexPairs[i].second.second;
        newIndices[label].push_back(idx);
      }
      allDetectionIndices->push_back(newIndices);
      totalKeepNum += keepTopK;
    } else {
      allDetectionIndices->push_back(indices);
      totalKeepNum += numDetected;
    }
  }
  return totalKeepNum;
}

void ProposalLayer::getDetectionOutput(
    const real* confData,
    const size_t numKept,
    const size_t numPriorBBoxes,
    const size_t numClasses,
    const size_t batchSize,
    const vector<map<size_t, vector<size_t>>>& allIndices,
    const vector<vector<UnnormalizedBBox>>& allDecodedBBoxes,
    Matrix& out) {
  MatrixPtr outBuffer;
  Matrix::resizeOrCreate(outBuffer, numKept, 7, false, false);
  real* bufferData = outBuffer->getData();
  size_t count = 0;
  for (size_t n = 0; n < batchSize; ++n) {
    for (map<size_t, vector<size_t>>::const_iterator it = allIndices[n].begin();
         it != allIndices[n].end();
         ++it) {
      size_t label = it->first;
      const vector<size_t>& indices = it->second;
      const vector<UnnormalizedBBox>& decodedBBoxes = allDecodedBBoxes[n];
      for (size_t i = 0; i < indices.size(); ++i) {
        size_t idx = indices[i];
        size_t confOffset = n * numPriorBBoxes * numClasses + idx * numClasses;
        bufferData[count * 7] = n;
        bufferData[count * 7 + 1] = label;
        bufferData[count * 7 + 2] = (confData + confOffset)[label];
        bufferData[count * 7 + 3] = decodedBBoxes[idx].xMin;
        bufferData[count * 7 + 4] = decodedBBoxes[idx].yMin;
        bufferData[count * 7 + 5] = decodedBBoxes[idx].xMax;
        bufferData[count * 7 + 6] = decodedBBoxes[idx].yMax;
        ++count;
      }
    }
  }
  out.copyFrom(bufferData, numKept * 7);
}

void ProposalLayer::decodeTarget(const std::vector<real>& anchorBoxData,
                                 const std::vector<real>& locPredData,
                                 UnnormalizedBBox& predBox) {
  real anchorBoxWidth = anchorBoxData[2] - anchorBoxData[0] + 1;
  real anchorBoxHeight = anchorBoxData[3] - anchorBoxData[1] + 1;
  real anchorBoxCenterX = (anchorBoxData[2] + anchorBoxData[0]) / 2;
  real anchorBoxCenterY = (anchorBoxData[3] + anchorBoxData[1]) / 2;

  real dx = locPredData[0];
  real dy = locPredData[1];
  real dw = locPredData[2];
  real dh = locPredData[3];

  real predCtrX = dx * anchorBoxWidth + anchorBoxCenterX;
  real predCtrY = dy * anchorBoxHeight + anchorBoxCenterY;
  real predWidth = std::exp(dw * anchorBoxWidth);
  real predHeight = std::exp(dh * anchorBoxHeight);

  // clip predicted box to image
  real xMin = static_cast<real>(0.);
  real yMin = static_cast<real>(0.);
  real xMax = anchorBoxData[5] - 1;
  real yMax = anchorBoxData[6] - 1;
  predBox.xMin = std::min(
      std::max(static_cast<real>(predCtrX - 0.5 * predWidth), xMin), xMax);
  predBox.yMin = std::min(
      std::max(static_cast<real>(predCtrY - 0.5 * predHeight), yMin), yMax);
  predBox.xMax = std::min(
      std::max(static_cast<real>(predCtrX + 0.5 * predWidth), xMin), xMax);
  predBox.yMax = std::min(
      std::max(static_cast<real>(predCtrY + 0.5 * predHeight), yMin), yMax);
}

void ProposalLayer::forward(PassType passType) {
  Layer::forward(passType);
  size_t batchSize = getInputValue(*getLocInputLayer(0))->getHeight();

  locSizeSum_ = 0;
  confSizeSum_ = 0;
  for (size_t n = 0; n < inputNum_; ++n) {
    const MatrixPtr inLoc = getInputValue(*getLocInputLayer(n));
    const MatrixPtr inConf = getInputValue(*getConfInputLayer(n));
    locSizeSum_ += inLoc->getElementCnt();
    confSizeSum_ += inConf->getElementCnt();
  }

  Matrix::resizeOrCreate(locTmpBuffer_, 1, locSizeSum_, false, useGpu_);
  Matrix::resizeOrCreate(
      confTmpBuffer_, confSizeSum_ / numClasses_, numClasses_, false, useGpu_);

  size_t locOffset = 0;
  size_t confOffset = 0;
  auto& layerConf = config_.inputs(0).proposal_conf();
  for (size_t n = 0; n < inputNum_; ++n) {
    const MatrixPtr inLoc = getInputValue(*getLocInputLayer(n));
    const MatrixPtr inConf = getInputValue(*getConfInputLayer(n));

    size_t height = getInput(*getLocInputLayer(n)).getFrameHeight();
    if (!height) height = layerConf.height();
    size_t width = getInput(*getLocInputLayer(n)).getFrameWidth();
    if (!width) width = layerConf.width();
    locOffset += appendWithPermute(*inLoc,
                                   height,
                                   width,
                                   locSizeSum_,
                                   locOffset,
                                   batchSize,
                                   *locTmpBuffer_,
                                   kNCHWToNHWC);
    confOffset += appendWithPermute(*inConf,
                                    height,
                                    width,
                                    confSizeSum_,
                                    confOffset,
                                    batchSize,
                                    *confTmpBuffer_,
                                    kNCHWToNHWC);
  }
  CHECK_EQ(locOffset, locSizeSum_ / batchSize);
  CHECK_EQ(confOffset, confSizeSum_ / batchSize);

  MatrixPtr priorValue;
  if (useGpu_) {
    Matrix::resizeOrCreate(locCpuBuffer_, 1, locSizeSum_, false, false);
    Matrix::resizeOrCreate(
        confCpuBuffer_, confSizeSum_ / numClasses_, numClasses_, false, false);
    MatrixPtr priorTmpValue = getInputValue(*getPriorBoxLayer());
    Matrix::resizeOrCreate(
        priorCpuValue_, 1, priorTmpValue->getElementCnt(), false, false);

    locCpuBuffer_->copyFrom(*locTmpBuffer_);
    confCpuBuffer_->copyFrom(*confTmpBuffer_);
    priorCpuValue_->copyFrom(*priorTmpValue);

    locBuffer_ = locCpuBuffer_;
    confBuffer_ = confCpuBuffer_;
    priorValue = priorCpuValue_;
  } else {
    priorValue = getInputValue(*getPriorBoxLayer());
    locBuffer_ = locTmpBuffer_;
    confBuffer_ = confTmpBuffer_;
  }
  confBuffer_->softmax(*confBuffer_);

  size_t numPriors = priorValue->getElementCnt() / 7;
  std::vector<std::vector<UnnormalizedBBox>> allDecodedBBoxes;
  for (size_t n = 0; n < batchSize; ++n) {
    std::vector<UnnormalizedBBox> decodedBBoxes;
    for (size_t i = 0; i < numPriors; ++i) {
      size_t priorOffset = i * 7;
      std::vector<real> anchorBoxData;
      for (size_t j = 0; j < 7; ++j)
        anchorBoxData.push_back(*(priorValue->getData() + priorOffset + j));
      size_t locPredOffset = n * numPriors * 4 + i * 4;
      std::vector<real> locPredData;
      for (size_t j = 0; j < 4; ++j)
        locPredData.push_back(*(locBuffer_->getData() + locPredOffset + j));
      UnnormalizedBBox bbox;
      decodeTarget(anchorBoxData, locPredData, bbox);
      decodedBBoxes.push_back(bbox);
    }
    allDecodedBBoxes.push_back(decodedBBoxes);
  }

  std::vector<std::map<size_t, std::vector<size_t>>> allIndices;
  size_t numKept = getDetectionIndices(confBuffer_->getData(),
                                       numPriors,
                                       numClasses_,
                                       backgroundId_,
                                       batchSize,
                                       confidenceThreshold_,
                                       nmsTopK_,
                                       nmsThreshold_,
                                       keepTopK_,
                                       minWidth_,
                                       minHeight_,
                                       allDecodedBBoxes,
                                       &allIndices);

  resetOutput(numKept, 7);
  MatrixPtr outV = getOutputValue();
  getDetectionOutput(confBuffer_->getData(),
                     numKept,
                     numPriors,
                     numClasses_,
                     batchSize,
                     allIndices,
                     allDecodedBBoxes,
                     *outV);
}

}  // namespace paddle
