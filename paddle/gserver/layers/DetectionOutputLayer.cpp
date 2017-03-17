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

#include <map>
#include <vector>
#include "Layer.h"

using std::vector;
using std::map;
using std::pair;

namespace paddle {

/**
 * The detection output layer for a SSD detection task. This layer apply the
 * Non-maximum suppression to the all predicted bounding box and keep the
 * Top-K bounding boxes.
 * - Input: This layer needs three input layers: This first input layer
 *          is the priorbox layer. The rest two input layers are convolution
 *          layers for generating bbox location offset and the classification
 *          confidence.
 * - Output: The predicted bounding box location and confidence.
 */

class DetectionOutputLayer : public Layer {
public:
  explicit DetectionOutputLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  LayerPtr getPriorBoxLayer() { return inputLayers_[0]; }
  LayerPtr getLocInputLayer(size_t index) { return inputLayers_[1 + index]; }
  LayerPtr getConfInputLayer(size_t index) {
    return inputLayers_[1 + inputNum_ + index];
  }
  void forward(PassType passType);

  void backward(const UpdateCallback& callback = nullptr) {}

  void forwardDataProcess(size_t batchSize);

  void generateIndices(const size_t numPriors,
                       const size_t batchSize,
                       const vector<vector<real>> allDecodeBBoxes,
                       size_t& numKept,
                       vector<map<size_t, vector<size_t>>>* allIndices);
  void generateOutput(const size_t numKept,
                      const size_t numPriors,
                      const size_t batchSize,
                      vector<map<size_t, vector<size_t>>> allIndices,
                      const vector<vector<real>> allDecodeBBoxes);
  void decodeBBox(const real* priorData,
                  const real* locPredData,
                  vector<real>* bbox);
  real jaccardOverlap(const real* bbox1, const real* bbox2);
  void applyNMSFast(const vector<real> bboxes,
                    const real* confData,
                    size_t classIdx,
                    size_t numPriors,
                    vector<size_t>* indices);
  template <typename T>
  static bool sortPairDescend(const pair<real, T>& pair1,
                              const pair<real, T>& pair2) {
    return pair1.first > pair2.first;
  }

protected:
  size_t numClasses_;
  size_t inputNum_;  // The number of input location/confidence layers
  real nmsThreshold_;
  real confidenceThreshold_;
  size_t topK_;      // The number of selected bounding boxes before NMS
  size_t keepTopK_;  // The number of kept bounding boxes after NMS
  size_t backgroundId_;

  size_t locSizeSum_;
  size_t confSizeSum_;

  MatrixPtr outCpuV_;
  MatrixPtr locBuffer_;
  MatrixPtr confBuffer_;
  MatrixPtr locTmpBuffer_;
  MatrixPtr confTmpBuffer_;
  MatrixPtr priorCpuValue_;
  MatrixPtr locCpuBuffer_;
  MatrixPtr confCpuBuffer_;
};

REGISTER_LAYER(detection_output, DetectionOutputLayer);

bool DetectionOutputLayer::init(const LayerMap& layerMap,
                                const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);
  auto& detOutConf = config_.inputs(0).detection_output_conf();
  numClasses_ = detOutConf.num_classes();
  inputNum_ = detOutConf.input_num();
  nmsThreshold_ = detOutConf.nms_threshold();
  confidenceThreshold_ = detOutConf.confidence_threshold();
  topK_ = detOutConf.top_k();
  keepTopK_ = detOutConf.keep_top_k();
  backgroundId_ = detOutConf.background_id();
  return true;
}

void DetectionOutputLayer::forward(PassType passType) {
  Layer::forward(passType);
  // Same as getConfInputLayer(0)
  size_t batchSize = getInputValue(*getLocInputLayer(0))->getHeight();

  // Allocate buffer memory
  locSizeSum_ = 0;
  confSizeSum_ = 0;
  for (size_t n = 0; n < inputNum_; n++) {
    const MatrixPtr inLoc = getInputValue(*getLocInputLayer(n));
    const MatrixPtr inConf = getInputValue(*getConfInputLayer(n));
    locSizeSum_ += inLoc->getElementCnt();
    confSizeSum_ += inConf->getElementCnt();
  }
  // locBuffer layout:
  // | xmin1 | ymin1 | xmax1 | ymax1 | xmin2 ......
  // confBuffer layout:
  // | class1 score | class2 score | ... |classN score | class1 score | ......
  Matrix::resizeOrCreate(locTmpBuffer_, 1, locSizeSum_, false, useGpu_);
  Matrix::resizeOrCreate(
      confTmpBuffer_, confSizeSum_ / numClasses_, numClasses_, false, useGpu_);
  locBuffer_ = locTmpBuffer_;
  confBuffer_ = confTmpBuffer_;

  forwardDataProcess(batchSize);
  // priorValue layout:
  // | xmin1 | ymin1 | xmax1 | ymax1 | xmin1Var | ymin1Var | xmax1Var
  // | ymax1Var | xmin2 | ......
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
  }
  confBuffer_->softmax(*confBuffer_);

  size_t numPriors = priorValue->getElementCnt() / 8;
  // Decode all predict bbox.
  vector<vector<real>> allDecodeBBoxes;
  for (size_t n = 0; n < batchSize; n++) {
    vector<real> decodeBBoxes;
    for (size_t i = 0; i < numPriors; i++) {
      size_t priorOffset = i * 8;
      size_t locPredOffset = n * numPriors * 4 + i * 4;
      vector<real> bbox;
      decodeBBox(priorValue->getData() + priorOffset,
                 locBuffer_->getData() + locPredOffset,
                 &bbox);
      decodeBBoxes.insert(decodeBBoxes.end(), bbox.begin(), bbox.end());
    }
    allDecodeBBoxes.push_back(decodeBBoxes);
  }

  size_t numKept = 0;
  vector<map<size_t, vector<size_t>>> allIndices;
  generateIndices(numPriors, batchSize, allDecodeBBoxes, numKept, &allIndices);

  resetOutput(numKept,
              7);  // ImageId, label, confidence, xmin, ymin, xmax, ymax
  generateOutput(numKept, numPriors, batchSize, allIndices, allDecodeBBoxes);
}

void DetectionOutputLayer::generateOutput(
    const size_t numKept,
    const size_t numPriors,
    const size_t batchSize,
    vector<map<size_t, vector<size_t>>> allIndices,
    const vector<vector<real>> allDecodeBBoxes) {
  // Copy the Top-K result to the layer ouput
  MatrixPtr outV = getOutputValue();
  Matrix::resizeOrCreate(outCpuV_, numKept, 7, false, false);
  real* outData = outCpuV_->getData();
  size_t count = 0;
  for (size_t n = 0; n < batchSize; n++) {
    for (map<size_t, vector<size_t>>::iterator it = allIndices[n].begin();
         it != allIndices[n].end();
         it++) {
      size_t label = it->first;
      vector<size_t>& indices = it->second;
      const vector<real> decodeBBoxes = allDecodeBBoxes[n];
      for (size_t i = 0; i < indices.size(); i++) {
        size_t idx = indices[i];
        size_t confOffset = n * numPriors * numClasses_ + idx * numClasses_;
        outData[count * 7] = n;
        outData[count * 7 + 1] = label;
        outData[count * 7 + 2] = (confBuffer_->getData() + confOffset)[label];
        outData[count * 7 + 3] =
            std::max(std::min(decodeBBoxes[idx * 4], 1.f), 0.f);
        outData[count * 7 + 4] =
            std::max(std::min(decodeBBoxes[idx * 4 + 1], 1.f), 0.f);
        outData[count * 7 + 5] =
            std::max(std::min(decodeBBoxes[idx * 4 + 2], 1.f), 0.f);
        outData[count * 7 + 6] =
            std::max(std::min(decodeBBoxes[idx * 4 + 3], 1.f), 0.f);
        count++;
      }
    }
  }
  outV->copyFrom(outData, numKept * 7);
}

void DetectionOutputLayer::generateIndices(
    const size_t numPriors,
    const size_t batchSize,
    const vector<vector<real>> allDecodeBBoxes,
    size_t& numKept,
    vector<map<size_t, vector<size_t>>>* allIndices) {
  // Apply the NMS and keep the Top-K results.
  for (size_t n = 0; n < batchSize; n++) {
    const vector<real> decodeBBoxes = allDecodeBBoxes[n];
    size_t numDet = 0;
    map<size_t, vector<size_t>> indices;
    size_t confOffset = n * numPriors * numClasses_;
    for (size_t c = 0; c < numClasses_; c++) {
      if (c == backgroundId_) continue;
      applyNMSFast(decodeBBoxes,
                   confBuffer_->getData() + confOffset,
                   c,
                   numPriors,
                   &(indices[c]));
      numDet += indices[c].size();
    }
    if (keepTopK_ > 0 && numDet > keepTopK_) {
      vector<pair<real, pair<size_t, size_t>>> scoreIndexPairs;
      for (size_t c = 0; c < numClasses_; c++) {
        const vector<size_t>& labelIndices = indices[c];
        for (size_t i = 0; i < labelIndices.size(); i++) {
          size_t idx = labelIndices[i];
          scoreIndexPairs.push_back(std::make_pair(
              (confBuffer_->getData() + confOffset)[idx * numClasses_ + c],
              std::make_pair(c, idx)));
        }
      }
      // Keep top k results per image.
      std::sort(scoreIndexPairs.begin(),
                scoreIndexPairs.end(),
                sortPairDescend<pair<size_t, size_t>>);
      scoreIndexPairs.resize(keepTopK_);
      // Store the new indices.
      map<size_t, vector<size_t>> newIndices;
      for (size_t i = 0; i < scoreIndexPairs.size(); i++) {
        size_t label = scoreIndexPairs[i].second.first;
        size_t idx = scoreIndexPairs[i].second.second;
        newIndices[label].push_back(idx);
      }
      allIndices->push_back(newIndices);
      numKept += keepTopK_;
    } else {
      allIndices->push_back(indices);
      numKept += numDet;
    }
  }
}

real DetectionOutputLayer::jaccardOverlap(const real* bbox1,
                                          const real* bbox2) {
  real xMin1 = bbox1[0];
  real yMin1 = bbox1[1];
  real xMax1 = bbox1[2];
  real yMax1 = bbox1[3];
  real xMin2 = bbox2[0];
  real yMin2 = bbox2[1];
  real xMax2 = bbox2[2];
  real yMax2 = bbox2[3];

  real width1 = xMax1 - xMin1;
  real height1 = yMax1 - yMin1;
  real width2 = xMax2 - xMin2;
  real height2 = yMax2 - yMin2;
  real bboxSize1 = width1 * height1;
  real bboxSize2 = width2 * height2;

  real intersectWidth;
  real intersectHeight;

  if (!(xMin1 > xMax2 || yMin1 > yMax2 || xMax1 < xMin2 || yMax1 < yMin2)) {
    intersectWidth = std::min(xMax1, xMax2) - std::max(xMin1, xMin2);
    intersectHeight = std::min(yMax1, yMax2) - std::max(yMin1, yMin2);
    real intersectSize = intersectWidth * intersectHeight;
    real overlap = intersectSize / (bboxSize1 + bboxSize2 - intersectSize);
    return overlap;
  } else {
    return 0;
  }
}

void DetectionOutputLayer::applyNMSFast(const vector<real> bboxes,
                                        const real* confData,
                                        size_t classIdx,
                                        size_t numPriors,
                                        vector<size_t>* indices) {
  vector<pair<real, size_t>> scores;
  for (size_t i = 0; i < numPriors; i++) {
    size_t confOffset = i * numClasses_ + classIdx;
    if (confData[confOffset] > confidenceThreshold_) {
      scores.push_back(std::make_pair(confData[confOffset], i));
    }
  }
  std::stable_sort(scores.begin(), scores.end(), sortPairDescend<size_t>);
  if (topK_ > 0 && topK_ < scores.size()) scores.resize(topK_);

  while (scores.size() > 0) {
    const size_t idx = scores.front().second;
    bool keep = true;
    for (size_t i = 0; i < indices->size(); i++) {
      if (keep) {
        const size_t keptIdx = (*indices)[i];
        real overlap = jaccardOverlap(&bboxes[idx * 4], &bboxes[keptIdx * 4]);
        keep = overlap <= nmsThreshold_;
      } else {
        break;
      }
    }
    if (keep) indices->push_back(idx);
    scores.erase(scores.begin());
  }
}

void DetectionOutputLayer::decodeBBox(const real* priorData,
                                      const real* locPredData,
                                      vector<real>* bbox) {
  real priorWidth = priorData[2] - priorData[0];
  real priorHeight = priorData[3] - priorData[1];
  real priorCenterX = (priorData[0] + priorData[2]) / 2;
  real priorCenterY = (priorData[1] + priorData[3]) / 2;

  real decodeWidth = std::exp(priorData[6] * locPredData[2]) * priorWidth;
  real decodeHeight = std::exp(priorData[7] * locPredData[3]) * priorHeight;
  real decodeCenterX =
      priorData[4] * locPredData[0] * priorWidth + priorCenterX;
  real decodeCenterY =
      priorData[5] * locPredData[1] * priorHeight + priorCenterY;

  bbox->push_back(decodeCenterX - decodeWidth / 2);
  bbox->push_back(decodeCenterY - decodeHeight / 2);
  bbox->push_back(decodeCenterX + decodeWidth / 2);
  bbox->push_back(decodeCenterY + decodeHeight / 2);
}

void DetectionOutputLayer::forwardDataProcess(size_t batchSize) {
  // BatchTrans (from NCHW to NHWC) and concat
  size_t locOffset = 0;
  size_t confOffset = 0;
  // For unit test
  auto& detOutConf = config_.inputs(0).detection_output_conf();
  // Each input layer has different size
  for (size_t n = 0; n < inputNum_; n++) {
    const MatrixPtr inLoc = getInputValue(*getLocInputLayer(n));
    const MatrixPtr inConf = getInputValue(*getConfInputLayer(n));

    size_t locSize = inLoc->getElementCnt();
    size_t confSize = inConf->getElementCnt();
    size_t height = getInput(*getLocInputLayer(n)).getFrameHeight();
    if (!height) height = detOutConf.height();
    size_t width = getInput(*getLocInputLayer(n)).getFrameWidth();
    if (!width) width = detOutConf.width();
    size_t locChannels = locSize / (height * width * batchSize);
    size_t confChannels = confSize / (height * width * batchSize);
    size_t imgPixels = height * width;

    for (size_t i = 0; i < batchSize; i++) {
      // Concat with axis N (NCHW -> NHWC)
      size_t locBatchOffset = i * (locSizeSum_ / batchSize) + locOffset;
      size_t confBatchOffset = i * (confSizeSum_ / batchSize) + confOffset;
      const MatrixPtr inLocTmp =
          Matrix::create(inLoc->getData() + i * locChannels * imgPixels,
                         locChannels,
                         imgPixels,
                         false,
                         useGpu_);
      MatrixPtr outLocTmp =
          Matrix::create(locBuffer_->getData() + locBatchOffset,
                         imgPixels,
                         locChannels,
                         false,
                         useGpu_);
      inLocTmp->transpose(outLocTmp, false);
      const MatrixPtr inConfTmp =
          Matrix::create(inConf->getData() + i * confChannels * imgPixels,
                         confChannels,
                         imgPixels,
                         false,
                         useGpu_);
      MatrixPtr outConfTmp =
          Matrix::create(confBuffer_->getData() + confBatchOffset,
                         imgPixels,
                         confChannels,
                         false,
                         useGpu_);
      inConfTmp->transpose(outConfTmp, false);
    }
    locOffset += locChannels * imgPixels;
    confOffset += confChannels * imgPixels;
  }
  CHECK_EQ(locOffset, locSizeSum_ / batchSize);
  CHECK_EQ(confOffset, confSizeSum_ / batchSize);
}

}  // namespace paddle
