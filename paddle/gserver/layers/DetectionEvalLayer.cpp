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
#include "paddle/math/BaseMatrix.h"
#include "paddle/math/Matrix.h"

using std::map;
using std::vector;

namespace paddle {
/**
 * A layer for preparing the data for the detection evalutator.
 * - Input: Two and only two input layer are accepted. The input layer must be
 *          a detection_output layer's output and a group of bbox labels.
 * - Output: The structured data for the detection evaluator..
 */

class DetectionEvalLayer : public Layer {
public:
  explicit DetectionEvalLayer(const LayerConfig& config) : Layer(config) {}
  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  LayerPtr getDetOutputLayer() { return inputLayers_[0]; }
  LayerPtr getLabelLayer() { return inputLayers_[1]; }

  void forward(PassType passType);
  void backward(const UpdateCallback& callback) {}

  void generateBBox(const size_t batchSize,
                    const int* labelIndex,
                    MatrixPtr labelValue,
                    MatrixPtr detValue,
                    vector<map<size_t, vector<vector<real>>>>* allGtBBox,
                    vector<map<size_t, vector<vector<real>>>>* allDetBBox);
  void generateOutput(const size_t batchSize,
                      map<size_t, size_t> numPos,
                      vector<map<size_t, vector<vector<real>>>> allGtBBox,
                      vector<map<size_t, vector<vector<real>>>> allDetBBox,
                      real* outData);
  real jaccardOverlap(const real* bbox1, const real* bbox2);
  static bool sortBBoxDesend(const vector<real>& bbox1,
                             const vector<real>& bbox2) {
    return bbox1[0] > bbox2[0];
  }

protected:
  size_t numClasses_;
  real overlapThreshold_;
  size_t backgroundId_;
  bool evaluateDifficult_;
  size_t testNum_;
  MatrixPtr outCpuV_;
};

REGISTER_LAYER(detection_eval, DetectionEvalLayer);

bool DetectionEvalLayer::init(const LayerMap& layerMap,
                              const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);

  auto detEvalConf = config_.inputs(0).detection_eval_conf();
  numClasses_ = detEvalConf.num_classes();
  overlapThreshold_ = detEvalConf.overlap_threshold();
  backgroundId_ = detEvalConf.background_id();
  evaluateDifficult_ = detEvalConf.evaluate_difficult();
  testNum_ = 1;
  return true;
}

void DetectionEvalLayer::forward(PassType passType) {
  Layer::forward(passType);

  // detValue layout:
  // | imageId1 | labelId1 | confScore1 | xmin1 | ymin1 | xmax1 | ymax1 | .....
  // | imageIdN | labelIdN | confScoreN | xminN | yminN | xmaxN | ymaxN |
  MatrixPtr detValue;
  // labelValue layout:
  // classX_Y means the Yth object of the Xth sample.
  // | class1_1 | xmin1_1 | ymin1_1 | xmax1_1 | ymax1_1 | difficult1_1 | ......
  // | classN_M | xminN_M | yminN_M | xmaxN_M | ymaxN_M | difficultN_M
  MatrixPtr labelValue;
  // Copy data from GPU to CPU if use GPU
  if (useGpu_) {
    MatrixPtr detTmpValue = getInputValue(*getDetOutputLayer());
    Matrix::resizeOrCreate(detValue,
                           detTmpValue->getHeight(),
                           detTmpValue->getWidth(),
                           false,
                           false);
    MatrixPtr labelTmpValue = getInputValue(*getLabelLayer());
    Matrix::resizeOrCreate(labelValue,
                           labelTmpValue->getHeight(),
                           labelTmpValue->getWidth(),
                           false,
                           false);
    detValue->copyFrom(*detTmpValue);
    labelValue->copyFrom(*labelTmpValue);
  } else {
    detValue = getInputValue(*getDetOutputLayer());
    labelValue = getInputValue(*getLabelLayer());
  }

  Argument label = getInput(*getLabelLayer());
  const int* labelIndex = label.sequenceStartPositions->getData(false);
  int seqNum = label.getNumSequences();
  size_t batchSize = seqNum;

  map<size_t, size_t> numPos;
  vector<map<size_t, vector<vector<real>>>> allGtBBox;
  vector<map<size_t, vector<vector<real>>>> allDetBBox;

  generateBBox(
      batchSize, labelIndex, labelValue, detValue, &allGtBBox, &allDetBBox);

  // Count the ground truth number of each class
  for (size_t n = 0; n < batchSize; n++) {
    for (map<size_t, vector<vector<real>>>::iterator it = allGtBBox[n].begin();
         it != allGtBBox[n].end();
         it++) {
      size_t count = 0;
      if (evaluateDifficult_) {
        count = it->second.size();
      } else {
        for (size_t i = 0; i < it->second.size(); i++)
          if (!(it->second[i][5])) count++;
      }
      if (numPos.find(it->first) == numPos.end()) {
        numPos[it->first] = count;
      } else {
        numPos[it->first] += count;
      }
    }
  }

  size_t outHeight = numClasses_ + detValue->getHeight() - 1;
  Matrix::resizeOrCreate(outCpuV_, outHeight, 5, false, false);
  real* outData = outCpuV_->getData();

  generateOutput(batchSize, numPos, allGtBBox, allDetBBox, outData);

  resetOutput(outHeight, 5);
  MatrixPtr outV = getOutputValue();
  outV->copyFrom(*outCpuV_);
}

void DetectionEvalLayer::generateOutput(
    const size_t batchSize,
    map<size_t, size_t> numPos,
    vector<map<size_t, vector<vector<real>>>> allGtBBox,
    vector<map<size_t, vector<vector<real>>>> allDetBBox,
    real* outData) {
  size_t numDet = 0;
  for (size_t c = 0; c < numClasses_; c++) {
    if (c == backgroundId_) continue;
    outData[numDet * 5] = -1;
    outData[numDet * 5 + 1] = c;
    if (numPos.find(c) == numPos.end()) {
      outData[numDet * 5 + 2] = 0;
    } else {
      outData[numDet * 5 + 2] = numPos.find(c)->second;
    }
    outData[numDet * 5 + 3] = -1;
    outData[numDet * 5 + 4] = -1;
    numDet++;
  }
  // Insert detection evaluate status.
  for (size_t n = 0; n < batchSize; n++) {
    if (allGtBBox[n].size() == 0) {
      // No ground truth for current image. All detections become false pos.
      for (map<size_t, vector<vector<real>>>::iterator it =
               allDetBBox[n].begin();
           it != allDetBBox[n].end();
           it++) {
        size_t label = it->first;
        vector<vector<real>> predBBoxes = it->second;
        for (size_t i = 0; i < predBBoxes.size(); i++) {
          outData[numDet * 5] = n;
          outData[numDet * 5 + 1] = label;
          outData[numDet * 5 + 2] = predBBoxes[i][0];
          outData[numDet * 5 + 3] = 0;
          outData[numDet * 5 + 4] = 1;
          numDet++;
        }
      }
    } else {
      for (map<size_t, vector<vector<real>>>::iterator it =
               allDetBBox[n].begin();
           it != allDetBBox[n].end();
           it++) {
        size_t label = it->first;
        vector<vector<real>> predBBoxes = it->second;
        if (allGtBBox[n].find(label) == allGtBBox[n].end()) {
          // No ground truth for current label. All detections become falsepos.
          for (size_t i = 0; i < predBBoxes.size(); i++) {
            outData[numDet * 5] = n;
            outData[numDet * 5 + 1] = label;
            outData[numDet * 5 + 2] = predBBoxes[i][0];
            outData[numDet * 5 + 3] = 0;
            outData[numDet * 5 + 4] = 1;
            numDet++;
          }
        } else {
          vector<vector<real>> gtBBoxes = allGtBBox[n].find(label)->second;
          vector<bool> visited(gtBBoxes.size(), false);
          // Sort detections in descend order based on scores.
          std::sort(predBBoxes.begin(), predBBoxes.end(), sortBBoxDesend);
          for (size_t i = 0; i < predBBoxes.size(); i++) {
            outData[numDet * 5] = n;
            outData[numDet * 5 + 1] = label;
            outData[numDet * 5 + 2] = predBBoxes[i][0];
            // Compare with each ground truth bbox.
            float overlapMax = -1;
            size_t idxMax = 0;
            for (size_t j = 0; j < gtBBoxes.size(); j++) {
              real overlap =
                  jaccardOverlap(&(predBBoxes[i][0]), &(gtBBoxes[j][1]));
              if (overlap > overlapMax) {
                overlapMax = overlap;
                idxMax = j;
              }
            }
            if (overlapMax > overlapThreshold_) {
              if (evaluateDifficult_ ||
                  (!evaluateDifficult_ && gtBBoxes[idxMax][5])) {
                if (!visited[idxMax]) {
                  // True positive.
                  outData[numDet * 5 + 3] = 1;
                  outData[numDet * 5 + 4] = 0;
                  visited[idxMax] = true;
                } else {
                  // False positive (multiple detection).
                  outData[numDet * 5 + 3] = 0;
                  outData[numDet * 5 + 4] = 1;
                }
              }
            } else {
              // False positive.
              outData[numDet * 5 + 3] = 0;
              outData[numDet * 5 + 4] = 1;
            }
            numDet++;
          }
        }
      }
    }
  }
}

void DetectionEvalLayer::generateBBox(
    const size_t batchSize,
    const int* labelIndex,
    MatrixPtr labelValue,
    MatrixPtr detValue,
    vector<map<size_t, vector<vector<real>>>>* allGtBBox,
    vector<map<size_t, vector<vector<real>>>>* allDetBBox) {
  // Get the ground truth bbox
  for (size_t n = 0; n < batchSize; n++) {
    map<size_t, vector<vector<real>>> bboxes;
    for (int i = labelIndex[n]; i < labelIndex[n + 1]; i++) {
      vector<real> bbox;
      bbox.push_back(labelValue->getData()[i * 6 + 1]);
      bbox.push_back(labelValue->getData()[i * 6 + 2]);
      bbox.push_back(labelValue->getData()[i * 6 + 3]);
      bbox.push_back(labelValue->getData()[i * 6 + 4]);
      bbox.push_back(labelValue->getData()[i * 6 + 5]);
      int c = labelValue->getData()[i * 6];
      bboxes[c].push_back(bbox);
    }
    allGtBBox->push_back(bboxes);
  }

  // Get the predicted bbox
  size_t imageId = 0;
  for (size_t n = 0; n < detValue->getHeight(); n++) {
    map<size_t, vector<vector<real>>> bboxes;
    while (detValue->getData()[n * 7] == imageId && n < detValue->getHeight()) {
      vector<real> bbox;
      size_t label = detValue->getData()[n * 7 + 1];
      bbox.push_back(detValue->getData()[n * 7 + 2]);
      bbox.push_back(detValue->getData()[n * 7 + 3]);
      bbox.push_back(detValue->getData()[n * 7 + 4]);
      bbox.push_back(detValue->getData()[n * 7 + 5]);
      bbox.push_back(detValue->getData()[n * 7 + 6]);
      bboxes[label].push_back(bbox);
      n++;
    }
    imageId++;
    allDetBBox->push_back(bboxes);
  }
}

real DetectionEvalLayer::jaccardOverlap(const real* bbox1, const real* bbox2) {
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

}  // namespace paddle
