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

#include "Evaluator.h"
// #include "paddle/gserver/gradientmachines/NeuralNetwork.h"
using std::map;
using std::vector;
using std::pair;
using std::make_pair;

namespace paddle {

/**
 * @brief detection map Evaluator
 *
 * The config file api is detection_map_evaluator.
 */
class DetectionMAPEvaluator : public Evaluator {
public:
  DetectionMAPEvaluator()
      : evaluateDifficult_(false), cpuOutput_(nullptr), cpuLabel_(nullptr) {}

  void start() {
    Evaluator::start();
    allTruePos_.clear();
    allFalsePos_.clear();
    numPos_.clear();
  }

  static bool sortBBoxDesend(const vector<real>& bbox1,
                             const vector<real>& bbox2) {
    return bbox1[0] > bbox2[0];
  }

  static bool sortScorePairDescend(const pair<real, size_t>& pair1,
                                   const pair<real, size_t>& pair2) {
    return pair1.first > pair2.first;
  }

  void cumSum(vector<pair<real, size_t>> sortPairs,
              vector<size_t>* cumsum) const {
    std::stable_sort(sortPairs.begin(), sortPairs.end(), sortScorePairDescend);
    cumsum->clear();
    size_t sum = 0;
    for (size_t i = 0; i < sortPairs.size(); i++) {
      sum += sortPairs[i].second;
      cumsum->push_back(sum);
    }
  }

  real jaccardOverlap(const real* bbox1, const real* bbox2) {
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

  void generateBBox(const size_t batchSize,
                    const int* labelIndex,
                    MatrixPtr labelValue,
                    MatrixPtr detValue,
                    map<size_t, size_t>* numPos,
                    vector<map<size_t, vector<vector<real>>>>* allGtBBox,
                    vector<map<size_t, vector<vector<real>>>>* allDetBBox) {
    // Get the ground truth bbox
    for (size_t n = 0; n < batchSize; n++) {
      map<size_t, vector<vector<real>>> bboxes;
      for (int i = labelIndex[n]; i < labelIndex[n + 1]; i++) {
        vector<real> bbox;
        bbox.push_back(labelValue->getData()[i * 6 + 1]);  // xmin
        bbox.push_back(labelValue->getData()[i * 6 + 2]);  // ymin
        bbox.push_back(labelValue->getData()[i * 6 + 3]);  // xmax
        bbox.push_back(labelValue->getData()[i * 6 + 4]);  // ymax
        bbox.push_back(labelValue->getData()[i * 6 + 5]);  // difficult
        int c = labelValue->getData()[i * 6];
        bboxes[c].push_back(bbox);
      }
      allGtBBox->push_back(bboxes);
    }

    // Get the predicted bbox
    size_t imageId = 0;
    for (size_t n = 0; n < detValue->getHeight();) {
      map<size_t, vector<vector<real>>> bboxes;
      while (detValue->getData()[n * 7] == imageId &&
             n < detValue->getHeight()) {
        vector<real> bbox;
        size_t label = detValue->getData()[n * 7 + 1];
        bbox.push_back(detValue->getData()[n * 7 + 2]);  // score
        bbox.push_back(detValue->getData()[n * 7 + 3]);  // xmin
        bbox.push_back(detValue->getData()[n * 7 + 4]);  // ymin
        bbox.push_back(detValue->getData()[n * 7 + 5]);  // xmax
        bbox.push_back(detValue->getData()[n * 7 + 6]);  // ymax
        bboxes[label].push_back(bbox);
        n++;
      }
      imageId++;
      if (imageId > batchSize) break;
      allDetBBox->push_back(bboxes);
    }
    for (size_t n = 0; n < batchSize; n++) {
      for (map<size_t, vector<vector<real>>>::iterator it =
               (*allGtBBox)[n].begin();
           it != (*allGtBBox)[n].end();
           it++) {
        size_t count = 0;
        if (evaluateDifficult_) {
          count = it->second.size();
        } else {
          for (size_t i = 0; i < it->second.size(); i++)
            if (!(it->second[i][4])) count++;
        }
        if (numPos->find(it->first) == numPos->end() && count != 0) {
          (*numPos)[it->first] = count;
        } else {
          (*numPos)[it->first] += count;
        }
      }
    }
  }

  void calcTFPos(const size_t batchSize,
                 vector<map<size_t, vector<vector<real>>>> allGtBBox,
                 vector<map<size_t, vector<vector<real>>>> allDetBBox,
                 map<size_t, vector<pair<real, size_t>>>* allTruePos,
                 map<size_t, vector<pair<real, size_t>>>* allFalsePos) {
    for (size_t n = 0; n < allDetBBox.size(); n++) {
      if (allGtBBox[n].size() == 0) {
        // No ground truth for current image. All detections become false pos.
        for (map<size_t, vector<vector<real>>>::iterator it =
                 allDetBBox[n].begin();
             it != allDetBBox[n].end();
             it++) {
          size_t label = it->first;
          vector<vector<real>> predBBoxes = it->second;
          for (size_t i = 0; i < predBBoxes.size(); i++) {
            (*allTruePos)[label].push_back(make_pair(predBBoxes[i][0], 0));
            (*allFalsePos)[label].push_back(make_pair(predBBoxes[i][0], 1));
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
            // No ground truth for current label, All detections falsePos.
            for (size_t i = 0; i < predBBoxes.size(); i++) {
              (*allTruePos)[label].push_back(make_pair(predBBoxes[i][0], 0));
              (*allFalsePos)[label].push_back(make_pair(predBBoxes[i][0], 1));
            }
          } else {
            vector<vector<real>> gtBBoxes = allGtBBox[n].find(label)->second;
            vector<bool> visited(gtBBoxes.size(), false);
            // Sort detections in descend order based on scores.
            std::sort(predBBoxes.begin(), predBBoxes.end(), sortBBoxDesend);
            for (size_t i = 0; i < predBBoxes.size(); i++) {
              // Compare with each ground truth bbox.
              float overlapMax = -1;
              size_t idxMax = 0;
              for (size_t j = 0; j < gtBBoxes.size(); j++) {
                real overlap =
                    jaccardOverlap(&(predBBoxes[i][1]), &(gtBBoxes[j][0]));
                if (overlap > overlapMax) {
                  overlapMax = overlap;
                  idxMax = j;
                }
              }
              if (overlapMax > overlapThreshold_) {
                if (evaluateDifficult_ ||
                    (!evaluateDifficult_ && !gtBBoxes[idxMax][4])) {
                  if (!visited[idxMax]) {
                    // True positive.
                    (*allTruePos)[label].push_back(
                        make_pair(predBBoxes[i][0], 1));
                    (*allFalsePos)[label].push_back(
                        make_pair(predBBoxes[i][0], 0));
                    visited[idxMax] = true;
                  } else {
                    // False positive (multiple detection).
                    (*allTruePos)[label].push_back(
                        make_pair(predBBoxes[i][0], 0));
                    (*allFalsePos)[label].push_back(
                        make_pair(predBBoxes[i][0], 1));
                  }
                }
              } else {
                // False positive.
                (*allTruePos)[label].push_back(make_pair(predBBoxes[i][0], 0));
                (*allFalsePos)[label].push_back(make_pair(predBBoxes[i][0], 1));
              }
            }
          }
        }
      }
    }  // for loop
  }    // clacTFPos

  real evalImp(std::vector<Argument>& arguments) {
    overlapThreshold_ = config_.classification_threshold();
    backgroundId_ = config_.positive_label();
    evaluateDifficult_ = config_.delimited();
    apVersion_ = config_.chunk_scheme();

    // cpuOutput layout:
    // | imageId1 | labelId1 | confScore1 | xmin1 | ymin1 | xmax1 | ymax1 | ...
    // | imageIdN | labelIdN | confScoreN | xminN | yminN | xmaxN | ymaxN |
    // cpuLabel layout:
    // classX_Y means the Yth object of the Xth sample.
    // | class1_1 | xmin1_1 | ymin1_1 | xmax1_1 | ymax1_1 | difficult1_1 | ...
    // | classN_M | xminN_M | yminN_M | xmaxN_M | ymaxN_M | difficultN_M |
    // Copy data to CPU if use GPU
    MatrixPtr detTmpValue = arguments[0].value;
    Matrix::resizeOrCreate(cpuOutput_,
                           detTmpValue->getHeight(),
                           detTmpValue->getWidth(),
                           false,
                           false);
    MatrixPtr labelTmpValue = arguments[1].value;
    Matrix::resizeOrCreate(cpuLabel_,
                           labelTmpValue->getHeight(),
                           labelTmpValue->getWidth(),
                           false,
                           false);
    cpuOutput_->copyFrom(*detTmpValue);
    cpuLabel_->copyFrom(*labelTmpValue);

    Argument label = arguments[1];
    const int* labelIndex = label.sequenceStartPositions->getData(false);
    int batchSize = label.getNumSequences();

    vector<map<size_t, vector<vector<real>>>> allGtBBox;
    vector<map<size_t, vector<vector<real>>>> allDetBBox;

    generateBBox(batchSize,
                 labelIndex,
                 cpuLabel_,
                 cpuOutput_,
                 &numPos_,
                 &allGtBBox,
                 &allDetBBox);

    calcTFPos(batchSize, allGtBBox, allDetBBox, &allTruePos_, &allFalsePos_);
    return 0;
  }

  void printStats(std::ostream& os) const {
    real mAP = 0.0;
    size_t count = 0;
    for (map<size_t, size_t>::const_iterator it = numPos_.begin();
         it != numPos_.end();
         it++) {
      size_t label = it->first;
      size_t labelNumPos = it->second;
      if (labelNumPos == 0 || allTruePos_.find(label) == allTruePos_.end())
        continue;
      vector<pair<real, size_t>> labelTruePos = allTruePos_.find(label)->second;
      vector<pair<real, size_t>> labelFalsePos =
          allFalsePos_.find(label)->second;
      // Compute average precision.
      vector<size_t> tpCumSum;
      cumSum(labelTruePos, &tpCumSum);
      vector<size_t> fpCumSum;
      cumSum(labelFalsePos, &fpCumSum);
      std::vector<real> prec, rec;
      size_t num = tpCumSum.size();
      // Compute Precision.
      for (size_t i = 0; i < num; i++) {
        CHECK_LE(tpCumSum[i], labelNumPos);
        prec.push_back(static_cast<real>(tpCumSum[i]) /
                       static_cast<real>(tpCumSum[i] + fpCumSum[i]));
        rec.push_back(static_cast<real>(tpCumSum[i]) / labelNumPos);
      }
      // VOC2007 style
      if (apVersion_ == "11point") {
        vector<real> maxPrecs(11, 0.0);
        int startIdx = num - 1;
        for (int j = 10; j >= 0; --j)
          for (int i = startIdx; i >= 0; --i) {
            if (rec[i] < j / 10.) {
              startIdx = i;
              if (j > 0) maxPrecs[j - 1] = maxPrecs[j];
              break;
            } else {
              if (maxPrecs[j] < prec[i]) maxPrecs[j] = prec[i];
            }
          }
        for (int j = 10; j >= 0; --j) mAP += maxPrecs[j] / 11;
        count++;
      } else if (apVersion_ == "Integral") {
        // Nature integral
        real averagePrecs = 0.;
        real prevRec = 0.;
        for (size_t i = 0; i < num; i++) {
          if (fabs(rec[i] - prevRec) > 1e-6)
            averagePrecs += prec[i] * fabs(rec[i] - prevRec);
          prevRec = rec[i];
        }
        mAP += averagePrecs;
        count++;
      } else {
        LOG(FATAL) << "Unkown ap version: " << apVersion_;
      }
    }
    if (count != 0) mAP /= count;
    os << "Detection mAP=" << mAP * 100;
  }

  void distributeEval(ParameterClient2* client) {
    LOG(INFO) << "Distribute detection evaluator is not implemented. "
              << "The evaluate mAP may not accurate.";
  }

private:
  real overlapThreshold_;
  bool evaluateDifficult_;
  size_t backgroundId_;
  std::string apVersion_;

  MatrixPtr cpuOutput_;
  MatrixPtr cpuLabel_;

  map<size_t, size_t> numPos_;
  map<size_t, vector<pair<real, size_t>>> allTruePos_;
  map<size_t, vector<pair<real, size_t>>> allFalsePos_;

  void calcMAPInfo(const MatrixPtr& output,
                   const IVectorPtr& label,
                   const MatrixPtr& weight);

  void calcMAPInfoMulti(const MatrixPtr& output,
                        const MatrixPtr& label,
                        const MatrixPtr& weight);
};

REGISTER_EVALUATOR(detection_map, DetectionMAPEvaluator);

}  // namespace paddle
