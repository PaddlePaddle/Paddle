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

#include "Evaluator.h"
#include "paddle/gserver/layers/DetectionUtil.h"

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

  virtual void start() {
    Evaluator::start();
    allTruePos_.clear();
    allFalsePos_.clear();
    numPos_.clear();
  }

  virtual real evalImp(std::vector<Argument>& arguments) {
    overlapThreshold_ = config_.overlap_threshold();
    backgroundId_ = config_.background_id();
    evaluateDifficult_ = config_.evaluate_difficult();
    apType_ = config_.ap_type();

    MatrixPtr detectTmpValue = arguments[0].value;
    Matrix::resizeOrCreate(cpuOutput_,
                           detectTmpValue->getHeight(),
                           detectTmpValue->getWidth(),
                           false,
                           false);

    MatrixPtr labelTmpValue = arguments[1].value;
    Matrix::resizeOrCreate(cpuLabel_,
                           labelTmpValue->getHeight(),
                           labelTmpValue->getWidth(),
                           false,
                           false);

    cpuOutput_->copyFrom(*detectTmpValue);
    cpuLabel_->copyFrom(*labelTmpValue);

    Argument label = arguments[1];
    const int* labelIndex = label.sequenceStartPositions->getData(false);
    size_t batchSize = label.getNumSequences();

    vector<map<size_t, vector<NormalizedBBox>>> allGTBBoxes;
    vector<map<size_t, vector<pair<real, NormalizedBBox>>>> allDetectBBoxes;

    for (size_t n = 0; n < batchSize; ++n) {
      map<size_t, vector<NormalizedBBox>> bboxes;
      for (int i = labelIndex[n]; i < labelIndex[n + 1]; ++i) {
        vector<NormalizedBBox> bbox;
        getBBoxFromLabelData(cpuLabel_->getData() + i * 6, 1, bbox);
        int c = cpuLabel_->getData()[i * 6];
        bboxes[c].push_back(bbox[0]);
      }
      allGTBBoxes.push_back(bboxes);
    }

    size_t n = 0;
    const real* cpuOutputData = cpuOutput_->getData();
    for (size_t imgId = 0; imgId < batchSize; ++imgId) {
      map<size_t, vector<pair<real, NormalizedBBox>>> bboxes;
      size_t curImgId = static_cast<size_t>((cpuOutputData + n * 7)[0]);
      while (curImgId == imgId && n < cpuOutput_->getHeight()) {
        vector<real> label;
        vector<real> score;
        vector<NormalizedBBox> bbox;
        getBBoxFromDetectData(cpuOutputData + n * 7, 1, label, score, bbox);
        bboxes[label[0]].push_back(make_pair(score[0], bbox[0]));
        ++n;
        curImgId = static_cast<size_t>((cpuOutputData + n * 7)[0]);
      }
      allDetectBBoxes.push_back(bboxes);
    }

    for (size_t n = 0; n < batchSize; ++n) {
      for (map<size_t, vector<NormalizedBBox>>::iterator it =
               allGTBBoxes[n].begin();
           it != allGTBBoxes[n].end();
           ++it) {
        size_t count = 0;
        if (evaluateDifficult_) {
          count = it->second.size();
        } else {
          for (size_t i = 0; i < it->second.size(); ++i)
            if (!(it->second[i].isDifficult)) ++count;
        }
        if (numPos_.find(it->first) == numPos_.end() && count != 0) {
          numPos_[it->first] = count;
        } else {
          numPos_[it->first] += count;
        }
      }
    }

    // calcTFPos
    calcTFPos(batchSize, allGTBBoxes, allDetectBBoxes);

    return 0;
  }

  virtual void printStats(std::ostream& os) const {
    real mAP = calcMAP();
    os << "Detection mAP=" << mAP;
  }

  virtual void distributeEval(ParameterClient2* client) {
    LOG(FATAL) << "Distribute detection evaluation not implemented.";
  }

 protected:
  void calcTFPos(const size_t batchSize,
                 const vector<map<size_t, vector<NormalizedBBox>>>& allGTBBoxes,
                 const vector<map<size_t, vector<pair<real, NormalizedBBox>>>>&
                     allDetectBBoxes) {
    for (size_t n = 0; n < allDetectBBoxes.size(); ++n) {
      if (allGTBBoxes[n].size() == 0) {
        for (map<size_t, vector<pair<real, NormalizedBBox>>>::const_iterator
                 it = allDetectBBoxes[n].begin();
             it != allDetectBBoxes[n].end();
             ++it) {
          size_t label = it->first;
          for (size_t i = 0; i < it->second.size(); ++i) {
            allTruePos_[label].push_back(make_pair(it->second[i].first, 0));
            allFalsePos_[label].push_back(make_pair(it->second[i].first, 1));
          }
        }
      } else {
        for (map<size_t, vector<pair<real, NormalizedBBox>>>::const_iterator
                 it = allDetectBBoxes[n].begin();
             it != allDetectBBoxes[n].end();
             ++it) {
          size_t label = it->first;
          vector<pair<real, NormalizedBBox>> predBBoxes = it->second;
          if (allGTBBoxes[n].find(label) == allGTBBoxes[n].end()) {
            for (size_t i = 0; i < predBBoxes.size(); ++i) {
              allTruePos_[label].push_back(make_pair(predBBoxes[i].first, 0));
              allFalsePos_[label].push_back(make_pair(predBBoxes[i].first, 1));
            }
          } else {
            vector<NormalizedBBox> gtBBoxes =
                allGTBBoxes[n].find(label)->second;
            vector<bool> visited(gtBBoxes.size(), false);
            // Sort detections in descend order based on scores
            std::sort(predBBoxes.begin(),
                      predBBoxes.end(),
                      sortScorePairDescend<NormalizedBBox>);
            for (size_t i = 0; i < predBBoxes.size(); ++i) {
              real maxOverlap = -1.0;
              size_t maxIdx = 0;
              for (size_t j = 0; j < gtBBoxes.size(); ++j) {
                real overlap =
                    jaccardOverlap(predBBoxes[i].second, gtBBoxes[j]);
                if (overlap > maxOverlap) {
                  maxOverlap = overlap;
                  maxIdx = j;
                }
              }
              if (maxOverlap > overlapThreshold_) {
                if (evaluateDifficult_ ||
                    (!evaluateDifficult_ && !gtBBoxes[maxIdx].isDifficult)) {
                  if (!visited[maxIdx]) {
                    allTruePos_[label].push_back(
                        make_pair(predBBoxes[i].first, 1));
                    allFalsePos_[label].push_back(
                        make_pair(predBBoxes[i].first, 0));
                    visited[maxIdx] = true;
                  } else {
                    allTruePos_[label].push_back(
                        make_pair(predBBoxes[i].first, 0));
                    allFalsePos_[label].push_back(
                        make_pair(predBBoxes[i].first, 1));
                  }
                }
              } else {
                allTruePos_[label].push_back(make_pair(predBBoxes[i].first, 0));
                allFalsePos_[label].push_back(
                    make_pair(predBBoxes[i].first, 1));
              }
            }
          }
        }
      }
    }
  }

  real calcMAP() const {
    real mAP = 0.0;
    size_t count = 0;
    for (map<size_t, size_t>::const_iterator it = numPos_.begin();
         it != numPos_.end();
         ++it) {
      size_t label = it->first;
      size_t labelNumPos = it->second;
      if (labelNumPos == 0 || allTruePos_.find(label) == allTruePos_.end())
        continue;
      vector<pair<real, size_t>> labelTruePos = allTruePos_.find(label)->second;
      vector<pair<real, size_t>> labelFalsePos =
          allFalsePos_.find(label)->second;
      // Compute average precision.
      vector<size_t> tpCumSum;
      getAccumulation(labelTruePos, &tpCumSum);
      vector<size_t> fpCumSum;
      getAccumulation(labelFalsePos, &fpCumSum);
      std::vector<real> precision, recall;
      size_t num = tpCumSum.size();
      // Compute Precision.
      for (size_t i = 0; i < num; ++i) {
        CHECK_LE(tpCumSum[i], labelNumPos);
        precision.push_back(static_cast<real>(tpCumSum[i]) /
                            static_cast<real>(tpCumSum[i] + fpCumSum[i]));
        recall.push_back(static_cast<real>(tpCumSum[i]) / labelNumPos);
      }
      // VOC2007 style
      if (apType_ == "11point") {
        vector<real> maxPrecisions(11, 0.0);
        int startIdx = num - 1;
        for (int j = 10; j >= 0; --j)
          for (int i = startIdx; i >= 0; --i) {
            if (recall[i] < j / 10.) {
              startIdx = i;
              if (j > 0) maxPrecisions[j - 1] = maxPrecisions[j];
              break;
            } else {
              if (maxPrecisions[j] < precision[i])
                maxPrecisions[j] = precision[i];
            }
          }
        for (int j = 10; j >= 0; --j) mAP += maxPrecisions[j] / 11;
        ++count;
      } else if (apType_ == "Integral") {
        // Nature integral
        real averagePrecisions = 0.;
        real prevRecall = 0.;
        for (size_t i = 0; i < num; ++i) {
          if (fabs(recall[i] - prevRecall) > 1e-6)
            averagePrecisions += precision[i] * fabs(recall[i] - prevRecall);
          prevRecall = recall[i];
        }
        mAP += averagePrecisions;
        ++count;
      } else {
        LOG(FATAL) << "Unkown ap version: " << apType_;
      }
    }
    if (count != 0) mAP /= count;
    return mAP * 100;
  }

  void getAccumulation(vector<pair<real, size_t>> inPairs,
                       vector<size_t>* accuVec) const {
    std::stable_sort(
        inPairs.begin(), inPairs.end(), sortScorePairDescend<size_t>);
    accuVec->clear();
    size_t sum = 0;
    for (size_t i = 0; i < inPairs.size(); ++i) {
      sum += inPairs[i].second;
      accuVec->push_back(sum);
    }
  }

  std::string getTypeImpl() const { return "detection_map"; }

  real getValueImpl() const { return calcMAP(); }

 private:
  real overlapThreshold_;  // overlap threshold when determining whether matched
  bool evaluateDifficult_;  // whether evaluate difficult ground truth
  size_t backgroundId_;     // class index of background
  std::string apType_;      // how to calculate mAP (Integral or 11point)

  MatrixPtr cpuOutput_;
  MatrixPtr cpuLabel_;

  map<size_t, size_t> numPos_;  // counts of true objects each classification
  map<size_t, vector<pair<real, size_t>>>
      allTruePos_;  // true positive prediction
  map<size_t, vector<pair<real, size_t>>>
      allFalsePos_;  // false positive prediction
};

REGISTER_EVALUATOR(detection_map, DetectionMAPEvaluator);

}  // namespace paddle
