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

#pragma once

#include <float.h>
#include <algorithm>
#include <vector>
#include "paddle/math/Matrix.h"

using std::vector;
using std::pair;
using std::map;

namespace paddle {

template <typename T>
struct BBoxBase {
  BBoxBase(T xMin, T yMin, T xMax, T yMax)
      : xMin(xMin), yMin(yMin), xMax(xMax), yMax(yMax), isDifficult(false) {}

  BBoxBase() {}

  T getWidth() const { return xMax - xMin; }

  T getHeight() const { return yMax - yMin; }

  T getCenterX() const { return (xMin + xMax) / 2; }

  T getCenterY() const { return (yMin + yMax) / 2; }

  T getArea() const { return getWidth() * getHeight(); }

  // coordinate of bounding box
  T xMin;
  T yMin;
  T xMax;
  T yMax;
  // whether difficult object (e.g. object with heavy occlusion is difficult)
  bool isDifficult;
};

struct NormalizedBBox : BBoxBase<real> {
  NormalizedBBox() : BBoxBase<real>() {}
};

enum PermMode { kNCHWToNHWC, kNHWCToNCHW };

/**
 * @brief First permute input maxtrix then append to output matrix
 */
size_t appendWithPermute(const Matrix& inMatrix,
                         size_t height,
                         size_t width,
                         size_t outTotalSize,
                         size_t outOffset,
                         size_t batchSize,
                         Matrix& outMatrix,
                         PermMode permMode);

/**
 * @brief First permute input maxtrix then decompose to output
 */
size_t decomposeWithPermute(const Matrix& inMatrix,
                            size_t height,
                            size_t width,
                            size_t totalSize,
                            size_t offset,
                            size_t batchSize,
                            Matrix& outMatrix,
                            PermMode permMode);

/**
 * @brief Compute jaccard overlap between two bboxes.
 * @param bbox1 The first bbox
 * @param bbox2 The second bbox
 */
real jaccardOverlap(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2);

/**
 * @brief Compute offset parameters between prior bbox and ground truth bbox
 * and variances of prior bbox are considered
 * @param priorBBox Input prior bbox
 * @param priorBBoxVar Variance parameters of prior bbox
 * @param gtBBox Groundtruth bbox
 * @param outVec Output vector
 */
void encodeBBoxWithVar(const NormalizedBBox& priorBBox,
                       const vector<real>& priorBBoxVar,
                       const NormalizedBBox& gtBBox,
                       vector<real>& outVec);

/**
 * @brief Decode prior bbox with offset parameters
 * and variances of prior bbox are considered
 * @param priorBBox Prior bbox to be decoded
 * @param priorBBoxVar Variance parameters of prior bbox
 * @param locPredData Offset parameters
 */
NormalizedBBox decodeBBoxWithVar(const NormalizedBBox& priorBBox,
                                 const vector<real>& priorBBoxVar,
                                 const vector<real>& locPredData);

/**
 * @brief Extract bboxes from prior matrix, the layout is
 * xmin1 | ymin1 | xmax1 | ymax1 | xmin1Var | ymin1Var | xmax1Var | ymax1Var ...
 * @param priorData Matrix of prior value
 * @param numBBoxes Number of bbox to be extracted
 * @param bboxVec Append to the vector
 */
void getBBoxFromPriorData(const real* priorData,
                          const size_t numBBoxes,
                          vector<NormalizedBBox>& bboxVec);

/**
 * @brief Extract labels, scores and bboxes from detection matrix, the layout is
 * imageId | label | score | xmin | ymin | xmax | ymax
 * @param detectData Matrix of detection value
 * @param numBBoxes Number of bbox to be extracted
 * @param labelVec Label of bbox
 * @param scoreVec Score of bbox
 * @param bboxVec Append to the vector
 */
void getBBoxFromDetectData(const real* detectData,
                           const size_t numBBoxes,
                           vector<real>& labelVec,
                           vector<real>& scoreVec,
                           vector<NormalizedBBox>& bboxVec);

/**
 * @brief Extract variances from prior matrix, the layout is
 * xmin1 | ymin1 | xmax1 | ymax1 | xmin1Var | ymin1Var | xmax1Var | ymax1Var ...
 * @param priorData Matrix of prior value
 * @param num Number to be extracted
 * @param varVec Append to the vector
 */
void getBBoxVarFromPriorData(const real* priorData,
                             const size_t num,
                             vector<vector<real>>& varVec);

/**
 * @brief Extract bboxes from label matrix, the layout is
 * class1_1 | xmin1_1 | ymin1_1 | xmax1_1 | ymax1_1 | difficult1_1 | ...
 * @param labelData Matrix of label value
 * @param numBBoxes Number to be extracted
 * @param bboxVec Append to the vector
 */
void getBBoxFromLabelData(const real* labelData,
                          const size_t numBBoxes,
                          vector<NormalizedBBox>& bboxVec);

/**
* @brief Match prior bbox to groundtruth bbox, the strategy is:
1. Find the most overlaped bbox pair (prior and groundtruth)
2. For rest of prior bboxes find the most overlaped groundtruth bbox
* @param priorBBoxes prior bbox
* @param gtBBoxes groundtruth bbox
* @param overlapThreshold Low boundary of overlap (judge whether matched)
* @param matchIndices For each prior bbox, groundtruth bbox index if matched
otherwise -1
* @param matchOverlaps For each prior bbox, overap with all groundtruth bboxes
*/
void matchBBox(const vector<NormalizedBBox>& priorBBoxes,
               const vector<NormalizedBBox>& gtBBoxes,
               real overlapThreshold,
               vector<int>* matchIndices,
               vector<real>* matchOverlaps);

/**
* @brief Generate positive bboxes and negative bboxes,
|positive bboxes|/|negative bboxes| is negPosRatio
* @param priorValue Prior value
* @param numPriorBBoxes Number of prior bbox
* @param gtValue Groundtruth value
* @param gtStartPosPtr Since groundtruth value stored as sequence type,
this parameter indicates start position of each record
* @param seqNum Number of sequence
* @param maxConfScore Classification score for prior bbox, used to mine
negative examples
* @param batchSize Image number
* @param overlapThreshold Low boundary of overap
* @param negOverlapThreshold Upper boundary of overap (judge negative example)
* @param negPosRatio Control number of negative bboxes
* @param matchIndicesVecPtr Save indices of matched prior bbox
* @param negIndicesVecPtr Save indices of negative prior bbox
*/
pair<size_t, size_t> generateMatchIndices(
    const Matrix& priorValue,
    const size_t numPriorBBoxes,
    const Matrix& gtValue,
    const int* gtStartPosPtr,
    const size_t seqNum,
    const vector<vector<real>>& maxConfScore,
    const size_t batchSize,
    const real overlapThreshold,
    const real negOverlapThreshold,
    const size_t negPosRatio,
    vector<vector<int>>* matchIndicesVecPtr,
    vector<vector<int>>* negIndicesVecPtr);

/**
 * @brief Get max confidence score for each prior bbox
 * @param confData Confidence scores, layout is
 * class1 score | class2 score | ... | classN score ...
 * @param batchSize Image number
 * @param numPriorBBoxes Prior bbox number
 * @param numClasses Classes number
 * @param backgroundId Background id
 * @param maxConfScoreVecPtr Ouput
 */
void getMaxConfidenceScores(const real* confData,
                            const size_t batchSize,
                            const size_t numPriorBBoxes,
                            const size_t numClasses,
                            const size_t backgroundId,
                            vector<vector<real>>* maxConfScoreVecPtr);

template <typename T>
bool sortScorePairDescend(const pair<real, T>& pair1,
                          const pair<real, T>& pair2);

template <>
bool sortScorePairDescend(const pair<real, NormalizedBBox>& pair1,
                          const pair<real, NormalizedBBox>& pair2);

/**
 * @brief Do NMS for bboxes to remove duplicated bboxes
 * @param bboxes BBoxes to apply NMS
 * @param confScoreData Confidence scores
 * @param classIdx Class to do NMS
 * @param topK Number to keep
 * @param confThreshold Low boundary of confidence score
 * @param nmsThreshold Threshold of overlap
 * @param numPriorBBoxes Total number of prior bboxes
 * @param numClasses Total class number
 * @param indices Indices of high quality bboxes
 */
void applyNMSFast(const vector<NormalizedBBox>& bboxes,
                  const real* confScoreData,
                  size_t classIdx,
                  size_t topK,
                  real confThreshold,
                  real nmsThreshold,
                  size_t numPriorBBoxes,
                  size_t numClasses,
                  vector<size_t>* indices);

/**
 * @brief Get detection results which satify requirements
 * @param numPriorBBoxes Prior bbox number
 * @param numClasses Class number
 * @param backgroundId Background class
 * @param batchSize Image number
 * @param confThreshold Threshold of class confidence
 * @param nmsTopK Used in NMS operation to keep top k bbox
 * @param nmsThreshold Used in NMS, threshold of overlap
 * @param keepTopK How many bboxes keeped in an image
 * @param allDecodedBBoxes Decoded bboxes for all images
 * @param allDetectionIndices Save detection bbox indices
 */
size_t getDetectionIndices(
    const real* confData,
    const size_t numPriorBBoxes,
    const size_t numClasses,
    const size_t backgroundId,
    const size_t batchSize,
    const real confThreshold,
    const size_t nmsTopK,
    const real nmsThreshold,
    const size_t keepTopK,
    const vector<vector<NormalizedBBox>>& allDecodedBBoxes,
    vector<map<size_t, vector<size_t>>>* allDetectionIndices);

/**
 * @brief Get detection results
 * @param confData Confidence scores
 * @param numPriorBBoxes Prior bbox number
 * @param numClasses Class number
 * @param batchSize Image number
 * @param allIndices Indices of predicted bboxes
 * @param allDecodedBBoxes BBoxes decoded
 * @param out Output matrix
 * image number | label | confidence score | xMin | yMin | xMax | yMax
 */
void getDetectionOutput(const real* confData,
                        const size_t numKept,
                        const size_t numPriorBBoxes,
                        const size_t numClasses,
                        const size_t batchSize,
                        const vector<map<size_t, vector<size_t>>>& allIndices,
                        const vector<vector<NormalizedBBox>>& allDecodedBBoxes,
                        Matrix& out);

NormalizedBBox clipBBox(const NormalizedBBox& bbox);

}  // namespace paddle
