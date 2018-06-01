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

#include "CostLayer.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include "paddle/utils/Logging.h"

#include "paddle/math/SparseMatrix.h"

namespace paddle {

bool CostLayer::init(const LayerMap& layerMap,
                     const ParameterMap& parameterMap) {
  bool ret = Layer::init(layerMap, parameterMap);
  coeff_ = config_.coeff();
  if (!ret) return ret;
  CHECK_GE(inputLayers_.size(), 2UL);
  CHECK_LE(inputLayers_.size(), 3UL);
  if (inputLayers_.size() == 3) {
    weightLayer_ = inputLayers_[2];
  }
  return true;
}

void CostLayer::forward(PassType passType) {
  Layer::forward(passType);

  /* malloc memory for the output_ if necessary */
  int batchSize = getInputValue(*getOutputLayer())->getHeight();
  int size = 1;
  resetOutput(batchSize, size);

  const MatrixPtr& output = getInputValue(*getOutputLayer());
  Argument label = getInput(*getLabelLayer());

  /* get the cost value for each sample*/
  forwardImp(*output, label, *getOutputValue());
  if (weightLayer_) {
    const MatrixPtr& weight = getInputValue(*weightLayer_);
    getOutputValue()->dotMul(*getOutputValue(), *weight);
  }
}

void CostLayer::backward(const UpdateCallback& callback) {
  (void)callback;

  const Argument& output = getInput(*getOutputLayer());
  Argument label = getInput(*getLabelLayer());

  bool support = true;
  if (weightLayer_) {
    support = output.grad->getAbsSum() == 0;
  }

  backwardImp(*output.value, label, *output.grad);

  if (weightLayer_) {
    CHECK(support) << "Weighted cost layer '" << getName()
                   << "' must be the last layer "
                      "connected to the output layer '"
                   << getOutputLayer()->getName() << "'";
    output.grad->rowScale(0, *output.grad, *getInputValue(*weightLayer_));
  }
  if (coeff_ != real(1.0f)) {
    output.grad->add(coeff_, 0);
  }
}

//
// class MultiClassCrossEntropy
//
bool MultiClassCrossEntropy::init(const LayerMap& layerMap,
                                  const ParameterMap& parameterMap) {
  return CostLayer::init(layerMap, parameterMap);
}

void MultiClassCrossEntropy::forwardImp(Matrix& output,
                                        Argument& label,
                                        Matrix& target) {
  target.oneHotCrossEntropy(output, *label.ids);
}

void MultiClassCrossEntropy::backwardImp(Matrix& output,
                                         Argument& label,
                                         Matrix& outputG) {
  outputG.oneHotCrossEntropyBp(output, *label.ids);
}

//
// class MultiClassCrossEntropyWithSelfNorm
//
REGISTER_LAYER(multi_class_cross_entropy_with_selfnorm,
               MultiClassCrossEntropyWithSelfNorm);

bool MultiClassCrossEntropyWithSelfNorm::init(
    const LayerMap& layerMap, const ParameterMap& parameterMap) {
  return CostLayer::init(layerMap, parameterMap);
}

void MultiClassCrossEntropyWithSelfNorm::forwardImp(Matrix& output,
                                                    Argument& label,
                                                    Matrix& target) {
  Matrix::resizeOrCreate(sftMaxSum_, output.getHeight(), 1, false, useGpu_);
  output.rowSum(*sftMaxSum_);
  sftMaxSum_->log2();

  target.oneHotCrossEntropy(output, *label.ids);
  target.add(*sftMaxSum_);

  sftMaxSum_->square2();
  target.add(*sftMaxSum_, config_.softmax_selfnorm_alpha());
}

void MultiClassCrossEntropyWithSelfNorm::backwardImp(Matrix& output,
                                                     Argument& label,
                                                     Matrix& outputG) {
  Matrix::resizeOrCreate(sftMaxSum_, output.getHeight(), 1, false, useGpu_);
  output.rowSum(*sftMaxSum_);

  Matrix::resizeOrCreate(sumInv_, output.getHeight(), 1, false, useGpu_);
  sftMaxSum_->reciprocal2(*sumInv_);

  outputG.oneHotCrossEntropyBp(output, *label.ids);
  outputG.addColumnVector(*sumInv_);

  sftMaxSum_->log2();
  sumInv_->dotMul(*sumInv_, *sftMaxSum_);
  sumInv_->mulScalar(2 * config_.softmax_selfnorm_alpha());

  outputG.addColumnVector(*sumInv_);
}

//
// class SoftBinaryClassCrossEntropy
//
REGISTER_LAYER(soft_binary_class_cross_entropy, SoftBinaryClassCrossEntropy);

bool SoftBinaryClassCrossEntropy::init(const LayerMap& layerMap,
                                       const ParameterMap& parameterMap) {
  return CostLayer::init(layerMap, parameterMap);
}

void SoftBinaryClassCrossEntropy::forwardImp(Matrix& output,
                                             Argument& label,
                                             Matrix& target) {
  Matrix::resizeOrCreate(
      targetPerDim_, output.getHeight(), output.getWidth(), false, useGpu_);

  targetPerDim_->softCrossEntropy(output, *label.value);
  targetPerDim_->rowSum(target);
}

void SoftBinaryClassCrossEntropy::backwardImp(Matrix& output,
                                              Argument& label,
                                              Matrix& outputG) {
  outputG.softCrossEntropyBp(output, *label.value);
}

//
// class SumOfSquaresCostLayer
//

REGISTER_LAYER(square_error, SumOfSquaresCostLayer);

bool SumOfSquaresCostLayer::init(const LayerMap& layerMap,
                                 const ParameterMap& parameterMap) {
  return CostLayer::init(layerMap, parameterMap);
}

void SumOfSquaresCostLayer::forwardImp(Matrix& output,
                                       Argument& label,
                                       Matrix& target) {
  target.sumOfSquares(output, *label.value);
}

void SumOfSquaresCostLayer::backwardImp(Matrix& output,
                                        Argument& label,
                                        Matrix& outputG) {
  outputG.sumOfSquaresBp(output, *label.value);
}

//
// class SmoothL1CostLayer
//

REGISTER_LAYER(smooth_l1, SmoothL1CostLayer);

bool SmoothL1CostLayer::init(const LayerMap& layerMap,
                             const ParameterMap& parameterMap) {
  return CostLayer::init(layerMap, parameterMap);
}

void SmoothL1CostLayer::forwardImp(Matrix& output,
                                   Argument& label,
                                   Matrix& target) {
  MatrixPtr targetCpu, outputCpu, labelCpu;
  if (useGpu_) {
    targetCpu =
        Matrix::create(target.getHeight(), target.getWidth(), false, false);
    outputCpu =
        Matrix::create(output.getHeight(), output.getWidth(), false, false);
    labelCpu = Matrix::create(
        label.value->getHeight(), label.value->getWidth(), false, false);
    targetCpu->copyFrom(target);
    outputCpu->copyFrom(output);
    labelCpu->copyFrom(*label.value);
    targetCpu->smoothL1(*outputCpu, *labelCpu, 1.0);
    target.copyFrom(*targetCpu);
  } else {
    target.smoothL1(output, *label.value, 1.0);
  }
}

void SmoothL1CostLayer::backwardImp(Matrix& output,
                                    Argument& label,
                                    Matrix& outputG) {
  MatrixPtr outputGCpu, outputCpu, labelCpu;
  if (useGpu_) {
    outputGCpu =
        Matrix::create(outputG.getHeight(), outputG.getWidth(), false, false);
    outputCpu =
        Matrix::create(output.getHeight(), output.getWidth(), false, false);
    labelCpu = Matrix::create(
        label.value->getHeight(), label.value->getWidth(), false, false);
    outputGCpu->copyFrom(outputG);
    outputCpu->copyFrom(output);
    labelCpu->copyFrom(*label.value);
    outputGCpu->smoothL1Bp(*outputCpu, *labelCpu, 1.0);
    outputG.copyFrom(*outputGCpu);
  } else {
    outputG.smoothL1Bp(output, *label.value, 1.0);
  }
}

//
// class RankingCost
//
bool RankingCost::init(const LayerMap& layerMap,
                       const ParameterMap& parameterMap) {
  posPairCount_ = 0;
  negPairCount_ = 0;

  bool ret = Layer::init(layerMap, parameterMap);
  if (!ret) return ret;
  CHECK_GE(inputLayers_.size(), 3UL);
  CHECK_LE(inputLayers_.size(), 4UL);
  if (inputLayers_.size() == 4) {
    weightLayer_ = inputLayers_[3];
  }
  return true;
}

void RankingCost::forward(PassType passType) {
  Layer::forward(passType);

  /* malloc memory for the output_ if necessary */
  int batchSize = getInputValue(*getOutputLayer(0))->getHeight();
  int size = 1;
  resizeOutput(batchSize, size);
  Matrix::resizeOrCreate(margin_, batchSize, size, /* trans= */ false, useGpu_);
  MatrixPtr label = getInputValue(*getLabelLayer());
  if (!label) {
    // input label is not in value, try ids
    IVectorPtr idLabel = getInput(*getLabelLayer()).ids;
    CHECK(idLabel) << "label layer has neither value nor ids";
    CHECK_EQ((size_t)batchSize, idLabel->getSize());
    Matrix::resizeOrCreate(
        labelBuf_, batchSize, /*width*/ 1, /*trans*/ false, useGpu_);
    labelBuf_->copyFrom(*idLabel);
    label = labelBuf_;
  }

  MatrixPtr output[] = {getInputValue(*getOutputLayer(0)),
                        getInputValue(*getOutputLayer(1))};
  MatrixPtr target = this->getOutputValue();
  margin_->sub(*output[0], *output[1]);

  // for validation
  size_t height = output[0]->getHeight();
  target->biggerThan(*(output[0]), *(output[1]), *label);
  double total = static_cast<double>(height);
  if (weightLayer_) {
    const MatrixPtr& weight = getInputValue(*weightLayer_);
    target->dotMul(*target, *weight);
    total = weight->getSum();
  }
  double pos = target->getSum();
  posPairCount_ += pos;
  negPairCount_ += (total - pos);

  // forward
  target->logisticRegressionLoss(*margin_, *label);
  if (weightLayer_) {
    const MatrixPtr& weight = getInputValue(*weightLayer_);
    target->dotMul(*target, *weight);
  }
}

void RankingCost::backward(const UpdateCallback& callback) {
  (void)callback;

  MatrixPtr label = getInputValue(*getLabelLayer());
  if (!label) {
    // input label is not in value, but in ids
    // use labelBuf_ (should already resized and copied during forward)
    label = labelBuf_;
  }

  Matrix::resizeOrCreate(
      marginGrad_, label->getHeight(), 1, /* trans= */ false, useGpu_);
  marginGrad_->zeroMem();
  marginGrad_->logisticRegressionLossBp(*margin_, *label);
  if (weightLayer_) {
    const MatrixPtr& weight = getInputValue(*weightLayer_);
    marginGrad_->dotMul(*marginGrad_, *weight);
  }

  getInputGrad(0)->add(*marginGrad_);
  getInputGrad(1)->sub(*marginGrad_);
}

void RankingCost::onPassEnd() {
  double ratio = posPairCount_ / ((negPairCount_ <= 0) ? 1.0 : negPairCount_);
  LOG(INFO) << "calc pos/neg: " << ratio << " pos= " << posPairCount_
            << " neg= " << negPairCount_;

  posPairCount_ = 0;
  negPairCount_ = 0;
}

//
// class LambdaCost
//
REGISTER_LAYER(lambda_cost, LambdaCost);

bool LambdaCost::init(const LayerMap& layerMap,
                      const ParameterMap& parameterMap) {
  truncationSize_ = config_.ndcg_num();
  maxSortSize_ = config_.max_sort_size();
  if (maxSortSize_ != -1) {
    CHECK_GE(maxSortSize_, truncationSize_)
        << "maxSortSize must be greater than or equal to NDCG size!";
  }
  LOG(INFO) << "LambdaRank v1.3, NDCG size = " << truncationSize_
            << ", Max partial sort size = " << maxSortSize_;
  CHECK(!useGpu_) << "LambdaRank supports CPU only!";
  return Layer::init(layerMap, parameterMap);
}

void LambdaCost::forward(PassType passType) {
  Layer::forward(passType);

  /* malloc memory for the output_ if necessary */
  int batchSize = getInputValue(*getOutputLayer())->getHeight();
  resizeOutput(batchSize, 1);

  MatrixPtr score = getInputValue(*getScoreLayer());
  MatrixPtr output = getInputValue(*getOutputLayer());
  MatrixPtr target = this->getOutputValue();

  real* scoreData = score->getData();
  real* outputData = output->getData();
  real* targetData = target->getData();

  auto startPos = getInput(*getOutputLayer()).sequenceStartPositions;
  const int* startPosData = startPos->getData(false);
  size_t batchNum = startPos->getSize() - 1;
  for (size_t i = 0; i < batchNum; ++i) {
    int beginPos = startPosData[i];
    int endPos = startPosData[i + 1];
    real NDCG = calcNDCG(
        outputData + beginPos, scoreData + beginPos, endPos - beginPos);
    for (int j = beginPos; j < endPos; ++j) {
      targetData[j] = NDCG;
    }
  }
}

void LambdaCost::backward(const UpdateCallback& callback) {
  (void)callback;
  MatrixPtr score = getInputValue(*getScoreLayer());
  MatrixPtr output = getInputValue(*getOutputLayer());
  Matrix::resizeOrCreate(marginGrad_,
                         score->getHeight(),
                         1,
                         /* trans= */ false,
                         useGpu_);
  marginGrad_->zeroMem();

  real* gradData = marginGrad_->getData();
  real* scoreData = score->getData();
  real* outputData = output->getData();

  auto startPos = getInput(*getOutputLayer()).sequenceStartPositions;
  const int* startPosData = startPos->getData(false);
  size_t batchNum = startPos->getSize() - 1;

  for (size_t i = 0; i < batchNum; ++i) {
    int beginPos = startPosData[i];
    int endPos = startPosData[i + 1];
    calcGrad(outputData + beginPos,
             scoreData + beginPos,
             gradData + beginPos,
             endPos - beginPos);
  }

  getInputGrad(0)->add(*marginGrad_);
}

void LambdaCost::calcGrad(const real* outputScore,
                          const real* score,
                          real* gradData,
                          int size) {
  CHECK_GE(size, truncationSize_)
      << "Invalid: (Sample num in the same list) < (NDCG truncation num) !";
  int sortSize = maxSortSize_ == -1 ? size : std::min(maxSortSize_, size);

  scorePair_.clear();
  for (int i = 0; i < size; ++i) {
    scorePair_.push_back(std::make_pair(score[i], i));
  }
  if (size <= sortSize) {
    std::sort(scorePair_.begin(),
              scorePair_.end(),
              [](const std::pair<real, int>& a, const std::pair<real, int>& b) {
                return a.first > b.first;
              });
  } else {
    std::partial_sort(
        scorePair_.begin(),
        scorePair_.begin() + sortSize,
        scorePair_.end(),
        [](const std::pair<real, int>& a, const std::pair<real, int>& b) {
          return a.first > b.first;
        });
  }

  real maxDCG = 0;
  for (int i = 0; i < truncationSize_; ++i) {
    maxDCG += (std::pow(2, scorePair_[i].first) - 1) / std::log(i + 2);
  }
  CHECK_GT(maxDCG, 0) << "Invalid: max DCG = 0!";

  for (int i = 0; i < sortSize; ++i) {
    for (int j = i + 1; j < size; ++j) {
      int index_i = scorePair_[i].second;
      int index_j = scorePair_[j].second;
      real score_i = score[index_i];
      real score_j = score[index_j];
      real dcgDif = 0;
      if (j < sortSize) {
        dcgDif = (std::pow(2, score_i) - std::pow(2, score_j)) *
                 (1 / std::log(i + 2) - 1 / std::log(j + 2));
      } else {
        dcgDif =
            (std::pow(2, score_i) - std::pow(2, score_j)) / std::log(i + 2);
      }

      real lambda_ij =
          -std::abs(dcgDif) /
          (1 + std::exp(outputScore[index_i] - outputScore[index_j]));
      gradData[index_i] += lambda_ij / maxDCG;
      gradData[index_j] -= lambda_ij / maxDCG;
    }
  }
}

real LambdaCost::calcNDCG(const real* outputScore,
                          const real* score,
                          int size) {
  CHECK_GE(size, truncationSize_)
      << "Invalid: (Sample num in the same list) < (NDCG truncation num) !";

  outputScorePair_.clear();
  for (int i = 0; i < size; ++i) {
    outputScorePair_.push_back(std::make_pair(outputScore[i], i));
  }
  std::partial_sort(
      outputScorePair_.begin(),
      outputScorePair_.begin() + truncationSize_,
      outputScorePair_.end(),
      [](const std::pair<real, int>& a, const std::pair<real, int>& b) {
        return a.first > b.first;
      });

  real DCG = 0;
  for (int i = 0; i < truncationSize_; ++i) {
    DCG +=
        (std::pow(2, score[outputScorePair_[i].second]) - 1) / std::log(i + 2);
  }

  scoreVec_.resize(size);
  std::copy(score, score + size, scoreVec_.begin());
  real maxDCG = 0;
  std::partial_sort(scoreVec_.begin(),
                    scoreVec_.begin() + truncationSize_,
                    scoreVec_.end(),
                    std::greater<real>());
  for (int i = 0; i < truncationSize_; ++i) {
    maxDCG += (std::pow(2, scoreVec_[i]) - 1) / std::log(i + 2);
  }
  CHECK_GT(maxDCG, 0) << "Invalid: max DCG = 0!";

  return DCG / maxDCG;
}

//
// class MultiBinaryLabelCrossEntropy
//

REGISTER_LAYER(multi_binary_label_cross_entropy, MultiBinaryLabelCrossEntropy);

bool MultiBinaryLabelCrossEntropy::init(const LayerMap& layerMap,
                                        const ParameterMap& parameterMap) {
  return CostLayer::init(layerMap, parameterMap);
}

void MultiBinaryLabelCrossEntropy::forwardImp(Matrix& output,
                                              Argument& label,
                                              Matrix& target) {
  MatrixPtr value = nullptr;
  if (label.ids) {
    CHECK(!label.value);
    value = label.ids->toOneHotSparseMatrix(output.getWidth(), useGpu_);
  } else {
    CHECK(label.value);
    value = label.value;
  }

  if (dynamic_cast<CpuSparseMatrix*>(value.get()) ||
      dynamic_cast<GpuSparseMatrix*>(value.get())) {
    target.multiBinaryLabelCrossEntropy(output, *value);
  } else {
    Matrix::resizeOrCreate(
        targetPerDim_, output.getHeight(), output.getWidth(), false, useGpu_);

    targetPerDim_->binaryLabelCrossEntropy(output, *value);
    targetPerDim_->rowSum(target);
  }
}

void MultiBinaryLabelCrossEntropy::backwardImp(Matrix& output,
                                               Argument& label,
                                               Matrix& outputG) {
  MatrixPtr value = nullptr;
  if (label.ids) {
    CHECK(!value);
    value = label.ids->toOneHotSparseMatrix(output.getWidth(), useGpu_);
  } else {
    CHECK(label.value);
    value = label.value;
  }

  if (dynamic_cast<CpuSparseMatrix*>(value.get()) ||
      dynamic_cast<GpuSparseMatrix*>(value.get())) {
    outputG.multiBinaryLabelCrossEntropyBp(output, *value);
  } else {
    outputG.binaryLabelCrossEntropyBp(output, *value);
  }
}

bool HuberCost::init(const LayerMap& layerMap,
                     const ParameterMap& parameterMap) {
  CostLayer::init(layerMap, parameterMap);
  if (useGpu_) {
    tmpCpuInput_.reserve(inputLayers_.size());
    for (size_t i = 0; i < inputLayers_.size(); i++) {
      tmpCpuInput_.push_back(Argument());
    }
  }
  return true;
}

void HuberCost::forwardImp(Matrix& output, Argument& label, Matrix& cost) {
  if (useGpu_) {
    for (size_t i = 0; i < inputLayers_.size(); i++) {
      tmpCpuInput_[i].resizeAndCopyFrom(
          getInput(i), false, HPPL_STREAM_DEFAULT);
    }
    hl_stream_synchronize(HPPL_STREAM_DEFAULT);
  }
}

//
// Huber loss for robust regression.
//
REGISTER_LAYER(huber_regression, HuberRegressionLoss);

bool HuberRegressionLoss::init(const LayerMap& layerMap,
                               const ParameterMap& parameterMap) {
  HuberCost::init(layerMap, parameterMap);
  delta_ = config_.delta();
  return true;
}

void HuberRegressionLoss::forwardImp(Matrix& output,
                                     Argument& label,
                                     Matrix& target) {
  HuberCost::forwardImp(output, label, target);
  size_t numSamples = target.getHeight();
  size_t dim = output.getWidth();
  CHECK(label.value);
  CHECK_EQ((*label.value).getHeight(), numSamples);
  CHECK_EQ(output.getHeight(), numSamples);
  CHECK_EQ(dim, (*label.value).getWidth());
  CHECK_EQ(target.getWidth(), (size_t)1);

  real* out = useGpu_ ? tmpCpuInput_[0].value->getData() : output.getData();
  real* lbl =
      useGpu_ ? tmpCpuInput_[1].value->getData() : (*label.value).getData();
  std::vector<real> cost(numSamples, 0);
  for (size_t i = 0; i < numSamples; ++i) {
    for (size_t j = 0; j < dim; ++j) {
      int index = i * dim + j;
      real a = std::abs(lbl[index] - out[index]);
      if (a <= delta_)
        cost[i] += a * a / 2;
      else
        cost[i] += delta_ * (a - delta_ / 2);
    }
  }
  target.copyFrom(cost.data(), numSamples);
}

void HuberRegressionLoss::backwardImp(Matrix& output,
                                      Argument& label,
                                      Matrix& outputG) {
  size_t numSamples = output.getHeight();
  size_t dim = output.getWidth();
  real* out = useGpu_ ? tmpCpuInput_[0].value->getData() : output.getData();
  real* lbl =
      useGpu_ ? tmpCpuInput_[1].value->getData() : (*label.value).getData();
  real* grad = useGpu_ ? tmpCpuInput_[0].grad->getData() : outputG.getData();
  for (size_t i = 0; i < numSamples; ++i) {
    for (size_t j = 0; j < dim; ++j) {
      int index = i * dim + j;
      real a = lbl[index] - out[index];
      if (std::abs(a) <= delta_)
        grad[index] += -a;
      else
        grad[index] += a > 0 ? -delta_ : delta_;
    }
  }
  if (useGpu_) outputG.copyFrom(grad, numSamples * dim);
}

//
// Huber loss for robust 2-classes classification
//
REGISTER_LAYER(huber_classification, HuberTwoClassification);

bool HuberTwoClassification::init(const LayerMap& layerMap,
                                  const ParameterMap& parameterMap) {
  return HuberCost::init(layerMap, parameterMap);
}

void HuberTwoClassification::forwardImp(Matrix& output,
                                        Argument& label,
                                        Matrix& target) {
  HuberCost::forwardImp(output, label, target);
  size_t numSamples = target.getHeight();
  CHECK(label.ids);
  CHECK_EQ((*label.ids).getSize(), numSamples);
  CHECK_EQ(output.getHeight(), numSamples);
  CHECK_EQ(output.getWidth(), (size_t)1);
  CHECK_EQ(target.getWidth(), (size_t)1);

  real* out = useGpu_ ? tmpCpuInput_[0].value->getData() : output.getData();
  int* lbl = useGpu_ ? tmpCpuInput_[1].ids->getData() : (*label.ids).getData();
  std::vector<real> cost(numSamples, 0);
  for (size_t i = 0; i < numSamples; ++i) {
    int y = 2 * lbl[i] - 1;
    real a = out[i] * y;
    if (a < -1)
      cost[i] = -4 * a;
    else if (a < 1)
      cost[i] = (1 - a) * (1 - a);
  }
  target.copyFrom(cost.data(), numSamples);
}

void HuberTwoClassification::backwardImp(Matrix& output,
                                         Argument& label,
                                         Matrix& outputG) {
  size_t numSamples = output.getHeight();
  real* out = useGpu_ ? tmpCpuInput_[0].value->getData() : output.getData();
  int* lbl = useGpu_ ? tmpCpuInput_[1].ids->getData() : (*label.ids).getData();
  real* grad = useGpu_ ? tmpCpuInput_[0].grad->getData() : outputG.getData();
  for (size_t i = 0; i < numSamples; ++i) {
    int y = 2 * lbl[i] - 1;
    real a = out[i] * y;
    if (a < -1)
      grad[i] += -4 * y;
    else if (a < 1)
      grad[i] += -2 * (1 - a) * y;
  }
  if (useGpu_) outputG.copyFrom(grad, numSamples);
}
/**
 * This cost layer compute the sum of its input as loss.
 * \f[
 * o(i) = \sum_{j=1}^D y_{ij}
 * \f]
 */
class SumCostLayer : public Layer {
 public:
  explicit SumCostLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override {
    bool ret = Layer::init(layerMap, parameterMap);
    if (!ret) return ret;
    CHECK_EQ(inputLayers_.size(), 1UL);
    return true;
  }

  void forward(PassType passType) override {
    Layer::forward(passType);
    const MatrixPtr& input = getInputValue(0);

    /* malloc memory for the output_ if necessary */
    int batchSize = input->getHeight();
    int size = 1;
    resizeOutput(batchSize, size);
    output_.value->sumRows(*input, /* scaleSum= */ 1, /* scaleDest= */ 0);
  }

  void backward(const UpdateCallback& callback = nullptr) override {
    getInputGrad(0)->add((real)1);
  }
};

REGISTER_LAYER(sum_cost, SumCostLayer);

}  // namespace paddle
