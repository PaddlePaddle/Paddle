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

#include "paddle/gserver/evaluators/Evaluator.h"
#include "paddle/gserver/gradientmachines/NeuralNetwork.h"
#include "paddle/utils/Stat.h"
#include "paddle/utils/StringUtil.h"

DECLARE_int32(trainer_id);

namespace paddle {

void Evaluator::eval(const NeuralNetwork& nn) {
  std::vector<Argument> arguments;
  arguments.reserve(config_.input_layers_size());
  for (const std::string& name : config_.input_layers()) {
    arguments.push_back(nn.getLayer(name)->getOutput());
  }
  SetDevice device(arguments[0].deviceId);
  real score = evalImp(arguments);
  totalScore_ += score;
  updateSamplesNum(arguments);
}
/**
 * @brief classification error Evaluator
 *
 * The config file api is classification_error_evaluator.
 */
class ClassificationErrorEvaluator : public Evaluator {
 public:
  /*
  ClassificationErrorEvaluator() : totalScore2_(0) {}

  virtual void start() {
    Evaluator::start();
    totalScore2_ = 0;
    } */

  virtual void updateSamplesNum(const std::vector<Argument>& arguments) {
    if (3 == arguments.size()) {
      numSamples_ += arguments[2].value->getSum();
    } else {
      numSamples_ += arguments[0].getBatchSize();
    }
  }

  MatrixPtr calcError(std::vector<Argument>& arguments) {
    CHECK_GE(arguments.size(), (size_t)2);
    CHECK_LE(arguments.size(), (size_t)3);
    MatrixPtr& output = arguments[0].value;
    IVectorPtr& label = arguments[1].ids;
    MatrixPtr& multiBinaryLabel = arguments[1].value;  // For multi binary label
    bool supportWeight = (3 == arguments.size()) ? true : false;
    MatrixPtr weight = supportWeight ? arguments[2].value : nullptr;
    if (nullptr == output ||
        (nullptr == label && nullptr == multiBinaryLabel) ||
        (supportWeight && nullptr == weight)) {
      return 0;
    }

    if (label != nullptr) {
      CHECK_EQ(label->getSize(), output->getHeight());
    } else {
      CHECK_EQ(multiBinaryLabel->getHeight(), output->getHeight());
      CHECK_EQ(multiBinaryLabel->getWidth(), output->getWidth());
    }
    if (supportWeight) {
      CHECK_EQ(output->getHeight(), weight->getHeight());
      CHECK_EQ((size_t)1, weight->getWidth());
    }

    const MatrixPtr errorMat = Matrix::create(output->getHeight(),
                                              1,
                                              /* trans= */ false,
                                              useGpu(arguments[0].deviceId));

    errorMat->zeroMem();

    if (label != nullptr) {
      errorMat->classificationError(*output, *label, config_.top_k());
    } else if (dynamic_cast<CpuSparseMatrix*>(multiBinaryLabel.get()) ||
               dynamic_cast<GpuSparseMatrix*>(multiBinaryLabel.get())) {
      errorMat->classificationErrorMulti(
          *output, *multiBinaryLabel, config_.classification_threshold());
    } else {
      errorMat->binaryClassificationError(
          0, *output, *multiBinaryLabel, config_.classification_threshold());
    }

    if (supportWeight) {
      errorMat->dotMul(*errorMat, *weight);
    }
    return errorMat;
  }

  void printStats(std::ostream& os) const {
    if (config_.top_k() == 1) {
      os << config_.name() << "="
         << (numSamples_ ? totalScore_ / numSamples_ : 0);
    } else {
      os << " top_" << config_.top_k()
         << "_error=" << (numSamples_ ? totalScore_ / numSamples_ : 0);
    }
  }

  virtual real evalImp(std::vector<Argument>& arguments) {
    MatrixPtr errorMat = calcError(arguments);
    return errorMat->getSum();
  }

  virtual void distributeEval(ParameterClient2* client) {
    mergeResultsOfAllClients(client);
  }

  // Evaluator interface
 protected:
  std::string getTypeImpl() const { return "classification_error"; }
};

/**
 * @brief sequence classification error Evaluator
 * @note sequence level classification error stats,
 * if any frame in one sequence has error, the sequence is error
 */
class SequenceClassificationErrorEvaluator
    : public ClassificationErrorEvaluator {
 public:
  virtual void updateSamplesNum(const std::vector<Argument>& arguments) {
    numSamples_ += arguments[0].getNumSequences();
  }

  virtual real evalImp(std::vector<Argument>& arguments) {
    auto sequenceStartPositions =
        arguments[0].sequenceStartPositions->getVector(false);
    CHECK(sequenceStartPositions != nullptr);
    const int* starts = sequenceStartPositions->getData();

    MatrixPtr errorMat = calcError(arguments);

    int errCounter = 0;
    CpuVector errorVec(0, nullptr);
    for (size_t i = 0; i < sequenceStartPositions->getSize() - 1; ++i) {
      errorVec.subVecFrom(
          errorMat->getData(), starts[i], starts[i + 1] - starts[i]);
      if (errorVec.getSum() > 0) {
        errCounter += 1;
      }
    }

    return static_cast<real>(errCounter);
  }

  virtual void distributeEval(ParameterClient2* client) {
    mergeResultsOfAllClients(client);
  }

  // Evaluator interface
 protected:
  std::string getTypeImpl() const { return "seq_classification_error"; }
};
REGISTER_EVALUATOR(seq_classification_error,
                   SequenceClassificationErrorEvaluator);
/**
 * @brief sum Evaluator
 * Calculate the sum of output or label
 *
 * The config file api is sum_evaluator.
 */
class SumEvaluator : public Evaluator {
 public:
  SumEvaluator() : cpuLabel_(nullptr), cpuWeight_(nullptr) {}

  virtual void updateSamplesNum(const std::vector<Argument>& arguments) {
    if (2 == arguments.size()) {
      numSamples_ += arguments[1].value->getSum();
    } else {
      numSamples_ += arguments[0].getBatchSize();
    }
  }

  virtual real evalImp(std::vector<Argument>& arguments) {
    REGISTER_TIMER("SumEvaluator");
    CHECK_GE(arguments.size(), (size_t)1);
    CHECK_LE(arguments.size(), (size_t)2);
    bool supportWeight = (2 == arguments.size()) ? true : false;
    if (supportWeight) {
      if (nullptr == arguments[1].value) {
        return 0;
      }
      CHECK_EQ(arguments[1].value->getWidth(), (size_t)1);
    }

    // The sum of output
    if (arguments[0].value) {
      if (supportWeight) {
        CHECK_EQ(arguments[0].value->getHeight(),
                 arguments[1].value->getHeight());
        MatrixPtr tmpMat = Matrix::create(arguments[0].value->getHeight(),
                                          arguments[0].value->getWidth(),
                                          /* trans= */ false,
                                          arguments[0].value->useGpu());
        tmpMat->copyFrom(*arguments[0].value);
        tmpMat->rowScale(0, *tmpMat, *arguments[1].value);
        return tmpMat->getSum();
      } else {
        return arguments[0].value->getSum();
      }
      // The sum of label
    } else if (arguments[0].ids) {
      size_t insNum = arguments[0].ids->getSize();
      IVectorPtr label = arguments[0].ids;
      MatrixPtr weight = supportWeight ? arguments[1].value : nullptr;
      if (dynamic_cast<GpuIVector*>(label.get())) {
        IVector::resizeOrCreate(cpuLabel_, insNum, false);
        cpuLabel_->copyFrom(*arguments[0].ids);

        if (supportWeight) {
          CHECK_EQ(insNum, arguments[1].value->getHeight());
          Matrix::resizeOrCreate(cpuWeight_, insNum, (size_t)1, false, false);
          cpuWeight_->copyFrom(*arguments[1].value);
        }

        label = cpuLabel_;
        weight = cpuWeight_;
      }

      if (supportWeight) {
        real score = 0.0;
        int* labelD = label->getData();
        real* weightD = weight->getData();
        for (size_t i = 0; i < insNum; ++i) {
          score += (labelD[i] * weightD[i]);
        }
        return score;
      } else {
        return label->getSum();
      }
    } else {
      return 0;
    }
  }

  virtual void distributeEval(ParameterClient2* client) {
    mergeResultsOfAllClients(client);
  }

 private:
  IVectorPtr cpuLabel_;
  MatrixPtr cpuWeight_;

  // Evaluator interface
 protected:
  std::string getTypeImpl() const { return "sum"; }
};
/**
 * @brief column sum Evaluator
 * @note column sum for the colIdx-th column *
 * - colIdx = 0: the 0-th column.
 * - colIdx > 0: the colIdx-th column.
 * - colIdx < 0: the last colIdx-th column.
 *
 * The config file api is column_sum_evaluator.
 *
 */
class ColumnSumEvaluator : public Evaluator {
 public:
  explicit ColumnSumEvaluator(int32_t colIdx)
      : colIdx_(colIdx), colNum_(0), sum_(nullptr) {}

  virtual void start() {
    Evaluator::start();
    if (nullptr != sum_) {
      sum_->zeroMem();
    }
  }

  virtual void updateSamplesNum(const std::vector<Argument>& arguments) {
    if (2 == arguments.size()) {
      numSamples_ += arguments[1].value->getSum();
    } else {
      numSamples_ += arguments[0].getBatchSize();
    }
  }

  virtual real evalImp(std::vector<Argument>& arguments) {
    REGISTER_TIMER("ColumnSumEvaluator");
    CHECK_GE(arguments.size(), (size_t)1);
    CHECK_LE(arguments.size(), (size_t)2);
    bool supportWeight = (2 == arguments.size()) ? true : false;
    if (nullptr == arguments[0].value ||
        (supportWeight && nullptr == arguments[1].value)) {
      return 0;
    }

    size_t insNum = arguments[0].value->getHeight();
    size_t colNum = arguments[0].value->getWidth();
    if (nullptr == sum_) {
      sum_ = Matrix::create((size_t)1, colNum, false, /* useGpu */ false);
      colNum_ = colNum;
      sum_->zeroMem();
    } else {
      CHECK_EQ(colNum, sum_->getWidth());
    }

    if (supportWeight) {
      CHECK_EQ(insNum, arguments[1].value->getHeight());
      CHECK_EQ((size_t)1, arguments[1].value->getWidth());
      MatrixPtr tmpMat = Matrix::create(insNum, colNum);
      if (arguments[0].value->useGpu()) {
        tmpMat->copyFrom(*arguments[0].value);
      }
      if (!arguments[1].value->useGpu()) {
        if (!arguments[0].value->useGpu()) {
          tmpMat->rowScale(0, *arguments[0].value, *arguments[1].value);
        } else {
          tmpMat->rowScale(0, *tmpMat, *arguments[1].value);
        }
      } else {
        MatrixPtr tmp2 = Matrix::create(insNum, 1);
        tmp2->copyFrom(*arguments[1].value);
        if (!arguments[0].value->useGpu()) {
          tmpMat->rowScale(0, *arguments[0].value, *tmp2);
        } else {
          tmpMat->rowScale(0, *tmpMat, *tmp2);
        }
      }
      sum_->accumulateColSum(*tmpMat);
    } else {
      if (!arguments[0].value->useGpu()) {
        sum_->accumulateColSum(*arguments[0].value);
      } else {
        MatrixPtr tmpMat = Matrix::create(insNum, colNum);
        tmpMat->copyFrom(*arguments[0].value);
        sum_->accumulateColSum(*tmpMat);
      }
    }
    return 0;
  }

  virtual void printStats(std::ostream& os) const {
    CHECK(colIdx_ + (int32_t)colNum_ >= 0 && colIdx_ - (int32_t)colNum_ < 0)
        << "column index [" << colIdx_ << "] out of range [-" << colNum_ << ", "
        << colNum_ << ")";
    size_t colIdx = 0;
    if (colIdx_ >= 0) {
      colIdx = colIdx_;
    } else {
      colIdx = colNum_ + colIdx_;
    }
    os << config_.name() << "="
       << (numSamples_ ? sum_->getElement(0, colIdx) / numSamples_ : 0);
  }

  void distributeEval(ParameterClient2* client) {
    client->reduce(
        sum_->getData(), sum_->getData(), colNum_, FLAGS_trainer_id, 0);
    client->reduce(&numSamples_, &numSamples_, 1, FLAGS_trainer_id, 0);
  }

 private:
  int32_t colIdx_;
  size_t colNum_;
  MatrixPtr sum_; /* cpu matrix */

  // Evaluator interface
 protected:
  std::string getTypeImpl() const {
    if (colIdx_ == -1)
      return "last-column-sum";
    else
      return "column-sum";
  }
};

void AucEvaluator::start() {
  Evaluator::start();
  memset(statPos_, 0, sizeof(statPos_));
  memset(statNeg_, 0, sizeof(statNeg_));
}

real AucEvaluator::evalImp(std::vector<Argument>& arguments) {
  REGISTER_TIMER("AucEvaluator");
  CHECK_GE(arguments.size(), (size_t)2);
  CHECK_LE(arguments.size(), (size_t)3);
  MatrixPtr output = arguments[0].value;
  IVectorPtr label = arguments[1].ids;
  MatrixPtr labelval = arguments[1].value;
  bool supportWeight = (3 == arguments.size()) ? true : false;
  MatrixPtr weight = supportWeight ? arguments[2].value : nullptr;

  if (nullptr == output || (supportWeight && nullptr == weight)) {
    return 0;
  }
  size_t insNum = output->getHeight();
  size_t outputDim = output->getWidth();
  // Copy label from value to a vector.
  if (nullptr == label && nullptr != labelval) {
    // label width is 1
    CHECK_EQ(1U, labelval->getWidth());
    VectorPtr vec =
        Vector::create(labelval->getData(), insNum, output->useGpu());
    label = vec->castToInt();
  }

  CHECK_EQ(insNum, label->getSize());
  if (supportWeight) {
    CHECK_EQ(insNum, weight->getHeight());
    CHECK_EQ((size_t)1, weight->getWidth());
  }

  CHECK(colIdx_ + (int32_t)outputDim >= 0 && colIdx_ - (int32_t)outputDim < 0)
      << "column index [" << colIdx_ << "] out of range [-" << outputDim << ", "
      << outputDim << ")";
  realColumnIdx_ = 0;
  if (colIdx_ >= 0) {
    realColumnIdx_ = colIdx_;
  } else {
    realColumnIdx_ = outputDim + colIdx_;
  }

  if (dynamic_cast<GpuMatrix*>(output.get())) {
    Matrix::resizeOrCreate(cpuOutput_,
                           insNum,
                           outputDim,
                           /* trans=*/false,
                           /* useGpu=*/false);
    cpuOutput_->copyFrom(*output);
    IVector::resizeOrCreate(cpuLabel_, insNum, false);
    cpuLabel_->copyFrom(*label);

    if (supportWeight) {
      Matrix::resizeOrCreate(cpuWeight_, insNum, (size_t)1, false, false);
      cpuWeight_->copyFrom(*weight);
    }

    output = cpuOutput_;
    label = cpuLabel_;
    weight = cpuWeight_;
  }

  real* outputD = output->getData();
  int* labelD = label->getData();
  real* weightD = supportWeight ? weight->getData() : nullptr;
  size_t pos = realColumnIdx_;

  for (size_t i = 0; i < insNum; ++i) {
    real value = outputD[pos];
    uint32_t binIdx = static_cast<uint32_t>(value * kBinNum_);
    CHECK(binIdx <= kBinNum_) << "bin index [" << binIdx
                              << "] out of range, predict value[" << value
                              << "]";
    real w = supportWeight ? weightD[i] : 1.0;
    if (labelD[i] == kNegativeLabel_) {
      statNeg_[binIdx] += w;
    } else {
      statPos_[binIdx] += w;
    }
    pos += outputDim;
  }
  return 0;
}

void AucEvaluator::distributeEval(ParameterClient2* client) {
  client->reduce(statPos_, statPos_, kBinNum_ + 1, FLAGS_trainer_id, 0);
  client->reduce(statNeg_, statNeg_, kBinNum_ + 1, FLAGS_trainer_id, 0);
}

double AucEvaluator::calcAuc() const {
  double totPos = 0.0;
  double totNeg = 0.0;
  double totPosPrev = 0.0;
  double totNegPrev = 0.0;
  double auc = 0.0;

  int64_t idx = kBinNum_;
  while (idx >= 0) {
    totPosPrev = totPos;
    totNegPrev = totNeg;
    totPos += statPos_[idx];
    totNeg += statNeg_[idx];
    auc += trapezoidArea(totNeg, totNegPrev, totPos, totPosPrev);
    --idx;
  }

  if (totPos > 0.0 && totNeg > 0.0) {
    return auc / totPos / totNeg;
  } else {
    return 0.0;
  }
}

real AucEvaluator::getValueImpl() const { return calcAuc(); }

std::string AucEvaluator::getTypeImpl() const {
  if (colIdx_ == -1) {
    return "last-column-auc";
  } else {
    return "auc";
  }
}

// class RankAucEvaluator
REGISTER_EVALUATOR(rankauc, RankAucEvaluator);

void RankAucEvaluator::start() { Evaluator::start(); }
void RankAucEvaluator::updateSamplesNum(
    const std::vector<Argument>& arguments) {
  numSamples_ += arguments[0].getNumSequences();
}
real RankAucEvaluator::evalImp(std::vector<Argument>& arguments) {
  CHECK_GE(arguments.size(), 2U);
  CHECK_LE(arguments.size(), 3U);
  double batchAuc = 0.0;
  output_ = arguments[0].value;
  click_ = arguments[1].value;
  size_t batchSize = output_->getHeight();
  CHECK(!output_->useGpu()) << "RankAUC evaluator does not support GPU!";

  if (arguments.size() == 3U) {
    pv_ = arguments[2].value;
  } else {
    Matrix::resizeOrCreate(pv_, batchSize, 1, false, false);
    std::fill(pv_->getData(), pv_->getData() + batchSize, 1.0);
  }

  real* outputData = output_->getData();
  real* clickData = click_->getData();
  real* pvData = pv_->getData();

  auto startPos = arguments[0].sequenceStartPositions->getVector(false);
  const int* startPosData = startPos->getData();
  size_t batchNum = startPos->getSize() - 1;
  for (size_t i = 0; i < batchNum; ++i) {
    int beginPos = startPosData[i];
    int endPos = startPosData[i + 1];
    batchAuc += calcRankAuc(outputData + beginPos,
                            clickData + beginPos,
                            pvData + beginPos,
                            endPos - beginPos);
  }
  return batchAuc;
}

double RankAucEvaluator::calcRankAuc(real* outputData,
                                     real* clickData,
                                     real* pvData,
                                     size_t size) {
  outputPair_.clear();
  for (size_t i = 0; i < size; ++i) {
    outputPair_.push_back(std::make_pair(outputData[i], i));
  }
  std::sort(outputPair_.begin(),
            outputPair_.end(),
            [](const std::pair<real, int>& a, const std::pair<real, int>& b) {
              return a.first > b.first;
            });
  double aucTmp = 0.0;
  double clickSum = 0.0;
  double oldClickSum = 0.0;
  double noClick = 0.0;
  double noClickSum = 0.0;

  double lastScore = outputPair_[0].first + 1.0;
  for (size_t i = 0; i < size; ++i) {
    if (lastScore != outputPair_[i].first) {
      aucTmp += (clickSum + oldClickSum) * noClick / 2.0;
      oldClickSum = clickSum;
      noClick = 0.0;
      lastScore = outputPair_[i].first;
    }
    size_t id = outputPair_[i].second;
    noClick += pvData[id] - clickData[id];
    noClickSum += noClick;
    clickSum += clickData[id];
  }
  aucTmp += (clickSum + oldClickSum) * noClick / 2.0;
  return (clickSum * noClickSum) == 0.0 ? 0.0
                                        : aucTmp / (clickSum * noClickSum);
}

std::string RankAucEvaluator::getTypeImpl() const { return "rankauc"; }

// class PrecisionRecallEvaluator
REGISTER_EVALUATOR(precision_recall, PrecisionRecallEvaluator);

void PrecisionRecallEvaluator::start() {
  Evaluator::start();
  statsInfo_.clear();
  values_.clear();
}

real PrecisionRecallEvaluator::evalImp(std::vector<Argument>& arguments) {
  REGISTER_TIMER("PrecisionRecallEvaluator");
  CHECK_GE(arguments.size(), (size_t)2);
  CHECK_LE(arguments.size(), (size_t)3);
  MatrixPtr output = arguments[0].value;
  IVectorPtr label = arguments[1].ids;
  MatrixPtr multiBinaryLabel = arguments[1].value;
  bool supportWeight = (3 == arguments.size()) ? true : false;
  MatrixPtr weight = supportWeight ? arguments[2].value : nullptr;
  if (nullptr == output || (nullptr == label && nullptr == multiBinaryLabel) ||
      (supportWeight && nullptr == weight)) {
    return 0;
  }

  size_t insNum = output->getHeight();
  size_t outputDim = output->getWidth();
  if (label != nullptr) {
    CHECK_EQ(insNum, label->getSize());
  } else {
    CHECK_EQ(insNum, multiBinaryLabel->getHeight());
    CHECK_EQ(outputDim, multiBinaryLabel->getWidth());
  }
  if (supportWeight) {
    CHECK_EQ(insNum, weight->getHeight());
    CHECK_EQ((size_t)1, weight->getWidth());
  }

  if (statsInfo_.size() != outputDim) {
    statsInfo_.clear();
    statsInfo_.resize(outputDim);
  }

  isMultiBinaryLabel_ = (nullptr == label) ? true : false;
  if (label != nullptr) {
    if (dynamic_cast<GpuMatrix*>(output.get())) {
      Matrix::resizeOrCreate(cpuOutput_, insNum, outputDim, false, false);
      cpuOutput_->copyFrom(*output);
      IVector::resizeOrCreate(cpuLabel_, insNum, false);
      cpuLabel_->copyFrom(*label);
      if (supportWeight) {
        Matrix::resizeOrCreate(cpuWeight_, insNum, (size_t)1, false, false);
        cpuWeight_->copyFrom(*weight);
      }

      output = cpuOutput_;
      label = cpuLabel_;
      weight = cpuWeight_;
    }
    calcStatsInfo(output, label, weight);
  } else {
    // Not support GPU for multi binary labels
    CHECK(dynamic_cast<CpuSparseMatrix*>(multiBinaryLabel.get()));
    calcStatsInfoMulti(output, multiBinaryLabel, weight);
  }
  return 0;
}

void PrecisionRecallEvaluator::printStats(std::ostream& os) const {
  PrintStatsInfo info;
  bool containMacroMicroInfo = getStatsInfo(&info);
  os << "positive_label=" << config_.positive_label()
     << " precision=" << info.precision << " recall=" << info.recall
     << " F1-score=" << info.f1;
  if (containMacroMicroInfo) {
    os << "macro-average-precision=" << info.macroAvgPrecision
       << " macro-average-recall=" << info.macroAvgRecall
       << " macro-average-F1-score=" << info.macroAvgF1Score;
    if (!isMultiBinaryLabel_) {
      // precision and recall are equal in this case
      os << " micro-average-precision=" << info.microAvgPrecision;
    } else {
      os << " micro-average-precision=" << info.microAvgPrecision
         << " micro-average-recall=" << info.microAvgRecall
         << " micro-average-F1-score=" << info.microAvgF1Score;
    }
  }
}

void PrecisionRecallEvaluator::calcStatsInfo(const MatrixPtr& output,
                                             const IVectorPtr& label,
                                             const MatrixPtr& weight) {
  size_t insNum = output->getHeight();
  size_t dim = output->getWidth();
  real* outputD = output->getData();
  int* labelD = label->getData();
  real* weightD = (weight != nullptr) ? weight->getData() : nullptr;
  for (size_t i = 0; i < insNum; ++i) {
    CHECK_GE(labelD[i], 0);
    CHECK_LT((size_t)labelD[i], dim);
    size_t maxIdx = 0;
    real maxValue = outputD[i * dim];
    for (size_t j = 1; j < dim; ++j) {
      size_t idx = i * dim + j;
      if (maxValue < outputD[idx]) {
        maxIdx = j;
        maxValue = outputD[idx];
      }
    }

    real w = (weightD != nullptr) ? weightD[i] : 1.0;
    if (maxIdx == (size_t)labelD[i]) {
      statsInfo_[maxIdx].TP += w;  // true positive for labelD[i]
      // true negative for all labels except for labelD[i]
      for (size_t j = 0; j < dim; ++j) {
        statsInfo_[j].TN += w;
      }
      statsInfo_[maxIdx].TN -= w;
    } else {
      statsInfo_[labelD[i]].FN += w;  // false negative for labelD[i]
      statsInfo_[maxIdx].FP += w;     // false positive for maxIdx
      // true negatives for all labels except for maxIdx and labelD[i]
      for (size_t j = 0; j < dim; ++j) {
        statsInfo_[j].TN += w;
      }
      statsInfo_[maxIdx].TN -= w;
      statsInfo_[labelD[i]].TN -= w;
    }
  }
}

void PrecisionRecallEvaluator::calcStatsInfoMulti(const MatrixPtr& output,
                                                  const MatrixPtr& label,
                                                  const MatrixPtr& weight) {
  size_t insNum = output->getHeight();
  size_t dim = output->getWidth();
  real* outputD = output->getData();
  auto labelD = dynamic_cast<CpuSparseMatrix*>(label.get());
  real* weightD = (weight != nullptr) ? weight->getData() : nullptr;
  real threshold = config_.classification_threshold();
  for (size_t i = 0; i < insNum; ++i) {
    for (size_t j = 0; j < dim; ++j) {
      real w = (weightD != nullptr) ? weightD[i] : 1.0;
      size_t idx = i * dim + j;
      if (outputD[idx] < threshold) {
        statsInfo_[j].TN += w;  // true negative
      } else {
        statsInfo_[j].FP += w;  // false positive
      }
    }

    const int* cols = labelD->getRowCols(i);
    for (size_t j = 0; j < labelD->getColNum(i); ++j) {
      CHECK_LT(size_t(cols[j]), dim);
      real w = (weightD != nullptr) ? weightD[i] : 1.0;
      size_t idx = i * dim + cols[j];
      if (outputD[idx] < threshold) {
        statsInfo_[cols[j]].FN += w;  // false negative
        statsInfo_[cols[j]].TN -= w;  // true negative
      } else {
        statsInfo_[cols[j]].TP += w;  // true positive
        statsInfo_[cols[j]].FP -= w;  // false positive
      }
    }
  }
}

void PrecisionRecallEvaluator::storeLocalValues() const {
  if (this->values_.size() == 0) {
    PrintStatsInfo info;
    bool containMacroMicroInfo = getStatsInfo(&info);
    values_["precision"] = info.precision;
    values_["recal"] = info.recall;
    values_["F1-score"] = info.f1;
    if (containMacroMicroInfo) {
      values_["macro-average-precision"] = info.macroAvgPrecision;
      values_["macro-average-recall"] = info.macroAvgRecall;
      values_["macro-average-F1-score"] = info.macroAvgF1Score;
      if (!isMultiBinaryLabel_) {
        // precision and recall are equal in this case
        values_["micro-average-precision"] = info.microAvgPrecision;
      } else {
        values_["micro-average-precision"] = info.microAvgPrecision;
        values_["micro-average-recall"] = info.microAvgRecall;
        values_["micro-average-F1-score"] = info.microAvgF1Score;
      }
    }
  }
}

void PrecisionRecallEvaluator::getNames(std::vector<std::string>* names) {
  this->storeLocalValues();
  names->reserve(this->values_.size());
  for (auto it = this->values_.begin(); it != this->values_.end(); ++it) {
    names->push_back(this->config_.name() + "." + it->first);
  }
}

real PrecisionRecallEvaluator::getValue(const std::string& name,
                                        Error* err) const {
  this->storeLocalValues();
  std::vector<std::string> buffers;
  paddle::str::split(name, '.', &buffers);
  auto it = this->values_.find(buffers[buffers.size() - 1]);
  if (it == this->values_.end()) {  // not found
    *err = Error("No such key %s", name.c_str());
    return .0f;
  }

  return it->second;
}

std::string PrecisionRecallEvaluator::getType(const std::string& name,
                                              Error* err) const {
  this->getValue(name, err);
  if (!err->isOK()) {
    return "";
  }
  return "precision_recall";
}

void PrecisionRecallEvaluator::distributeEval(ParameterClient2* client) {
  size_t size = 4 * statsInfo_.size();
  double* buf = new double[size];
  for (size_t i = 0; i < statsInfo_.size(); ++i) {
    buf[4 * i + 0] = statsInfo_[i].TP;
    buf[4 * i + 1] = statsInfo_[i].TN;
    buf[4 * i + 2] = statsInfo_[i].FP;
    buf[4 * i + 3] = statsInfo_[i].FN;
  }
  client->reduce(buf, buf, size, FLAGS_trainer_id, 0);
  for (size_t i = 0; i < statsInfo_.size(); ++i) {
    statsInfo_[i].TP = buf[4 * i + 0];
    statsInfo_[i].TN = buf[4 * i + 1];
    statsInfo_[i].FP = buf[4 * i + 2];
    statsInfo_[i].FN = buf[4 * i + 3];
  }
  delete[] buf;
}

bool PrecisionRecallEvaluator::getStatsInfo(
    PrecisionRecallEvaluator::PrintStatsInfo* info) const {
  int label = config_.positive_label();
  if (label != -1) {
    CHECK(label >= 0 && label < (int)statsInfo_.size())
        << "positive_label [" << label << "] should be in range [0, "
        << statsInfo_.size() << ")";
    info->precision = calcPrecision(statsInfo_[label].TP, statsInfo_[label].FP);
    info->recall = calcRecall(statsInfo_[label].TP, statsInfo_[label].FN);
    info->f1 = calcF1Score(info->precision, info->recall);
    return false;
  }

  // micro average method: precision = (TP1+TP2)/(TP1+FP1+TP2+FP2)
  // macro average method: precision = (precision1+precision2)/2
  double microTotalTP = 0;
  double microTotalFP = 0;
  double microTotalFN = 0;
  info->macroAvgPrecision = 0;
  info->macroAvgRecall = 0;
  size_t numLabels = statsInfo_.size();
  for (size_t i = 0; i < numLabels; ++i) {
    microTotalTP += statsInfo_[i].TP;
    microTotalFP += statsInfo_[i].FP;
    microTotalFN += statsInfo_[i].FN;
    info->macroAvgPrecision +=
        calcPrecision(statsInfo_[i].TP, statsInfo_[i].FP);
    info->macroAvgRecall += calcRecall(statsInfo_[i].TP, statsInfo_[i].FN);
  }
  info->macroAvgPrecision /= numLabels;
  info->macroAvgRecall /= numLabels;
  info->macroAvgF1Score =
      calcF1Score(info->macroAvgPrecision, info->macroAvgRecall);

  info->microAvgPrecision = calcPrecision(microTotalTP, microTotalFP);
  info->microAvgRecall = calcPrecision(microTotalTP, microTotalFN);
  info->microAvgF1Score =
      calcF1Score(info->microAvgPrecision, info->microAvgRecall);
  return true;
}

REGISTER_EVALUATOR(pnpair, PnpairEvaluator);
void PnpairEvaluator::start() {
  Evaluator::start();
  memset(pairArray_, 0, sizeof(pairArray_));
  predictArray_.clear();
}

real PnpairEvaluator::evalImp(std::vector<Argument>& arguments) {
  CHECK_GE(arguments.size(), 3UL);
  CHECK_LE(arguments.size(), 4UL);
  MatrixPtr output = arguments[0].value;
  IVectorPtr label = arguments[1].ids;
  IVectorPtr info = arguments[2].ids;
  bool supportWeight = (4 == arguments.size()) ? true : false;
  MatrixPtr weight = supportWeight ? arguments[3].value : nullptr;
  if (nullptr == output || nullptr == label ||
      (supportWeight && nullptr == weight)) {
    return 0;
  }
  size_t height = output->getHeight();
  size_t width = output->getWidth();
  CHECK_EQ(height, label->getSize());
  CHECK_EQ(height, info->getSize());
  if (supportWeight) {
    CHECK_EQ(height, weight->getHeight());
    CHECK_EQ((size_t)1, weight->getWidth());
  }

  if (dynamic_cast<GpuMatrix*>(output.get())) {
    Matrix::resizeOrCreate(cpuOutput_, height, width, false, false);
    IVector::resizeOrCreate(cpuLabel_, height, false);
    IVector::resizeOrCreate(cpuInfo_, height, false);
    cpuOutput_->copyFrom(*output);
    cpuLabel_->copyFrom(*label);
    cpuInfo_->copyFrom(*info);

    output = cpuOutput_;
    label = cpuLabel_;
    info = cpuInfo_;

    if (supportWeight) {
      Matrix::resizeOrCreate(cpuWeight_, height, (size_t)1, false, false);
      cpuWeight_->copyFrom(*weight);
      weight = cpuWeight_;
    }
  }

  real* outputs = output->getData();
  int* labels = label->getData();
  int* infos = info->getData();
  real* weights = supportWeight ? weight->getData() : nullptr;
  for (size_t i = 0; i < output->getHeight(); i++) {
    real y1 = outputs[i * width + (width - 1)];
    real w = supportWeight ? weights[i] : 1.0;
    predictArray_.push_back(PredictionResult(y1, labels[i], infos[i], w));
  }
  return 0;
}

void PnpairEvaluator::stat(size_t start,
                           size_t end,
                           PredictionResult* answers,
                           double& pos,
                           double& neg,
                           double& spe) {
  for (size_t i = start; i < end; i++) {
    for (size_t j = i + 1; j < end; j++) {
      CHECK_EQ(answers[i].queryid, answers[j].queryid);
      // The pair weight is the mean of the two samples' weight
      double weight = (answers[i].weight + answers[j].weight) / 2.0;
      if (answers[i].label != answers[j].label) {
        if ((answers[i].out > answers[j].out &&
             answers[i].label > answers[j].label) ||
            (answers[i].out < answers[j].out &&
             answers[i].label < answers[j].label)) {
          pos += weight;
        } else if ((answers[i].out > answers[j].out &&
                    answers[i].label < answers[j].label) ||
                   (answers[i].out < answers[j].out &&
                    answers[i].label > answers[j].label)) {
          neg += weight;
        } else {
          spe += weight;
        }
      }
    }
  }
}

void PnpairEvaluator::calc(std::vector<PredictionResult>& predictArray) {
  std::sort(predictArray.begin(),
            predictArray.end(),
            [](const PredictionResult& x, const PredictionResult& y) {
              return x.queryid < y.queryid;
            });

  double pos = 0;
  double neg = 0;
  double special = 0;
  auto start = predictArray.begin();
  while (start != predictArray.end()) {
    auto end = std::find_if(
        start + 1, predictArray.end(), [=](const PredictionResult& x) {
          return x.queryid != start->queryid;
        });
    CHECK(end != start);
    stat(start - predictArray.begin(),
         end - predictArray.begin(),
         predictArray.data(),
         pos,
         neg,
         special);

    start = end;
  }

  pairArray_[0] += pos;
  pairArray_[1] += neg;

  LOG(INFO) << " calc total pos pair: " << pos
            << " calc total neg pair: " << neg
            << " calc total special pair: " << special;
}

std::string PnpairEvaluator::getTypeImpl() const { return "pnpair"; }

ClassRegistrar<Evaluator> Evaluator::registrar_;
Evaluator* Evaluator::create(const EvaluatorConfig& config) {
  Evaluator* evaluator = registrar_.createByType(config.type());
  evaluator->init(config);
  return evaluator;
}

REGISTER_EVALUATOR(classification_error, ClassificationErrorEvaluator);
REGISTER_EVALUATOR(sum, SumEvaluator);
static InitFunction __reg_type_auc_sum__([]() {
  Evaluator::registrar_.registerClass(
      "last-column-sum", [] { return new ColumnSumEvaluator(-1); });
  Evaluator::registrar_.registerClass("last-column-auc",
                                      [] { return new AucEvaluator(-1); });
});

/**
 * @brief print value of each layer.
 *
 * The config file api is value_printer_evaluator.
 */
class ValuePrinter : public NotGetableEvaluator {
 public:
  virtual void eval(const NeuralNetwork& nn) {
    for (const std::string& name : config_.input_layers()) {
      nn.getLayer(name)->getOutput().printValueString(LOG(INFO),
                                                      "layer=" + name + " ");
    }
  }

  virtual void updateSamplesNum(const std::vector<Argument>& arguments) {}

  virtual real evalImp(std::vector<Argument>& arguments) { return 0; }
};
REGISTER_EVALUATOR(value_printer, ValuePrinter);

/**
 * @brief print gradient of each layer.
 *
 * The config file api is gradient_printer_evaluator.
 */
class GradientPrinter : public NotGetableEvaluator {
 public:
  virtual void eval(const NeuralNetwork& nn) {
    for (const std::string& name : config_.input_layers()) {
      const Argument& argu = nn.getLayer(name)->getOutput();
      if (argu.grad) {
        std::ostringstream os;
        argu.grad->print(os);
        LOG(INFO) << "layer=" << name << " grad matrix:\n" << os.str();
      }
    }
  }

  virtual void updateSamplesNum(const std::vector<Argument>& arguments) {}

  virtual real evalImp(std::vector<Argument>& arguments) { return 0; }
};
REGISTER_EVALUATOR(gradient_printer, GradientPrinter);
/**
 * @brief print row max id vctor of each layer
 *
 * The config file api is maxid_printer_evaluator.
 */
class MaxIdPrinter : public NotGetableEvaluator {
 private:
  IVectorPtr maxIds_;
  MatrixPtr maxValues_;

 public:
  MaxIdPrinter() {}

  virtual void eval(const NeuralNetwork& nn) {
    for (const std::string& name : config_.input_layers()) {
      const Argument& argu = nn.getLayer(name)->getOutput();
      if (argu.value) {
        size_t height = argu.value->getHeight();
        size_t width = config_.num_results();
        IVector::resizeOrCreate(maxIds_, height * width, false);
        Matrix::resizeOrCreate(maxValues_, height, width, false);
        argu.value->rowMax(*maxIds_, *maxValues_);
        std::ostringstream os;
        int* ids = maxIds_->getData();
        real* values = maxValues_->getData();
        for (size_t i = 0; i < height; ++i) {
          for (size_t j = 0; j < width; ++j) {
            size_t pos = i * width + j;
            os << ids[pos] << " : " << values[pos] << ", ";
          }
          os << std::endl;
        }
        LOG(INFO) << "layer=" << name << " row max id vector:\n" << os.str();
      }
    }
  }

  virtual void updateSamplesNum(const std::vector<Argument>& arguments) {}

  virtual real evalImp(std::vector<Argument>& arguments) { return 0; }
};
REGISTER_EVALUATOR(max_id_printer, MaxIdPrinter);
/**
 * @brief print sequence max frames of each layer
 *
 * The config file api is maxframe_printer_evaluator.
 */
class MaxFramePrinter : public NotGetableEvaluator {
 private:
  IVectorPtr maxIds_;
  MatrixPtr maxValues_;
  MatrixPtr value_;

 public:
  MaxFramePrinter() {
    value_ =
        Matrix::create(nullptr, /* height= */ 1, 1, /* trans= */ false, false);
  }

  virtual void eval(const NeuralNetwork& nn) {
    for (const std::string& name : config_.input_layers()) {
      const Argument& argu = nn.getLayer(name)->getOutput();

      CHECK_EQ(argu.value->getWidth(), 1LU);
      size_t numSequences = argu.getNumSequences();
      const int* starts = argu.sequenceStartPositions->getData(false);

      std::ostringstream os;
      for (size_t i = 0; i < numSequences; ++i) {
        size_t offset = starts[i];
        size_t size = starts[i + 1] - starts[i];
        value_->setData(argu.value->getData() + offset, 1LU, size);

        size_t height = 1LU;
        size_t width = std::min((size_t)config_.num_results(), size);
        IVector::resizeOrCreate(maxIds_, height * width, false);
        Matrix::resizeOrCreate(maxValues_, height, width, false);

        value_->rowMax(*maxIds_, *maxValues_);

        int* ids = maxIds_->getData();
        real* values = maxValues_->getData();
        for (size_t j = 0; j < width; ++j) {
          os << ids[j] << " : " << values[j] << ", ";
        }
        os << "total " << size << " frames" << std::endl;
      }
      LOG(INFO) << "layer=" << name << " sequence max frames:\n" << os.str();
    }
  }

  virtual void updateSamplesNum(const std::vector<Argument>& arguments) {}

  virtual real evalImp(std::vector<Argument>& arguments) { return 0; }
};
REGISTER_EVALUATOR(max_frame_printer, MaxFramePrinter);

/**
 * @brief print text according to index matrix and a dictionary.
 *
 * There can be multiple input to this layer:
 * - If there is only one input, the input must be a matrix containing
 *      the sequence of indices;
 * - If there are more than one input, the first input should be ids,
 *      and are interpreted as sample ids.
 *
 * The output format will be:
 *
 * - sequence without sub-sequence, and there is probability.
 *
 *     @code
 *      id \t prob space_seperated_tokens_from_dictionary_according_to_seq
 *     @endcode
 *
 * - sequence without sub-sequence, and there is not probability.
 *
 *     @code
 *      id \t space_seperated_tokens_from_dictionary_according_to_seq
 *     @endcode
 *
 * - sequence with sub-sequence, and there is not probability.
 *
 *     @code
 *      id \t space_seperated_tokens_from_dictionary_according_to_sub_seq
 *      \t \t space_seperated_tokens_from_dictionary_according_to_sub_seq
 *      ...
 *     @endcode
 *
 * Typically SequenceTextPrinter layer takes output of maxid or RecurrentGroup
 * with maxid (when generating) as an input.
 *
 * The config file api is seqtext_printer_evaluator.
 *
 */
class SequenceTextPrinter : public NotGetableEvaluator {
 private:
  /// dict_file, which contains a list of tokens
  std::vector<std::string> dict_;
  /// result_file, which is the output file
  std::ofstream os_;
  /// True/False, to indicate whether to use space to separate output tokens.
  /// Default is True. No space is added if set to False.
  bool delimited_;
  /// store the cpu version of argument.ids
  std::vector<IVectorPtr> cpuIds_;
  /// store the probability associated with each sequence
  std::vector<MatrixPtr> cpuIn_;

 public:
  SequenceTextPrinter() {}

  virtual void init(const EvaluatorConfig& config) {
    Evaluator::init(config);
    if (!config.dict_file().empty()) {
      loadFileList(config.dict_file(), dict_);
    }

    os_.open(config.result_file(), std::ofstream::trunc);
    CHECK(os_.is_open()) << "Failed to open file " << config.result_file();
    delimited_ = config.delimited();
  }

  virtual void updateSamplesNum(const std::vector<Argument>& arguments) {}

  virtual real evalImp(std::vector<Argument>& arguments) {
    CHECK_GE(arguments.size(), 1LU);
    bool hasId = arguments.size() > 1;
    size_t numSequences = arguments[0].getNumSequences();
    if (hasId) {
      CHECK_EQ(arguments[0].ids->getSize(), numSequences)
          << "first input must be sample id.";
    }
    for (size_t i = hasId ? 1 : 0; i < arguments.size(); ++i) {
      CHECK_EQ((size_t)arguments[i].getNumSequences(), numSequences);
    }

    auto resizeVector = [](IVectorPtr& dest, const IVectorPtr& src) {
      if (src && src->useGpu()) {
        IVector::resizeOrCreate(dest, src->getSize(), false);
        dest->copyFrom(*src);
      } else {
        dest = src;
      }
    };

    auto resizeMatrix = [](MatrixPtr& dest, const MatrixPtr& src) {
      if (src && src->useGpu()) {
        Matrix::resizeOrCreate(
            dest, src->getHeight(), src->getWidth(), false, false);
        dest->copyFrom(*src);
      } else {
        dest = src;
      }
    };

    cpuIds_.resize(arguments.size());
    cpuIn_.resize(arguments.size());
    for (size_t i = 0; i < arguments.size(); ++i) {
      resizeVector(cpuIds_[i], arguments[i].ids);
      resizeMatrix(cpuIn_[i], arguments[i].in);
    }

    int* sampleIds = nullptr;
    if (hasId) {
      sampleIds = cpuIds_[0]->getData();
    }

    for (size_t i = 0; i < numSequences; ++i) {
      os_ << (hasId ? sampleIds[i] : i);
      for (size_t j = hasId ? 1 : 0; j < arguments.size(); ++j) {
        int* output = cpuIds_[j]->getData();
        const int* starts = arguments[j].sequenceStartPositions->getData(false);

        auto seqPrint = [&](int start, int end) {
          os_ << "\t";
          for (int k = start; k < end; k++) {
            int id = output[k];
            os_ << (delimited_ ? " " : "");
            if (!dict_.empty()) {
              CHECK_LT((size_t)id, dict_.size());
              os_ << dict_[id];
            } else {
              os_ << id;
            }
          }
        };

        if (arguments[j].hasSubseq()) {
          // print sequence with sub-sequence
          const int* subStarts =
              arguments[j].subSequenceStartPositions->getData(false);
          int subSeqId_start = 0;
          int subSeqId_end = 0;
          for (size_t k = 0; k < (size_t)arguments[j].getNumSubSequences() + 1;
               ++k) {
            if (starts[i] == subStarts[k]) subSeqId_start = k;
            if (starts[i + 1] == subStarts[k]) subSeqId_end = k;
          }
          for (int k = subSeqId_start; k < subSeqId_end; k++) {
            seqPrint(subStarts[k], subStarts[k + 1]);
            os_ << std::endl;
          }

        } else {
          // print sequence without sub-sequence
          if (arguments[j].in) {  // beam print
            real* probs = cpuIn_[j]->rowBuf(i);
            os_ << std::endl;
            int start = starts[i];
            int seqEnd = starts[i + 1];
            for (size_t k = 0; k < arguments[j].in->getWidth(); ++k) {
              if (start == seqEnd) {
                break;
              }
              int end = start + output[start] + 2;
              CHECK_LE(end, seqEnd);
              CHECK_EQ(output[end - 1], -1);
              os_ << k << "\t" << probs[k];
              seqPrint(start + 1, end - 1);
              os_ << std::endl;
              start = end;
            }
          } else {
            seqPrint(starts[i], starts[i + 1]);
          }
        }
      }
      os_ << std::endl;
    }
    return 0;
  }
};
REGISTER_EVALUATOR(seq_text_printer, SequenceTextPrinter);
/**
 * @brief print classification error.
 *
 * The config file api is classification_error_printer_evaluator.
 */
class ClassificationErrorPrinter : public ClassificationErrorEvaluator {
 public:
  virtual void updateSamplesNum(const std::vector<Argument>& arguments) {}

  virtual real evalImp(std::vector<Argument>& arguments) {
    MatrixPtr errorMat = calcError(arguments);

    std::ostringstream os;
    errorMat->print(os);
    LOG(INFO) << "Printer=" << config_.name() << " Classification Error:\n"
              << os.str();

    if (auto startPos = arguments[0].sequenceStartPositions) {
      std::ostringstream os;
      startPos->getVector(false)->print(os, startPos->getSize());
      LOG(INFO) << "Printer=" << config_.name() << " sequence pos vector:\n"
                << os.str();
    }
    return 0;
  }
};
REGISTER_EVALUATOR(classification_error_printer, ClassificationErrorPrinter);

std::string DummyEvaluator::getTypeImpl() const { return "dummy"; }

}  // namespace paddle
