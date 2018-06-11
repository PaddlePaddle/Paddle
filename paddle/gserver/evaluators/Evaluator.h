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

#include <fstream>
#include "ModelConfig.pb.h"
#include "paddle/parameter/Argument.h"
#include "paddle/pserver/ParameterClient2.h"
#include "paddle/utils/ClassRegistrar.h"
#include "paddle/utils/Error.h"

namespace paddle {

class NeuralNetwork;
/**
 * @def REGISTER_EVALUATOR
 * @brief Macro for registering evaluator class
 */

#define REGISTER_EVALUATOR(__type_name, __class_name)                \
  static InitFunction __reg_type_##__type_name([]() {                \
    Evaluator::registrar_.registerClass<__class_name>(#__type_name); \
  })
/**
 * @brief Base class for Evaluator
 * Evaluating the performance of a model is very important.
 * It indicates how successful the scores(predictions) of a datasets
 * has been by a trained model.
 */
class Evaluator {
 public:
  static Evaluator* create(const EvaluatorConfig& config);

  Evaluator() : numSamples_(0), totalScore_(0) {}

  virtual ~Evaluator() {}

  virtual void init(const EvaluatorConfig& config) { config_ = config; }

  /**
   * @brief start to evaluate some data
   */
  virtual void start() {
    numSamples_ = 0;
    totalScore_ = 0;
  }

  /**
   * @brief Process a batch of data.
   */
  virtual void eval(const NeuralNetwork& nn);

  /**
   * @brief Process a batch of data.
   * @return the score for the batch if it make sense to sum the score across
   * batches.
   * @note Otherwise evaluator should return 0 and override finish() and
   * printStats() to do the right calculation.
   */
  virtual real evalImp(std::vector<Argument>& arguments) = 0;

  /**
   * @brief Update the number of processed samples
   */
  virtual void updateSamplesNum(const std::vector<Argument>& arguments) {
    numSamples_ += arguments[0].getBatchSize();
  }

  /// finish() should be called before distributeEval
  virtual void distributeEval(ParameterClient2* client) {
    LOG(FATAL) << "Not implemeted";
  }

  void mergeResultsOfAllClients(ParameterClient2* client) {
    double data[2] = {totalScore_, numSamples_};
    client->reduce(data, data, 2, FLAGS_trainer_id, 0);
    totalScore_ = data[0];
    numSamples_ = data[1];
  }

  /**
   * @brief finish the evaluation.
   */
  virtual void finish() {}

  /**
   * @brief print the statistics of evaluate result
   * @note finish() should be called before printStats
   */
  virtual void printStats(std::ostream& os) const {
    os << config_.name() << "="
       << (numSamples_ ? totalScore_ / numSamples_ : 0);
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const Evaluator& evaluator) {
    evaluator.printStats(os);
    return os;
  }

  friend std::ostream&& operator<<(std::ostream&& os,  // NOLINT
                                   const Evaluator& evaluator) {
    evaluator.printStats(os);
    return std::move(os);
  }

  static ClassRegistrar<Evaluator> registrar_;

  /**
   * @brief getNames will return all field names of current evaluator.
   *
   * The format of name is `evaluator_name.evaluator_fields`. If the evaluator
   * has multiple field, the name could be `evaluator_name.field1`. For example
   * the PrecisionRecallEvaluator contains `precision`, `recall` fields. The get
   * names will return `precision_recall_evaluator.precision`,
   * `precision_recall_evaluator.recal`, etc.
   *
   * Also, if current Evaluator is a combined evaluator. getNames will return
   * all names of all evaluators inside the combined evaluator.
   *
   * @param names [out]: the field names of current evaluator.
   * @note Never clear the names parameter inside getNames.
   */
  virtual void getNames(std::vector<std::string>* names) {
    names->push_back(config_.name());
  }

  /**
   * @brief getValue will return the current evaluate value of one field.
   *
   * @param name: The field name of current evaluator.
   * @param err [out]: The error state.
   *
   * @return The evaluate value(metric).
   */
  virtual real getValue(const std::string& name, Error* err) const {
    if (name != config_.name()) {
      *err = Error("no such name of evaluator %s", name.c_str());
      return .0f;
    }
    return this->getValueImpl();
  }

  /**
   * @brief getType will return the evaluator type by field name.
   *
   * Evaluate Type is the current type of evaluator in string. Such as 'auc',
   * 'precision_recall'. In combined evaluator, different name may get different
   * evaluate type because it could be evaluated by different evaluator inside.
   *
   * @param name: The field name of current Evaluator.
   * @param err: The error state. nullptr means don't care.
   * @return the evaluator type string.
   */
  virtual std::string getType(const std::string& name, Error* err) const {
    if (name != config_.name()) {
      *err = Error("no such name of evaluator %s", name.c_str());
      return std::string();
    }
    return this->getTypeImpl();
  }

 protected:
  /**
   * @brief getValueImpl The simplest way to define getValue result. If this
   * evaluator doesn't contain multiple fields, and do not throw any error, just
   * implemented this method to get the evaluate result(metric).
   * @return Evaluate result(metric).
   */
  virtual real getValueImpl() const {
    return numSamples_ != .0 ? totalScore_ / numSamples_ : .0;
  }

  /**
   * @brief getTypeImpl The simplest way to define getType result. If this
   * evaluator doesn't combine many evaluators, the get type should only return
   * itself type.
   * @return Evaluator type.
   */
  virtual std::string getTypeImpl() const { return "base"; }

 protected:
  EvaluatorConfig config_;
  double numSamples_;
  double totalScore_;
};

/**
 * @brief The NotGetableEvaluator class is the base class of evaluator that
 * cannot get value in runtime. The most NotGetableEvaluator is Printer
 * Evaluator, which is only used to debug network configuration.
 */
class NotGetableEvaluator : public Evaluator {
  // Evaluator interface
 public:
  void getNames(std::vector<std::string>* names) {}

  real getValue(const std::string& name, Error* err) const {
    *err = Error("Not implemented");
    return .0f;
  }

  std::string getType(const std::string& name, Error* err) const {
    *err = Error("Not implemented");
    return "";
  }
};

class DummyEvaluator : public Evaluator {
 public:
  DummyEvaluator() {}
  virtual void init(const EvaluatorConfig&) {}
  virtual void start() {}
  virtual void eval(const NeuralNetwork&) {}
  virtual real evalImp(std::vector<Argument>& arguments) {
    (void)arguments;
    return -1;
  }
  virtual void finish() {}
  virtual void printStats(std::ostream&) const {}

  // Evaluator interface
 protected:
  std::string getTypeImpl() const;
};
/**
 * @brief evaluate AUC using colIdx-th column as prediction.
 * The AUC(Area Under the Curve) is a common evaluation metric
 * for binary classification problems. It computes the area under
 * the receiver operating characteristic(ROC) curve.
 *
 * @note colIdx-th column
 *
 * - colIdx = 0: the 0-th column.
 * - colIdx > 0: the colIdx-th column.
 * - colIdx < 0: the last colIdx-th column.
 *
 * The config file api is auc_evaluator.
 *
 */
class AucEvaluator : public Evaluator {
 public:
  AucEvaluator(int32_t colIdx)
      : colIdx_(colIdx),
        realColumnIdx_(0),
        cpuOutput_(nullptr),
        cpuLabel_(nullptr),
        cpuWeight_(nullptr) {}

  virtual void start();

  virtual real evalImp(std::vector<Argument>& arguments);

  virtual void printStats(std::ostream& os) const {
    os << config_.name() << "=" << calcAuc();
  }

  virtual void distributeEval(ParameterClient2* client);

 private:
  static const uint32_t kBinNum_ = (1 << 24) - 1;
  static const int kNegativeLabel_ = 0;
  double statPos_[kBinNum_ + 1];
  double statNeg_[kBinNum_ + 1];
  int32_t colIdx_;
  uint32_t realColumnIdx_;
  MatrixPtr cpuOutput_;
  IVectorPtr cpuLabel_;
  MatrixPtr cpuWeight_;

  AucEvaluator() {}

  inline static double trapezoidArea(double X1,
                                     double X2,
                                     double Y1,
                                     double Y2) {
    return (X1 > X2 ? (X1 - X2) : (X2 - X1)) * (Y1 + Y2) / 2.0;
  }

  double calcAuc() const;

  // Evaluator interface
 protected:
  real getValueImpl() const;
  std::string getTypeImpl() const;
};

/**
 * @brief RankAucEvaluator calculates the AUC of each list (i.e., titles
 * under the same query), and averages them. Each list should be organized
 * as a sequence. The inputs of this evaluator is [output, click, pv]. If pv
 * is not provided, it will be set to 1. The types of click and pv are
 * dense value.
 */
class RankAucEvaluator : public Evaluator {
 public:
  // evaluate ranking AUC
  virtual void start();

  virtual void updateSamplesNum(const std::vector<Argument>& arguments);

  virtual real evalImp(std::vector<Argument>& arguments);

  virtual void distributeEval(ParameterClient2* client) {
    mergeResultsOfAllClients(client);
  }

 private:
  MatrixPtr output_;
  MatrixPtr click_;
  MatrixPtr pv_;
  std::vector<std::pair<real, int>> outputPair_;

  double calcRankAuc(real* outputData,
                     real* clickData,
                     real* pvData,
                     size_t size);

  // Evaluator interface
 protected:
  std::string getTypeImpl() const;
};

/**
 * @brief precision, recall and f1 score Evaluator
 * \f[
 * precision = \frac{tp}{tp+tn} \\
 * recall=\frac{tp}{tp+fn} \\
 * f1=2*\frac{precsion*recall}{precision+recall}
 * \f]
 *
 * The config file api is precision_recall_evaluator.
 */
class PrecisionRecallEvaluator : public Evaluator {
 public:
  // Evaluate precision, recall and F1 score
  PrecisionRecallEvaluator()
      : isMultiBinaryLabel_(false),
        cpuOutput_(nullptr),
        cpuLabel_(nullptr),
        cpuWeight_(nullptr) {}

  virtual void start();

  virtual real evalImp(std::vector<Argument>& arguments);

  virtual void printStats(std::ostream& os) const;

  virtual void distributeEval(ParameterClient2* client);

  void getNames(std::vector<std::string>* names);

  real getValue(const std::string& name, Error* err) const;

  std::string getType(const std::string& name, Error* err) const;

  struct StatsInfo {
    /// numbers of true positives
    double TP;
    /// numbers of true negatives
    double TN;
    /// numbers of false positives
    double FP;
    /// numbers of false negatives
    double FN;

    StatsInfo() : TP(0.0), TN(0.0), FP(0.0), FN(0.0) {}
  };

 private:
  bool isMultiBinaryLabel_;
  std::vector<StatsInfo> statsInfo_;

  MatrixPtr cpuOutput_;
  IVectorPtr cpuLabel_;
  MatrixPtr cpuWeight_;

  struct PrintStatsInfo {
    double precision;
    double recall;
    double f1;
    double macroAvgPrecision;
    double macroAvgRecall;
    double macroAvgF1Score;
    double microAvgPrecision;
    double microAvgRecall;
    double microAvgF1Score;
  };

  bool getStatsInfo(PrintStatsInfo* info) const;

  void calcStatsInfo(const MatrixPtr& output,
                     const IVectorPtr& label,
                     const MatrixPtr& weight);

  void calcStatsInfoMulti(const MatrixPtr& output,
                          const MatrixPtr& label,
                          const MatrixPtr& weight);

  inline static double calcPrecision(double TP, double FP) {
    if (TP > 0.0 || FP > 0.0) {
      return TP / (TP + FP);
    } else {
      return 1.0;
    }
  }

  inline static double calcRecall(double TP, double FN) {
    if (TP > 0.0 || FN > 0.0) {
      return TP / (TP + FN);
    } else {
      return 1.0;
    }
  }

  inline static double calcF1Score(double precision, double recall) {
    if (precision > 0.0 || recall > 0.0) {
      return 2 * precision * recall / (precision + recall);
    } else {
      return 0;
    }
  }

  mutable std::unordered_map<std::string, real> values_;

  void storeLocalValues() const;
};

/*
 * @brief positive-negative pair rate Evaluator
 *
 * The config file api is pnpair_evaluator.
 */
class PnpairEvaluator : public Evaluator {
 public:
  PnpairEvaluator()
      : cpuOutput_(nullptr),
        cpuLabel_(nullptr),
        cpuInfo_(nullptr),
        cpuWeight_(nullptr) {}

  virtual void start();
  virtual real evalImp(std::vector<Argument>& arguments);

  struct PredictionResult {
    PredictionResult(real __out, int __label, int __queryid, real __weight)
        : out(__out), label(__label), queryid(__queryid), weight(__weight) {}
    real out;
    int label;
    int queryid;
    real weight;
  };
  std::vector<PredictionResult> predictArray_;
  void printPredictResults() {
    std::ofstream fs(FLAGS_predict_file);
    CHECK(fs) << "Fail to open " << FLAGS_predict_file;
    for (auto& res : predictArray_) {
      fs << res.out << " " << res.label << " " << res.queryid << std::endl;
    }
  }

  void stat(size_t start,
            size_t end,
            PredictionResult* answers,
            double& pos,
            double& neg,
            double& spe);
  void calc(std::vector<PredictionResult>& predictArray);

  virtual void finish() { calc(predictArray_); }

  virtual void printStats(std::ostream& os) const {
    os << " pos/neg=" << this->getValueImpl();
  }

  virtual void distributeEval(ParameterClient2* client) {
    client->reduce(pairArray_, pairArray_, kPairArrayNum_, FLAGS_trainer_id, 0);
    LOG(INFO) << " distribute eval calc total pos pair: " << pairArray_[0]
              << " calc total neg pair: " << pairArray_[1];
  }

 private:
  static const uint32_t kPairArrayNum_ = 2;
  double pairArray_[kPairArrayNum_];
  MatrixPtr cpuOutput_;
  IVectorPtr cpuLabel_;
  IVectorPtr cpuInfo_;
  MatrixPtr cpuWeight_;

  // Evaluator interface
 protected:
  real getValueImpl() const {
    return pairArray_[0] / ((pairArray_[1] <= 0) ? 1.0 : pairArray_[1]);
  }
  std::string getTypeImpl() const;
};

}  // namespace paddle
