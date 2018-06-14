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
#include "paddle/gserver/gradientmachines/NeuralNetwork.h"
#include "paddle/utils/StringUtil.h"

namespace paddle {

/**
 * calculate sequence-to-sequence edit distance
 */
class CTCErrorEvaluator : public Evaluator {
 private:
  MatrixPtr outActivations_;
  int numTimes_, numClasses_, numSequences_, blank_;
  real deletions_, insertions_, substitutions_;
  int seqClassficationError_;
  mutable std::unordered_map<std::string, real> evalResults_;

  std::vector<int> path2String(const std::vector<int>& path) {
    std::vector<int> str;
    str.clear();
    int prevLabel = -1;
    for (std::vector<int>::const_iterator label = path.begin();
         label != path.end();
         label++) {
      if (*label != blank_ &&
          (str.empty() || *label != str.back() || prevLabel == blank_)) {
        str.push_back(*label);
      }
      prevLabel = *label;
    }
    return str;
  }

  std::vector<int> bestLabelSeq() {
    std::vector<int> path;
    path.clear();
    real* acts = outActivations_->getData();
    for (int i = 0; i < numTimes_; ++i) {
      path.push_back(std::max_element(acts + i * numClasses_,
                                      acts + (i + 1) * numClasses_) -
                     (acts + i * numClasses_));
    }
    return path2String(path);
  }

  /* "sp, dp, ip" is the weighting parameter of "substitution, deletion,
   * insertion"
   * in edit-distance error */
  real stringAlignment(std::vector<int>& gtStr,
                       std::vector<int>& recogStr,
                       bool backtrace = true,
                       real sp = 1.0,
                       real dp = 1.0,
                       real ip = 1.0) {
    std::vector<std::vector<int>> matrix;
    int substitutions, deletions, insertions;
    real distance;
    int n = gtStr.size();
    int m = recogStr.size();

    if (n == 0) {
      substitutions = 0;
      deletions = 0;
      insertions = m;
      distance = m;
    } else if (m == 0) {
      substitutions = 0;
      deletions = n;
      insertions = 0;
      distance = n;
    } else {
      substitutions = 0;
      deletions = 0;
      insertions = 0;
      distance = 0;
      // initialize the matrix
      matrix.resize(n + 1);
      for (int i = 0; i < n + 1; ++i) {
        matrix[i].resize(m + 1);
        for (int j = 0; j < m + 1; ++j) {
          matrix[i][j] = 0;
        }
      }
      for (int i = 0; i < n + 1; ++i) {
        matrix[i][0] = i;
      }
      for (int j = 0; j < m + 1; ++j) {
        matrix[0][j] = j;
      }

      // calculate the insertions, substitutions and deletions
      for (int i = 1; i < n + 1; ++i) {
        int s_i = gtStr[i - 1];
        for (int j = 1; j < m + 1; ++j) {
          int t_j = recogStr[j - 1];
          int cost = (s_i == t_j) ? 0 : 1;
          const int above = matrix[i - 1][j];
          const int left = matrix[i][j - 1];
          const int diag = matrix[i - 1][j - 1];
          const int cell = std::min(above + 1, std::min(left + 1, diag + cost));
          matrix[i][j] = cell;
        }
      }

      if (backtrace) {
        size_t i = n;
        size_t j = m;
        substitutions = 0;
        deletions = 0;
        insertions = 0;

        while (i != 0 && j != 0) {
          if (matrix[i][j] == matrix[i - 1][j - 1]) {
            --i;
            --j;
          } else if (matrix[i][j] == matrix[i - 1][j - 1] + 1) {
            ++substitutions;
            --i;
            --j;
          } else if (matrix[i][j] == matrix[i - 1][j] + 1) {
            ++deletions;
            --i;
          } else {
            ++insertions;
            --j;
          }
        }
        while (i != 0) {
          ++deletions;
          --i;
        }
        while (j != 0) {
          ++insertions;
          --j;
        }
        int diff = substitutions + deletions + insertions;
        if (diff != matrix[n][m]) {
          LOG(ERROR) << "Found path with distance " << diff
                     << " but Levenshtein distance is " << matrix[n][m];
        }

        distance = (sp * substitutions) + (dp * deletions) + (ip * insertions);
      } else {
        distance = (real)matrix[n][m];
      }
    }
    real maxLen = std::max(m, n);
    deletions_ += deletions / maxLen;
    insertions_ += insertions / maxLen;
    substitutions_ += substitutions / maxLen;

    if (distance != 0) {
      seqClassficationError_ += 1;
    }

    return distance / maxLen;
  }

  real editDistance(
      real* output, int numTimes, int numClasses, int* labels, int labelsLen) {
    numTimes_ = numTimes;
    numClasses_ = numClasses;
    blank_ = numClasses_ - 1;
    outActivations_ = Matrix::create(output, numTimes, numClasses);
    std::vector<int> recogStr, gtStr;
    recogStr = bestLabelSeq();
    for (int i = 0; i < labelsLen; ++i) {
      gtStr.push_back(labels[i]);
    }

    return stringAlignment(gtStr, recogStr);
  }

  void storeLocalValues() const {
    evalResults_["error"] = numSequences_ ? totalScore_ / numSequences_ : 0;
    evalResults_["deletion_error"] =
        numSequences_ ? deletions_ / numSequences_ : 0;
    evalResults_["insertion_error"] =
        numSequences_ ? insertions_ / numSequences_ : 0;
    evalResults_["substitution_error"] =
        numSequences_ ? substitutions_ / numSequences_ : 0;
    evalResults_["sequence_error"] =
        (real)seqClassficationError_ / numSequences_;
  }

 public:
  CTCErrorEvaluator()
      : numTimes_(0),
        numClasses_(0),
        numSequences_(0),
        blank_(0),
        deletions_(0),
        insertions_(0),
        substitutions_(0),
        seqClassficationError_(0) {}

  virtual real evalImp(std::vector<Argument>& arguments) {
    CHECK_EQ(arguments.size(), (size_t)2);
    Argument output, label;
    output.resizeAndCopyFrom(arguments[0], false, HPPL_STREAM_DEFAULT);
    label.resizeAndCopyFrom(arguments[1], false, HPPL_STREAM_DEFAULT);
    hl_stream_synchronize(HPPL_STREAM_DEFAULT);
    CHECK(label.sequenceStartPositions);
    CHECK(label.ids);
    size_t numSequences = label.sequenceStartPositions->getSize() - 1;
    const int* labelStarts = label.sequenceStartPositions->getData(false);
    const int* outputStarts = output.sequenceStartPositions->getData(false);
    real totalErr = 0;
    for (size_t i = 0; i < numSequences; ++i) {
      real err = 0;
      err = editDistance(
          output.value->getData() + output.value->getWidth() * outputStarts[i],
          outputStarts[i + 1] - outputStarts[i],
          output.value->getWidth(),
          label.ids->getData() + labelStarts[i],
          labelStarts[i + 1] - labelStarts[i]);

      totalErr += err;
    }

    return totalErr;
  }

  virtual void eval(const NeuralNetwork& nn) {
    Evaluator::eval(nn);
    std::vector<Argument> arguments;
    arguments.reserve(config_.input_layers_size());
    for (const std::string& name : config_.input_layers()) {
      arguments.push_back(nn.getLayer(name)->getOutput());
    }
  }

  virtual void updateSamplesNum(const std::vector<Argument>& arguments) {
    numSequences_ += arguments[1].getNumSequences();
  }

  virtual void start() {
    Evaluator::start();
    numSequences_ = 0;
    blank_ = 0;
    deletions_ = 0;
    insertions_ = 0;
    substitutions_ = 0;
    seqClassficationError_ = 0;
  }

  virtual void printStats(std::ostream& os) const {
    storeLocalValues();
    os << config_.name() << " error = " << evalResults_["error"];
    os << " deletions error = " << evalResults_["deletion_error"];
    os << " insertions error = " << evalResults_["insertion_error"];
    os << " substitution error = " << evalResults_["substitution_error"];
    os << " sequence error = " << evalResults_["sequence_error"];
  }

  virtual void distributeEval(ParameterClient2* client) {
    double buf[6] = {totalScore_,
                     (double)deletions_,
                     (double)insertions_,
                     (double)substitutions_,
                     (double)seqClassficationError_,
                     (double)numSequences_};
    client->reduce(buf, buf, 6, FLAGS_trainer_id, 0);
    totalScore_ = buf[0];
    deletions_ = (real)buf[1];
    insertions_ = (real)buf[2];
    substitutions_ = (real)buf[3];
    seqClassficationError_ = (int)buf[4];
    numSequences_ = (int)buf[5];
  }

  void getNames(std::vector<std::string>* names) {
    storeLocalValues();
    names->reserve(names->size() + evalResults_.size());
    for (auto it = evalResults_.begin(); it != evalResults_.end(); ++it) {
      names->push_back(config_.name() + "." + it->first);
    }
  }

  real getValue(const std::string& name, Error* err) const {
    storeLocalValues();

    std::vector<std::string> buffers;
    paddle::str::split(name, '.', &buffers);
    auto it = evalResults_.find(buffers[buffers.size() - 1]);

    if (it == evalResults_.end()) {
      *err = Error("Evaluator does not have the key %s", name.c_str());
      return 0.0f;
    }

    return it->second;
  }

  std::string getType(const std::string& name, Error* err) const {
    this->getValue(name, err);
    if (!err->isOK()) {
      return "";
    }
    return "ctc_edit_distance";
  }
};

REGISTER_EVALUATOR(ctc_edit_distance, CTCErrorEvaluator);

}  // namespace paddle
