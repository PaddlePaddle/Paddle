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

#pragma once

#include "paddle/utils/Util.h"

#include <stdio.h>

#include "hl_gpu.h"
#include "paddle/gserver/gradientmachines/GradientMachine.h"

#include "TrainerConfig.pb.h"

#include <stdlib.h>
#include <fstream>
#include <sstream>
#include "ParameterUpdater.h"

namespace paddle {
/**
 * @brief TrainerStats object will statistics sample processed and total cost.
 *
 * There are two stats in it, the 'AvgCost' and 'CurrentAvgCost'. 'AvgCost'
 * means cost through one pass(all mini-batches). 'CurrentAvgCost' means cost
 * through one mini-batch.
 */
class TrainerStats {
 public:
  /**
   * @brief reset all stats.
   *
   * often used before pass start.
   */
  inline void reset() {
    numProcessed_ = 0;
    totalCost_ = .0;
    this->resetCurrentStat();
  }

  /**
   * @brief reset current stat.
   *
   * 'current' means the most recent --log_period mini-batches
   */
  inline void resetCurrentStat() {
    currentCost_ = .0;
    currentSamples_ = 0;
  }

  /**
   * @brief add cost to stat.
   * @param numProcessed current mini-batch size
   * @param cost current mini-batch cost
   */
  inline void addCost(int64_t numProcessed, real cost) {
    this->numProcessed_ += numProcessed;
    this->totalCost_ += cost;
    this->currentSamples_ += numProcessed;
    this->currentCost_ += cost;
  }

  /**
   * @brief get average cost through on pass(all processed mini-batches)
   * @return pass average cost
   */
  inline real getAvgCost() const {
    CHECK_NE(this->numProcessed_, 0);
    return this->totalCost_ / this->numProcessed_;
  }

  /**
   * @brief get current mini-batch's average cost.
   * @return mini-batch average cost
   */
  inline real getCurrentAvgCost() const {
    CHECK_NE(this->currentSamples_, 0);
    return this->currentCost_ / this->currentSamples_;
  }

  /**
   * @brief get all processed samples' number
   * @return all processed samples' number
   */
  inline int64_t getNumProcessed() const { return this->numProcessed_; }

  /**
   * @brief same function as addCost. But it is simple to invoke.
   * For example:
   *
   * @code{.cpp}
   * TrainerStats stat;
   * cost = neuralNetwork.forward(batchSize);
   * stat += {batchSize, cost};
   * @endcode
   *
   * @param p a pair of parameter, first is numProcessed, second is cost.
   * @return *this
   */
  inline TrainerStats& operator+=(const std::pair<int64_t, real>& p) {
    this->addCost(p.first, p.second);
    return *this;
  }

  /**
   * @brief TrainerStats Constructor.
   *
   * reset stat when constructed.
   */
  inline TrainerStats() { this->reset(); }

  /**
   * @brief show stats to ostream.
   *
   * If there is no need to print current cost, set withCurrentCost to False.
   *
   * @param os output stream.
   * @param withCurrentCost print current cost or not.
   */
  void showStats(std::ostream& os, bool withCurrentCost = true) const {
    os << "samples=" << this->getNumProcessed()
       << " AvgCost=" << this->getAvgCost();
    if (withCurrentCost) {
      os << " CurrentCost=" << this->getCurrentAvgCost();
    }
  }

  /**
   * @brief get stats to std::string
   * @param withCurrentCost return current cost or not
   * @return stats string
   */
  std::string getStats(bool withCurrentCost = true) const {
    std::ostringstream os;
    this->showStats(os, withCurrentCost);
    return os.str();
  }

 private:
  int64_t numProcessed_;
  real totalCost_;
  real currentCost_;
  int64_t currentSamples_;
};

inline std::ostream& operator<<(std::ostream& os, const TrainerStats& stats) {
  stats.showStats(os);
  return os;
}

/**
 * TrainerInternalConfig
 * general configs for training
 */
struct TrainerInternalConfig {
  /**
   * @brief Create TrainerInternalConfig from GradientMachine::CreateMode and
   * command line arguments.
   * @param mode
   * @return
   */
  static std::unique_ptr<TrainerInternalConfig> createFromMode(
      GradientMachine::CreateMode mode);

  /**
   * indicate whether the training is local
   * if local, no parameter server is used
   */
  bool local;

  /**
   * indicate whether training uses GPU
   */
  bool use_gpu;

  /**
   * indicate number of trainer
   */
  int trainer_count;

  /**
   * how frequently to show param stats
   */
  int show_param_stats_period;

  /**
   * current trainer id
   */
  int trainer_id;

  /**
   * frequency to dump log
   */
  int log_period;

  /**
   * dot period
   */
  int dot_period;

  /**
   * num passes for training
   */
  int num_passes;

  /**
   * use old updater
   */
  bool use_old_updater;

  /**
   * whether to load and save parameter in pserver
   */
  bool loadsave_parameters_in_pserver;

  /**
   * training mode
   */
  GradientMachine::CreateMode mode;
};

}  //  namespace paddle
