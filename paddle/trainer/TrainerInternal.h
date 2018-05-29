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
#include <stdlib.h>
#include <fstream>

#include "ParameterUpdater.h"
#include "TrainerConfig.pb.h"
#include "TrainerConfigHelper.h"
#include "TrainerInternalConfig.h"
#include "hl_gpu.h"
#include "paddle/gserver/gradientmachines/GradientMachine.h"

namespace paddle {

/**
 * TrainerInteral
 * the core training class for driving training logic
 */
class TrainerInternal {
 public:
  struct ParaStat {
    real maxAbsGrad;
    real avgAbsGrad;
    ParaStat() : maxAbsGrad(.0), avgAbsGrad(.0) {}
  };

  TrainerInternal() {}

  /**
   * Intializes trainer internal class
   * @param config network config
   * @param machine gradient machine
   * @param intconfig training config
   * @param stats training stats
   * @param testing if it is in testing phase
   */
  void init(const std::shared_ptr<TrainerConfigHelper>& config,
            const GradientMachinePtr& machine,
            std::unique_ptr<TrainerInternalConfig>&& intconfig,
            const std::shared_ptr<TrainerStats>& stats,
            bool testing);

  virtual ~TrainerInternal() {}

  /**
   * CreateParameterUpdater
   * @param testing if it is in testing phase
   */
  void createParameterUpdater(bool testing);

  /**
   * FinishTrainPass
   * @param passId current pass id
   * @param batchId current batch id, starts from 0
   */
  void finishTrainPass(int passId, int batchId);

  /**
   * trainOneBatch
   * @param batchId current batch id
   * @param dataBatch data for the batch
   */
  void trainOneBatch(int64_t batchId,
                     const DataBatch& dataBatch,
                     std::vector<Argument>* outArgs);

  /**
   * showParameterStats
   * @param paraStats training stats
   */
  void showParameterStats(const std::vector<ParaStat>& paraStats);

  /**
   * getGradientMachine
   */
  inline const GradientMachinePtr& getGradientMachine() const {
    return gradientMachine_;
  }

  /**
   * getParameterUpdater
   */
  inline const std::shared_ptr<ParameterUpdater>& getParameterUpdater() {
    return parameterUpdater_;
  }

  /**
   * setCurrentEvaluator
   * @param eval evaluator to set
   */
  inline void setCurrentEvaluator(Evaluator* eval) { currentEvaluator_ = eval; }

  /**
   * setEvaluator
   * @param eval evaluator to set
   */
  inline void setEvaluator(Evaluator* eval) { evaluator_ = eval; }

  /**
   * forwardBackwardBatch
   * @param inArgs input argument for data batch
   * @param outArgs output argument from neural network
   * @param updateCallback layerwise parameter gradient statistics
   * @param doPipelineUpdate whether to do pipeline update
   */
  virtual void forwardBackwardBatch(const std::vector<Argument>& inArgs,
                                    std::vector<Argument>& outArgs,
                                    PassType& passType,
                                    UpdateCallback updateCallback,
                                    bool doPipelineUpdate);

 protected:
  std::shared_ptr<ParameterUpdater> parameterUpdater_;
  GradientMachinePtr gradientMachine_;
  std::shared_ptr<TrainerConfigHelper> config_;
  std::unique_ptr<TrainerInternalConfig> intconfig_;
  std::shared_ptr<TrainerStats> stats_;
  Evaluator* currentEvaluator_;
  Evaluator* evaluator_;
};

}  // namespace paddle
