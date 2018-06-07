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
#include "paddle/gserver/dataproviders/DataProvider.h"
#include "paddle/gserver/gradientmachines/GradientMachine.h"

#include <stdlib.h>
#include <fstream>
#include "ParamUtil.h"
#include "ParameterUpdater.h"
#include "Tester.h"
#include "TrainerConfigHelper.h"
#include "TrainerInternal.h"

DECLARE_int32(num_passes);

namespace paddle {

/**
 * Trainer Class
 *
 * Trainer combines GradientMachine, ParameterUpdater, DataProvider together to
 * train/test a NeuralNetwork.
 */
class Trainer {
 public:
  /**
   * Ctor.
   * @return
   */
  Trainer() : acceptedPassId_(0) {}

  virtual ~Trainer() {}

  /**
   * initialize a new trainer using config
   *
   * @param config TrainerConfig.
   * @param testing true if only for testing
   * @param gradientMachine GradientMachine that will be trained.
   *                        nullptr if create from config.
   * @param dataProvider Train Data Provider. null if create from config.
   * @param testDataProvider Test Data Provider. null if create from config.
   */
  virtual void init(
      const std::shared_ptr<TrainerConfigHelper>& config,
      bool testing = false,
      const std::shared_ptr<GradientMachine>& gradientMachine = nullptr,
      const std::shared_ptr<DataProvider>& dataProvider = nullptr,
      const std::shared_ptr<DataProvider>& testDataProvider = nullptr);

  /**
   * Train until num_passes reached.
   * One pass means neural network train through all training data.
   *
   * @param numPasses the number of traning pass.
   * @note Durning neural network training, the num passes may set a very large
   * value, and kill training process when result is good enough.
   */
  void train(size_t numPasses = (size_t)FLAGS_num_passes);

  /**
   * compare the gradient from bp with finite difference
   * @return  the maximal difference
   */
  real checkGradient();

  void startTrain();
  void finishTrain();
  void startTrainPass();
  void finishTrainPass();
  void trainOneDataBatch(DataBatch& dataBatch);
  void time();

  /**
   * given a dataBatch and the current parameter value
   * calculate its gradient and return the cost.
   *
   * TODO(yuyang18): I think this method is deprecated and buggy. Should it be
   * removed?
   */
  real calcGradient(const DataBatch& dataBatch,
                    const Vector& value,
                    Vector& gradient);

  /**
   * Get Trainer Config.
   */
  const TrainerConfig& getConfig() const { return config_->getConfig(); }

  /**
   * Get Train Data Provider
   */
  const DataProviderPtr& getDataProvider() { return dataProvider_; }

  /**
   * Get Gradient Machine.
   */
  const GradientMachinePtr& getGradientMachine() {
    return trainerInternal_.getGradientMachine();
  }

  /**
   * Get batch size in optimization config.
   * @note This method didn't return the actual batch size. Just batch size
   * set in the optimization config. The actual batch size in one trainer may
   * less than batch size in config due to there are not enough data.
   */
  int getBatchSize();

  /**
   * Do test job
   */
  void test();

  /**
   * Get parameter util ptr
   *
   * TODO(yuyang18): Make it return a smart pointer.
   */
  ParameterUtil* getParameterUtilPtr();

 protected:
  /**
   * Train one pass of data.
   *
   * SGD Method.
   */
  void trainOnePass();

  /**
   * Train one pass in one batch.
   *
   */
  void trainOnePassBatch(int passId);

  /**
   * set parameter gradient to zero
   */
  void clearGradient();

  void createTester();

 private:
  std::unique_ptr<TesterConfig> createTesterConfig();

 protected:
  std::shared_ptr<TrainerConfigHelper> config_;
  std::shared_ptr<TrainerStats> stats_;

  DataProviderPtr dataProvider_;
  DataProviderPtr testDataProvider_;
  MachineState trainState_;
  MachineState testState_;

  struct TrainPassContext {
    int64_t batchId;
    real avgTestCost;
    int64_t numAvgTests;
    int passId;
    int passInnerId;
  };
  std::vector<paddle::Argument> forwardOutput_;

  TrainPassContext trainPassContext_;

  std::unique_ptr<Evaluator> evaluator_;
  std::unique_ptr<Evaluator> currentEvaluator_;
  std::unique_ptr<Evaluator> averageEvaluator_;
  // training mode
  // used to decide which GradientMachine and ParameterUpdater to create
  GradientMachine::CreateMode mode_;
  int testing_;
  int acceptedPassId_;

  // trainer tester
  std::unique_ptr<Tester> tester_;

  // parameter util
  std::unique_ptr<ParameterUtil> paramUtil_;

  // trainer Internal
  TrainerInternal trainerInternal_;
};

}  // namespace paddle
