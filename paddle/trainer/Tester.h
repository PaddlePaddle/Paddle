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

#include "TrainerConfig.pb.h"

#include <stdlib.h>
#include <fstream>
#include "ParamUtil.h"
#include "ParameterUpdater.h"
#include "TesterConfig.h"
#include "TrainerInternalConfig.h"

namespace paddle {

/**
 * Neural Network test logics code.
 * It is a private class for Trainer.
 */
class Tester {
 public:
  /**
   * Ctor
   * @param config Trainer Config.
   * @param intconfig Tester Config.
   * @param gradientMachine Gradient machine(neuralnetwork) that will be tested.
   * @param parameterUpdater Parameter Updater. Not for updating parameter, just
   *                         for getting parameter from parameter-server.
   * @param testDataProvider Test data provider.
   */
  Tester(const std::shared_ptr<TrainerConfigHelper>& config,
         std::unique_ptr<TesterConfig>&& intconfig,
         const GradientMachinePtr& gradientMachine,
         const std::shared_ptr<ParameterUpdater>& parameterUpdater,
         std::shared_ptr<DataProvider> testDataProvider);

  /**
   * test one period.
   *
   * One period means 2 things.
   *   if test_period !=0 and not test_all_data_in_one_period, then
   *      will test test_period * batch_size data.
   *   else
   *      will test whole test data.
   *
   * It is convenience to test small set of data when test data set is large and
   * is training at same time.
   */
  void testOnePeriod();
  void startTestPeriod();
  void finishTestPeriod();
  void testOneDataBatch(const DataBatch& dataBatch,
                        std::vector<Argument>* outArgs);

  /**
   * Test for given data batch.
   * @param dataBatch Data batch.
   * @param evaluator Evaluator
   * @return cost
   */
  real forwardOneBatch(const DataBatch& dataBatch,
                       Evaluator* evaluator,
                       std::vector<Argument>* outArgs);

  /**
   * performance the full pass of test given test data provider
   */
  void test();

 protected:
  std::shared_ptr<ParameterClient2> testParameterClient_;
  std::shared_ptr<TrainerConfigHelper> config_;
  std::unique_ptr<TesterConfig> intconfig_;
  GradientMachinePtr gradientMachine_;
  std::shared_ptr<ParameterUpdater> parameterUpdater_;
  std::unique_ptr<Evaluator> testEvaluator_;
  std::unique_ptr<ParameterUtil> paramUtil_;
  DataProviderPtr testDataProvider_;
  TrainerStats stats_;

  // Used for saving the values of output layers
  std::ofstream os_;
  std::vector<MatrixPtr> cpuMat_;
  std::vector<IVectorPtr> cpuVec_;
  struct {
    int64_t numSamples;
    real cost;
  } testContext_;

 private:
  /**
   * Test one batch by batchId. It is only used for testOnePass.
   *
   * Durning testOnePass, each log_period will print cost statistics.
   *
   * @param batchId current batch id (from 0)
   * @return num of tested samples. Zero if end of pass.
   */
  int64_t testOneBatchById(int64_t batchId);

  /**
   * Test whole pass in one batch.
   *
   *
   * @param passId current pass id (from 0)
   */
  void testOnePassBatch(int passId);

  /**
   * test for one pass in several mini-batches.
   *
   * Used for sgd method.
   *
   * @param passId current pass id (from 0)
   */
  void testOnePass(int passId);

  /**
   * print the outArgs to a stream
   *
   * used for save feature file
   *
   * @param [in] outArgs output arguments for network.
   * @param [in,out] os output stream.
   */
  void printOutput(const std::vector<Argument>& outArgs, std::ostream& os);
};

}  //  namespace paddle
