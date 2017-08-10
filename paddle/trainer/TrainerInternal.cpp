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

#include "TrainerInternal.h"

#include <fenv.h>
#include <stdio.h>

#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

#include <google/protobuf/text_format.h>

#include "paddle/gserver/gradientmachines/NeuralNetwork.h"
#include "paddle/gserver/layers/ValidationLayer.h"
#include "paddle/utils/GlobalConstants.h"
#include "paddle/utils/PythonUtil.h"
#include "paddle/utils/Stat.h"
#include "paddle/utils/Util.h"

#include "RemoteParameterUpdater.h"
#include "ThreadParameterUpdater.h"

namespace paddle {

void TrainerInternal::init(const std::shared_ptr<TrainerConfigHelper>& config,
                           const GradientMachinePtr& gradientMachine,
                           std::unique_ptr<TrainerInternalConfig>&& intconfig,
                           const std::shared_ptr<TrainerStats>& stats,
                           bool testing) {
  config_ = config;
  intconfig_ = std::move(intconfig);
  stats_ = stats;

  //! in training will use parameter updater definitly.
  //! But only use parameter in testing mode when some parameter in pserver.
  if (!testing || (config_->getOptConfig().use_sparse_remote_updater() &&
                   intconfig_->loadsave_parameters_in_pserver)) {
    createParameterUpdater(testing);
  }

  gradientMachine_ = gradientMachine;
  if (!gradientMachine) {
    CHECK(config_->getConfig().has_model_config())
        << "Missing model_config in trainer_config";
    gradientMachine_.reset(
        GradientMachine::create(config_->getConfig().model_config(),
                                intconfig_->mode,
                                parameterUpdater_->getParameterTypes()));
  }
}

void TrainerInternal::trainOneBatch(int64_t batchId,
                                    const DataBatch& dataBatch,
                                    std::vector<Argument>* outArgs) {
  // true means updating parameter whenever gradient is ready during backward()
  bool doPipelineUpdate =
      (intconfig_->mode != GradientMachine::kSgdSparseCpuTraining) &&
      (intconfig_->local || intconfig_->use_gpu ||
       intconfig_->trainer_count <= 1);

  int64_t actualBatchSize = dataBatch.getSize();
  if (actualBatchSize == 0) {
    return;
  }

  bool showStats = intconfig_->show_param_stats_period > 0 &&
                   (batchId + 1) % intconfig_->show_param_stats_period == 0 &&
                   intconfig_->trainer_id == 0;

  std::vector<ParaStat> paraStats;
  if (showStats) {
    paraStats.resize(gradientMachine_->getParameters().size());
  }

  const std::vector<Argument>& inArgs = dataBatch.getStreams();

  PassType passType = parameterUpdater_->startBatch(actualBatchSize);

  if (config_->getOptConfig().use_sparse_remote_updater()) {
    REGISTER_TIMER("prefetch");
    gradientMachine_->prefetch(inArgs);
    parameterUpdater_->getParametersRemote();
  }

  UpdateCallback updateCallback = [this, showStats, &paraStats](
      Parameter* para) {
    if (showStats) {
      //! @TODO(yuyang18) Show stats is actually a ParameterHook, refactor
      // it
      //! to ParameterHook.
      auto& grad = para->getBuf(PARAMETER_GRADIENT);
      SetDevice device(para->getDeviceId());
      paraStats[para->getID()].avgAbsGrad = grad->getAbsSum() / para->getSize();
      paraStats[para->getID()].maxAbsGrad = grad->getAbsMax();
    }
    parameterUpdater_->update(para);
  };

  {
#ifndef PADDLE_DISABLE_TIMER
    Timer timer;
    timer.start();
#endif
    REGISTER_TIMER("forwardBackward");
    forwardBackwardBatch(
        inArgs, *outArgs, passType, updateCallback, doPipelineUpdate);
#ifndef PADDLE_DISABLE_TIMER
    timer.stop();
    parameterUpdater_->setForwardbackwardTime(timer.get());
#endif
  }

  if (!doPipelineUpdate) {
    auto& parameters = gradientMachine_->getNonStaticParameters();
    for (auto& para : parameters) {
      updateCallback(para.get());
    }
  }

  real cost = 0;
  {
    REGISTER_TIMER("sumCost");
    cost = Argument::sum(*outArgs);
  }

  if (batchId % intconfig_->log_period == 0) {
    currentEvaluator_->start();
    stats_->resetCurrentStat();
  }
  {
    REGISTER_TIMER("eval");
    gradientMachine_->eval(currentEvaluator_);
    gradientMachine_->eval(evaluator_);
  }

  *stats_ += {actualBatchSize, cost};
  {
    REGISTER_TIMER("finishBatch");
    parameterUpdater_->finishBatch(cost);
  }

  if (showStats) {
    showParameterStats(paraStats);
  }
  if ((batchId + 1) % intconfig_->log_period == 0) {
    currentEvaluator_->finish();

    if (intconfig_->dot_period > 0) {
      std::cerr << std::endl;
    }
    LOG(INFO) << " Batch=" << batchId + 1 << " " << *stats_
              << " Eval: " << *evaluator_
              << " CurrentEval: " << *currentEvaluator_;
  } else if (intconfig_->dot_period > 0 &&
             (batchId + 1) % intconfig_->dot_period == 0) {
    std::cerr << ".";
  }
}

/**
 * finish train pass
 */
void TrainerInternal::finishTrainPass(int passId, int batchId) {
  gradientMachine_->onPassEnd();
  parameterUpdater_->finishPass();
  evaluator_->finish();
  LOG(INFO) << " Pass=" << passId << " Batch=" << batchId << " "
            << stats_->getStats(false /*without current cost*/)
            << " Eval: " << *evaluator_;
}

void TrainerInternal::showParameterStats(
    const std::vector<ParaStat>& paraStats) {
  std::vector<ParameterPtr>& parameters = gradientMachine_->getParameters();
  for (auto& parameter : parameters) {
    SetDevice device(parameter->getDeviceId());
    real sum = parameter->getBuf(PARAMETER_VALUE)->getAbsSum();
    const auto& lr = parameter->getBuf(PARAMETER_LEARNING_RATE);
    std::ostringstream osLrHistogram;
    if (lr) {
      if (VLOG_IS_ON(2)) {
        osLrHistogram << " lr_histogram: ";
        lr->histogram(osLrHistogram);
      } else {
        osLrHistogram << " max_lr=" << std::setw(11) << lr->getMax()
                      << " min_lr=" << std::setw(11) << lr->getMin()
                      << " avg_lr=" << std::setw(11)
                      << lr->getSum() / parameter->getSize();
      }
    }
    int pid = parameter->getID();
    LOG(INFO) << std::setiosflags(std::ios::left) << std::setfill(' ')
              << std::setw(20) << parameter->getName()
              << " avg_abs_val=" << std::setw(11) << sum / parameter->getSize()
              << " max_val=" << std::setw(11)
              << parameter->getBuf(PARAMETER_VALUE)->getAbsMax()
              << " avg_abs_grad=" << std::setw(11) << paraStats[pid].avgAbsGrad
              << " max_grad=" << std::setw(11) << paraStats[pid].maxAbsGrad
              << osLrHistogram.str();
  }
}

void TrainerInternal::createParameterUpdater(bool testing) {
  const std::string& alg = config_->getOptConfig().algorithm();
  parameterUpdater_.reset(ParameterUpdaterCreators::tryCreateUpdater(
      alg, config_->getOptConfig(), intconfig_->local, intconfig_->num_passes));
  if (parameterUpdater_) {
    return;
  }

  if (!intconfig_->local) {
    if (testing && config_->getOptConfig().use_sparse_remote_updater()) {
      std::unique_ptr<ParameterUpdater> localUpdater;
      localUpdater.reset(
          new SgdLocalUpdater(config_->getOptConfig()));  // do nothing
      parameterUpdater_.reset(
          new SparseRemoteParameterUpdaterComposite(config_->getOptConfig(),
                                                    intconfig_->num_passes,
                                                    testing,
                                                    std::move(localUpdater)));
    } else {
      if (GradientMachine::kSgdSparseCpuTraining == intconfig_->mode &&
          !intconfig_->use_old_updater) {
        intconfig_->use_old_updater = true;
        LOG(INFO) << "Sgd sparse training can not work with"
                  << " ConcurrentRemoteParameterUpdater,"
                  << " automatically reset --use_old_updater=true";
      }

      std::unique_ptr<ParameterUpdater> localUpdater;
      if (config_->getOptConfig().num_batches_per_send_parameter() > 1) {
        CHECK(alg == TrainAlgorithm::SGD || alg == TrainAlgorithm::AsyncSGD)
            << "Unsupported algorithm in remote-local mode: " << alg;
        if (GradientMachine::kSgdSparseCpuTraining == intconfig_->mode) {
          localUpdater.reset(new SgdThreadUpdater(*config_));
        } else {
          localUpdater.reset(new SgdLocalUpdater(*config_));
        }
      }

      localUpdater.reset(
          intconfig_->use_old_updater
              ? new RemoteParameterUpdater(
                    *config_, intconfig_->num_passes, std::move(localUpdater))
              : new ConcurrentRemoteParameterUpdater(
                    *config_, intconfig_->num_passes, std::move(localUpdater)));

      if (config_->getOptConfig().use_sparse_remote_updater()) {
        localUpdater.reset(
            new SparseRemoteParameterUpdaterComposite(*config_,
                                                      intconfig_->num_passes,
                                                      testing,
                                                      std::move(localUpdater)));
      }

      this->parameterUpdater_ = std::move(localUpdater);
    }
  } else {
    CHECK_EQ(config_->getOptConfig().num_batches_per_send_parameter(), 1)
        << "num_batches_per_send_parameter should be one in local mode!";

    if (GradientMachine::kSgdSparseCpuTraining == intconfig_->mode) {
      parameterUpdater_.reset(new SgdThreadUpdater(*config_));
    } else if (alg == TrainAlgorithm::SGD || alg == TrainAlgorithm::AsyncSGD) {
      if (config_->getModelConfig().type() == "recursive_nn") {
        parameterUpdater_.reset(new SgdCpuUpdater(*config_));
      } else if (intconfig_->use_gpu &&
                 config_->getOptConfig().do_average_in_cpu() &&
                 config_->getOptConfig().average_window() > 0) {
        parameterUpdater_.reset(new SgdUpdaterWithCpuAverager(*config_));
      } else {
        parameterUpdater_.reset(new SgdLocalUpdater(*config_));
      }
    } else {
      LOG(FATAL) << "Unsupported algorithm in local mode: " << alg;
    }
  }
}

void TrainerInternal::forwardBackwardBatch(const std::vector<Argument>& inArgs,
                                           std::vector<Argument>& outArgs,
                                           PassType& passType,
                                           UpdateCallback updateCallback,
                                           bool doPipelineUpdate) {
  gradientMachine_->forwardBackward(
      inArgs, &outArgs, passType, doPipelineUpdate ? updateCallback : nullptr);
}

}  // namespace paddle
