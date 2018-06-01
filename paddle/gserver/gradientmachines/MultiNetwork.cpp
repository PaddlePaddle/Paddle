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

#include <algorithm>
#include "paddle/utils/Stat.h"
#include "paddle/utils/Util.h"

#include "MultiNetwork.h"

#include "NeuralNetwork.h"
#include "ParallelNeuralNetwork.h"

namespace paddle {

void MultiNetwork::init(const ModelConfig& config,
                        ParamInitCallback callback,
                        const std::vector<ParameterType>& parameterTypes,
                        bool useGpu) {
  CHECK_GT(config.sub_models_size(), 1) << "sub_models_size should GT 1";
  // check submodel[0] is root
  CHECK_EQ("root", config.sub_models(0).name())
      << "sub_models(0) should be root";
  // ignore root
  subNetworks_.resize(config.sub_models_size() - 1);
  // base class
  NeuralNetwork::init(config, callback, parameterTypes, useGpu);
  // sub networks
  for (int i = 1; i < config.sub_models_size(); ++i) {
    std::string subModelName = config.sub_models(i).name();
    if (FLAGS_parallel_nn) {
      subNetworks_[i - 1] = std::unique_ptr<ParallelNeuralNetwork>(
          new ParallelNeuralNetwork(subModelName, this));
    } else {
      subNetworks_[i - 1] = std::unique_ptr<NeuralNetwork>(
          NeuralNetwork::newNeuralNetwork(subModelName, this));
    }
    subNetworks_[i - 1]->init(config);
  }
}

void MultiNetwork::prefetch(const std::vector<Argument>& inArgs) {
  std::vector<std::vector<Argument>> argumentGroups;
  Argument::splitByDataId(inArgs, &argumentGroups);
  // check group size is equal to sub network size
  CHECK_EQ(argumentGroups.size(), subNetworks_.size());
  for (size_t i = 0; i < subNetworks_.size(); i++) {
    if (argumentGroups[i].size() == 1 && argumentGroups[i][0].dataId == -1) {
      // check input args: if dataId is -1, then skip this sub network
      continue;
    }
    subNetworks_[i]->prefetch(argumentGroups[i]);
  }
}

void MultiNetwork::forward(const std::vector<Argument>& inArgs,
                           std::vector<Argument>* outArgs,
                           PassType passType) {
  // split inArgs to several vectors
  std::vector<std::vector<Argument>> argumentGroups;
  Argument::splitByDataId(inArgs, &argumentGroups);

  // check group size is equal to sub network size
  CHECK_EQ(argumentGroups.size(), subNetworks_.size());
  std::vector<Argument> tempOutArgs;
  outArgs->clear();

  for (size_t i = 0; i < subNetworks_.size(); i++) {
    tempOutArgs.clear();
    if (argumentGroups[i].size() == 1 && argumentGroups[i][0].dataId == -1) {
      // check input args: if dataId is -1, then skip this sub network
      continue;
    }
    subNetworks_[i]->forward(argumentGroups[i], &tempOutArgs, passType);
    for (const auto& elem : tempOutArgs) {
      outArgs->push_back(elem);
      outArgs->back().dataId = i;
    }
  }
}

void MultiNetwork::backward(const UpdateCallback& callback) {
  for (size_t i = 0; i < subNetworks_.size(); i++) {
    subNetworks_[i]->backward(callback);
  }
}

void MultiNetwork::forwardBackward(const std::vector<Argument>& inArgs,
                                   std::vector<Argument>* outArgs,
                                   PassType passType,
                                   const UpdateCallback& callback) {
  forward(inArgs, outArgs, passType);
  backward(callback);
}

void MultiNetwork::onPassEnd() {
  for (size_t i = 0; i < subNetworks_.size(); i++) {
    subNetworks_[i]->onPassEnd();
  }
}

void MultiNetwork::start() {
  for (auto& subNetwork : subNetworks_) {
    subNetwork->start();
  }
}

void MultiNetwork::finish() {
  for (size_t i = 0; i < subNetworks_.size(); i++) {
    subNetworks_[i]->finish();
  }
}

class MultiCombinedEvaluator : public Evaluator {
 public:
  MultiCombinedEvaluator() {}
  void addEvaluator(std::unique_ptr<Evaluator>&& evaluator) {
    evaluators_.emplace_back(std::move(evaluator));
  }
  virtual void start() {
    for (auto& evaluator : evaluators_) {
      evaluator->start();
    }
  }

  virtual void finish() {
    for (auto& evaluator : evaluators_) {
      evaluator->finish();
    }
  }

  virtual void eval(const NeuralNetwork& nn) {
    const MultiNetwork& multiNetwork = dynamic_cast<const MultiNetwork&>(nn);
    CHECK_EQ(evaluators_.size(), multiNetwork.getSubNetworks().size());
    int size = evaluators_.size();
    for (int i = 0; i < size; i++) {
      // one evaluator for one subNetwork
      evaluators_[i]->eval(*multiNetwork.getSubNetworks()[i]);
    }
  }

  virtual real evalImp(std::vector<Argument>& arguments) {
    (void)arguments;
    return -1;
  }

  virtual void printStats(std::ostream& os) const {
    for (auto& evaluator : evaluators_) {
      evaluator->printStats(os);
      os << ' ';
    }
  }

  virtual void distributeEval(ParameterClient2* client) {
    for (auto& evaluator : evaluators_) {
      evaluator->distributeEval(client);
    }
  }

 protected:
  std::vector<std::unique_ptr<Evaluator>> evaluators_;
};

Evaluator* MultiNetwork::makeEvaluator() const {
  MultiCombinedEvaluator* multiCombinedEvaluator = new MultiCombinedEvaluator();
  for (size_t i = 0; i < subNetworks_.size(); i++) {
    std::unique_ptr<Evaluator> evaluator(subNetworks_[i]->makeEvaluator());
    multiCombinedEvaluator->addEvaluator(std::move(evaluator));
  }
  return multiCombinedEvaluator;
}

void MultiNetwork::eval(Evaluator* evaluator) const { evaluator->eval(*this); }

}  // namespace paddle
