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

#include "GradientMachine.h"
#include "NeuralNetwork.h"

#include "paddle/utils/Locks.h"

namespace paddle {

class MultiNetwork : public NeuralNetwork {
 public:
  explicit MultiNetwork(std::string subModelName = "")
      : NeuralNetwork(subModelName) {}

  virtual void init(const ModelConfig& config,
                    ParamInitCallback callback,
                    const std::vector<ParameterType>& parameterTypes,
                    bool useGpu);

  virtual void prefetch(const std::vector<Argument>& inArgs);

  virtual void forward(const std::vector<Argument>& inArgs,
                       std::vector<Argument>* outArgs,
                       PassType passType);

  virtual void backward(const UpdateCallback& callback = nullptr);

  void forwardBackward(const std::vector<Argument>& inArgs,
                       std::vector<Argument>* outArgs,
                       PassType passType,
                       const UpdateCallback& callback);

  virtual void onPassEnd();

  virtual Evaluator* makeEvaluator() const;

  virtual void eval(Evaluator* evaluator) const;

  const std::vector<std::unique_ptr<NeuralNetwork>>& getSubNetworks() const {
    return subNetworks_;
  }

  virtual void start();

  virtual void finish();

 protected:
  std::vector<std::unique_ptr<NeuralNetwork>> subNetworks_;
};
}  // namespace paddle
