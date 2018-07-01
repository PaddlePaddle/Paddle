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

#include "PaddleAPI.h"
#include "PaddleAPIPrivate.h"
#include "paddle/trainer/Trainer.h"

struct ParameterConfigPrivate {
  paddle::ParameterPtr parameter;
  paddle::ParameterConfig config;

  inline paddle::ParameterConfig* getConfigPtr() {
    if (parameter != nullptr) {
      auto& conf = parameter->getConfig();
      return const_cast<paddle::ParameterConfig*>(&conf);
    } else {
      return &config;
    }
  }
};

TrainerConfig::TrainerConfig() : m(new TrainerConfigPrivate()) {}

TrainerConfig::~TrainerConfig() { delete m; }

TrainerConfig* TrainerConfig::createFromTrainerConfigFile(
    const std::string& confPath) {
  LOG(INFO) << "load trainer config from " << confPath;
  auto conf = std::make_shared<paddle::TrainerConfigHelper>(confPath);
  auto retv = new TrainerConfig();
  retv->m->conf = conf;
  return retv;
}

TrainerConfig* TrainerConfig::createFromProtoString(const std::string& str) {
  auto retv = new TrainerConfig();
  paddle::TrainerConfig trainerConfigProto;
  auto conf = std::make_shared<paddle::TrainerConfigHelper>(trainerConfigProto);
  CHECK(conf->getMutableConfig().ParseFromString(str));
  retv->m->conf = conf;
  return retv;
}

ModelConfig::ModelConfig() : m(new ModelConfigPrivate()) {}

ModelConfig::~ModelConfig() { delete m; }

ModelConfig* TrainerConfig::getModelConfig() const {
  auto retv = new ModelConfig();
  retv->m->conf = m->conf;
  return retv;
}

ParameterConfig::ParameterConfig() : m(new ParameterConfigPrivate()) {}

ParameterConfig::~ParameterConfig() { delete m; }

ParameterConfig* ParameterConfig::createParameterConfigFromParameterSharedPtr(
    void* ptr) {
  auto& p = *(paddle::ParameterPtr*)(ptr);
  if (p != nullptr) {
    auto conf = new ParameterConfig();
    conf->m->parameter = p;
    return conf;
  } else {
    return nullptr;
  }
}

ParameterConfig* ParameterConfig::createParameterConfigFromParameterPtr(
    void* ptr) {
  auto& p = *(paddle::Parameter*)(ptr);
  auto conf = new ParameterConfig();
  conf->m->config = p.getConfig();
  return conf;
}

std::string ParameterConfig::toProtoString() const {
  return m->getConfigPtr()->SerializeAsString();
}

void* ParameterConfig::getRawPtr() { return m->getConfigPtr(); }

OptimizationConfig::OptimizationConfig() : m(new OptimizationConfigPrivate()) {}

OptimizationConfig::~OptimizationConfig() { delete m; }

std::string OptimizationConfig::toProtoString() {
  return m->getConfig().SerializeAsString();
}

OptimizationConfig* TrainerConfig::getOptimizationConfig() const {
  auto opt_config = new OptimizationConfig();
  opt_config->m->trainer_config = m->conf;
  return opt_config;
}

OptimizationConfig* OptimizationConfig::createFromProtoString(
    const std::string& str) {
  auto conf = new OptimizationConfig();
  conf->m->config.ParseFromString(str);
  return conf;
}
