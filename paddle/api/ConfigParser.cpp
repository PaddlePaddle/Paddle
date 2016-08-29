/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

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
#include "paddle/trainer/Trainer.h"

struct TrainerConfigPrivate {
  std::shared_ptr<paddle::TrainerConfig> conf;
  TrainerConfigPrivate() : conf(std::make_shared<paddle::TrainerConfig>()) {}
};

struct ModelConfigPrivate {
  std::shared_ptr<paddle::TrainerConfig> conf;
};

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

struct OptimizationConfigPrivate {
  std::shared_ptr<paddle::TrainerConfig> trainer_config;
  paddle::OptimizationConfig config;

  paddle::OptimizationConfig& getConfig() {
    if (trainer_config != nullptr) {
      return *trainer_config->mutable_opt_config();
    } else {
      return config;
    }
  }
};

TrainerConfig::TrainerConfig() : m(new TrainerConfigPrivate()) {}

TrainerConfig::~TrainerConfig() { delete m; }

TrainerConfig* TrainerConfig::createFromTrainerConfigFile(
    const std::string& confPath) {
  LOG(INFO) << "load trainer config from " << confPath;
  paddle::TrainerConfigHelper helper(confPath);
  //! TODO(yuyang18): Make TrainerConfigPrivate to TrainerConfigHelper
  auto retv = new TrainerConfig();
  *retv->m->conf = helper.getConfig();
  return retv;
}

ModelConfig::ModelConfig() : m(new ModelConfigPrivate()) {}

ModelConfig::~ModelConfig() { delete m; }

ModelConfig* TrainerConfig::getModelConfig() const {
  auto retv = new ModelConfig();
  retv->m->conf = m->conf;
  return retv;
}

void* ModelConfig::getPaddleModelConfig() const {
  return m->conf->mutable_model_config();
}

ParameterConfig::ParameterConfig() : m(new ParameterConfigPrivate()) {}

ParameterConfig::~ParameterConfig() {
  if (m) {
    delete m;
  }
}

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

OptimizationConfig::~OptimizationConfig() {
  if (m) {
    delete m;
  }
}

std::string OptimizationConfig::toProtoString() {
  return m->getConfig().SerializeAsString();
}

OptimizationConfig* TrainerConfig::getOptimizationConfig() const {
  auto opt_config = new OptimizationConfig();
  opt_config->m->trainer_config = m->conf;
  return opt_config;
}

void* OptimizationConfig::getRawPtr() { return &m->getConfig(); }

OptimizationConfig* OptimizationConfig::createFromProtoString(
    const std::string& str) {
  auto conf = new OptimizationConfig();
  conf->m->config.ParseFromString(str);
  return conf;
}
