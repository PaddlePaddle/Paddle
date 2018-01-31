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

#include "TrainerConfigHelper.h"
#include "ParamUtil.h"
#include "TrainerConfig.pb.h"
#include "paddle/utils/Flags.h"
#include "paddle/utils/PythonUtil.h"

DECLARE_string(config);
DECLARE_string(init_model_path);
DECLARE_int32(start_pass);
DECLARE_string(save_dir);
DECLARE_int32(trainer_id);
DECLARE_bool(local);
DECLARE_bool(with_cost);
DECLARE_bool(with_gpu);
DECLARE_bool(parallel_nn);
DECLARE_string(config_args);
DECLARE_bool(use_mkldnn);
DECLARE_bool(use_mkl_packed);

const char *kConfigParserModuleName = "paddle.trainer.config_parser";
const char *kConfigParserFuncName = "parse_config_and_serialize";

namespace paddle {

struct TrainerConfigHelperPrivate {
  TrainerConfig conf;
};

TrainerConfigHelper::TrainerConfigHelper(const std::string &configFilePath)
    : m(new TrainerConfigHelperPrivate()) {
  std::ostringstream configArgs;
  configArgs << "trainer_id=" << FLAGS_trainer_id << ",local=" << FLAGS_local
             << ",with_cost=" << FLAGS_with_cost << ",use_gpu=" << FLAGS_use_gpu
             << ",parallel_nn=" << FLAGS_parallel_nn
             << ",use_mkldnn=" << FLAGS_use_mkldnn
             << ",use_mkl_packed=" << FLAGS_use_mkl_packed
             << ",cudnn_version=" << hl_get_cudnn_lib_version();
  if (!FLAGS_config_args.empty()) {
    configArgs << "," << FLAGS_config_args;
  }

  VLOG(3) << "Parsing trainer config " << configFilePath;
  std::string configProtoStr =
      callPythonFunc(kConfigParserModuleName,
                     kConfigParserFuncName,
                     {configFilePath, configArgs.str()});
  CHECK(m->conf.ParseFromString(configProtoStr));
}

TrainerConfigHelper::TrainerConfigHelper(const TrainerConfig &config)
    : m(new TrainerConfigHelperPrivate()) {
  m->conf = config;
}

TrainerConfigHelper::~TrainerConfigHelper() { delete m; }

const TrainerConfig &TrainerConfigHelper::getConfig() const { return m->conf; }

TrainerConfig &TrainerConfigHelper::getMutableConfig() { return m->conf; }

const OptimizationConfig &TrainerConfigHelper::getOptConfig() const {
  return m->conf.opt_config();
}

const ModelConfig &TrainerConfigHelper::getModelConfig() const {
  return m->conf.model_config();
}

const DataConfig *TrainerConfigHelper::getDataConfigPtr() const {
  if (m->conf.has_data_config()) {
    return &m->conf.data_config();
  } else {
    return nullptr;
  }
}

const DataConfig &TrainerConfigHelper::getTestDataConfig() const {
  CHECK(m->conf.has_test_data_config());
  return m->conf.test_data_config();
}

bool TrainerConfigHelper::hasDataConfig() const {
  return m->conf.has_data_config();
}

bool TrainerConfigHelper::hasTestDataConfig() const {
  return m->conf.has_test_data_config();
}

void TrainerConfigHelper::updateConfigFromFlags() {
  if (!FLAGS_save_dir.empty()) {
    m->conf.set_save_dir(FLAGS_save_dir);
  }
  if (!FLAGS_init_model_path.empty()) {
    m->conf.set_init_model_path(FLAGS_init_model_path);
  }
  if (FLAGS_start_pass != 0) {
    m->conf.set_start_pass(FLAGS_start_pass);
  }
}

void TrainerConfigHelper::disableRemoteSparseUpdater() {
  m->conf.mutable_opt_config()->set_use_sparse_remote_updater(false);
}

void TrainerConfigHelper::disableRemoteSparseUpdaterForEachParams() {
  this->disableRemoteSparseUpdater();
  for (int i = 0; i < m->conf.model_config().parameters_size(); ++i) {
    m->conf.mutable_model_config()
        ->mutable_parameters(i)
        ->set_sparse_remote_update(false);
  }
}

OptimizationConfig &TrainerConfigHelper::getOptConfig() {
  return *m->conf.mutable_opt_config();
}

void TrainerConfigHelper::setSaveDir(const std::string &saveDir) {
  m->conf.set_save_dir(saveDir);
}

const std::string &TrainerConfigHelper::getSaveDir() const {
  return m->conf.save_dir();
}

std::string TrainerConfigHelper::getConfigNameFromPath(
    const std::string &modelPath) {
  std::ifstream s(path::join(modelPath, "path.txt"));
  CHECK(s.is_open()) << " fail to open path.txt";
  std::string ss;
  getline(s, ss);
  VLOG(3) << "fileName " << path::join(modelPath, ss);
  s.close();
  return path::join(modelPath, ss);
}

std::string TrainerConfigHelper::getConfigNameFromPassId(
    int passId, const std::string &modelPath) {
  constexpr int kBufLen = 100;
  char buf[kBufLen];
  snprintf(buf, kBufLen, "pass-%05d", passId);
  return TrainerConfigHelper::getConfigNameFromPath(path::join(modelPath, buf));
}

std::string TrainerConfigHelper::getConfigName(bool *ok) const {
  std::string retv = "";

  if (!m->conf.config_file().empty()) {
    retv = m->conf.config_file();
  } else if (!m->conf.init_model_path().empty()) {
    retv = getConfigNameFromPath(m->conf.init_model_path());
  } else if (m->conf.start_pass() >= 1) {
    retv = getConfigNameFromPassId(m->conf.start_pass(), m->conf.save_dir());
  }

  if (ok) {
    *ok = !retv.empty();
  }

  return retv;
}

std::shared_ptr<TrainerConfigHelper> TrainerConfigHelper::createFromFlags() {
  std::string configPath;
  if (!FLAGS_config.empty()) {
    configPath = FLAGS_config;
  } else if (!FLAGS_init_model_path.empty()) {
    configPath = getConfigNameFromPath(FLAGS_init_model_path);
  } else if (FLAGS_start_pass >= 1) {
    configPath =
        getConfigNameFromPassId(FLAGS_start_pass - 1, FLAGS_init_model_path);
  } else {
    return nullptr;
  }
  return std::make_shared<TrainerConfigHelper>(configPath);
}

std::shared_ptr<TrainerConfigHelper>
TrainerConfigHelper::createFromFlagConfig() {
  CHECK(!FLAGS_config.empty());
  return std::make_shared<TrainerConfigHelper>(FLAGS_config);
}

}  // namespace paddle
