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

#include "ParamUtil.h"

#include <fenv.h>
#include <stdio.h>

#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

#include <google/protobuf/text_format.h>
#include <paddle/utils/Version.h>

#include "paddle/utils/GlobalConstants.h"
#include "paddle/utils/PythonUtil.h"
#include "paddle/utils/Stat.h"
#include "paddle/utils/Util.h"

#include "TesterConfig.h"
#include "paddle/gserver/gradientmachines/NeuralNetwork.h"
#include "paddle/gserver/layers/ValidationLayer.h"

namespace paddle {

ParameterUtil::ParameterUtil(
    const std::shared_ptr<TrainerConfigHelper> &config,
    std::unique_ptr<ParameterUtilConfig> &&intconfig,
    const GradientMachinePtr &gradientMachine,
    const std::shared_ptr<ParameterUpdater> &parameterUpdater) {
  config_ = config;
  intConfig_ = std::move(intconfig);
  gserver_ = gradientMachine;
  pUpdater_ = parameterUpdater;
}

bool ParameterUtil::loadParameters(int passId, bool local, bool remote) {
  constexpr int kBufLen = 100;
  char buf[kBufLen];
  snprintf(buf, kBufLen, "pass-%05d", passId);
  std::string doneFile = path::join(config_->getSaveDir(), buf, "done");
  if (!fileExist(doneFile.c_str())) return false;
  loadParametersWithPath(path::join(config_->getSaveDir(), buf), local, remote);
  return true;
}

void ParameterUtil::loadParametersWithPath(const std::string &dir,
                                           bool local,
                                           bool remote) {
  if (local) {
    gserver_->loadParameters(dir);
  }
  if (remote && pUpdater_) {
    pUpdater_->loadParametersRemote(dir);
  }
}

void ParameterUtil::saveParametersOnePass(int passId, int passInnerId) {
  pUpdater_->apply();
  saveParameters(passId, passInnerId);
  if (intConfig_->save_only_one_ && passId >= intConfig_->saving_period_) {
    deleteParameters(passId - intConfig_->saving_period_);
  }
  pUpdater_->restore();
}

void ParameterUtil::saveParameters(int passId, int passInnerId) {
  constexpr int kBufLen = 100;
  char buf[kBufLen];
  if (passInnerId > 0) {
    snprintf(buf, kBufLen, "pass-%05d-%03d", passId, passInnerId);
  } else {
    snprintf(buf, kBufLen, "pass-%05d", passId);
  }

  std::string basePath = config_->getSaveDir();
  if (basePath.find('/') == std::string::npos) {
    basePath = "./" + basePath;
  }
  mkDirRecursively(basePath.c_str());

  std::string saveDir = path::join(basePath, buf);
  mkDir(saveDir.c_str());
  if (!intConfig_->load_save_param_pserver_) {
    pUpdater_->getParametersRemote(true /*full parameter*/,
                                   true /*after apply*/);
  }

  gserver_->saveParameters(saveDir);
  if (intConfig_->load_save_param_pserver_) {
    pUpdater_->saveParametersRemote(saveDir);
  }
  std::string doneFile = path::join(saveDir, "done");
  touchFile(doneFile.c_str());
  std::ofstream out(doneFile);
  version::printVersion(out);
  out.close();
  VLOG(1) << "save dir " << saveDir;
  saveConfigWithPath(saveDir);
}

void ParameterUtil::deleteParameters(int passId, int passInnerId) {
  constexpr int kBufLen = 100;
  char buf[kBufLen];
  const std::string &saveDir = config_->getSaveDir();
  if (passInnerId > 0) {
    snprintf(buf,
             kBufLen,
             "%s/pass-%05d-%03d",
             saveDir.c_str(),
             passId,
             passInnerId);
  } else {
    snprintf(buf, kBufLen, "%s/pass-%05d", saveDir.c_str(), passId);
  }
  mkDir(saveDir.c_str());
  LOG(INFO) << "delete dir " << buf;
  rmDir(buf);
}

void ParameterUtil::saveConfigWithPath(const std::string &path) {
  std::string src;
  // save config in some path
  if (!intConfig_->config_.empty()) {
    src = intConfig_->config_;
  } else {
    bool ok;
    src = config_->getConfigName(&ok);
    if (!ok) {
      return;
    }
  }
  copyFileToPath(src, path);

  // save other import config file name to path.txt
  std::string ss = path::join(path, "path.txt");
  std::ofstream os(ss);
  std::string fileName = path::basename(src);
  CHECK(os.write(fileName.c_str(), fileName.length()))
      << "Fail to write config file name " << ss;
  VLOG(1) << "fileName " << fileName;
  os.close();

  // copy other import config files
  for (int i = 0; i < config_->getConfig().config_files_size(); ++i) {
    copyFileToPath(config_->getConfig().config_files(i), path);
  }
}

}  // namespace paddle
