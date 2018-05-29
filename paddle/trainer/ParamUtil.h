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
#include "ParameterUpdater.h"
#include "TrainerConfig.pb.h"
#include "TrainerConfigHelper.h"

namespace paddle {

/**
 * Configuration for parameter utils.
 */
struct ParameterUtilConfig {
  DISABLE_COPY(ParameterUtilConfig);

  ParameterUtilConfig(bool save_only_one,
                      int saving_period,
                      bool load_save_parameters_in_pserver,
                      std::string config)
      : save_only_one_(save_only_one),
        saving_period_(saving_period),
        load_save_param_pserver_(load_save_parameters_in_pserver),
        config_(config) {}

  bool save_only_one_;
  int saving_period_;
  bool load_save_param_pserver_;
  std::string config_;
};

/**
 * ParameterUtil
 * Utility class for loading and saving parameters
 */
class ParameterUtil {
 public:
  /**
   * Ctor.
   *
   * @param config
   * @param intconfig
   * @param gradientMachine
   * @param parameterUpdater
   * @return
   */
  ParameterUtil(const std::shared_ptr<TrainerConfigHelper> &config,
                std::unique_ptr<ParameterUtilConfig> &&intconfig,
                const GradientMachinePtr &gradientMachine,
                const std::shared_ptr<ParameterUpdater> &parameterUpdater);

  /// Load parameter from the saved parameter file as pass passId
  /// if loadsave_parameters_in_pserver is set, some parameters MUST
  /// load in pserver, which is "remote".
  /// loadParameters can choose to load local/remote parameter, or both.
  bool loadParameters(int passId, bool local = true, bool remote = false);

  /// load parameters given path info
  void loadParametersWithPath(const std::string &dir,
                              bool local = true,
                              bool remote = false);

  /// Save parameter to dist for pass passId
  /// passInnerId means saving times in one pass, some users want to
  /// save parameters when have processed some batches in one pass
  /// passInnerId = 0 means do not need to save in one inner pass
  void saveParameters(int passId, int passInnerId = 0);

  /// save parameters for one pass, when passInnerId > 0 means saving
  /// the passInnerId times in one pass
  void saveParametersOnePass(int passId, int passInnerId = 0);

  /// delete parameter from disk via passId
  void deleteParameters(int passId, int passInnerId = 0);

  /// save config given path info
  void saveConfigWithPath(const std::string &path);

  /**
   * Try to load parameter from config.
   * @return true if can load from trainer config.
   */
  inline bool tryLoadParametersFromConfig() {
    auto &c = config_->getConfig();
    if (!c.init_model_path().empty()) {
      loadParametersWithPath(c.init_model_path());
      return true;
    } else if (c.start_pass() > 0) {
      CHECK(loadParameters(c.start_pass() - 1));
      return true;
    } else {
      return false;
    }
  }

 private:
  std::shared_ptr<TrainerConfigHelper> config_;
  std::unique_ptr<ParameterUtilConfig> intConfig_;
  GradientMachinePtr gserver_;
  std::shared_ptr<ParameterUpdater> pUpdater_;
};

}  //  namespace paddle
