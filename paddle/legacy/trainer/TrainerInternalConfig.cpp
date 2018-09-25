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

#include "TrainerInternalConfig.h"

DEFINE_int32(show_parameter_stats_period,
             0,
             "Whether to show parameter stats during training");

DEFINE_int32(dot_period, 1, "Print '.' every so many batches");

DEFINE_bool(use_old_updater, false, "Use the old RemoteParameterUpdater");

DECLARE_int32(num_passes);

DECLARE_bool(local);

namespace paddle {

std::unique_ptr<TrainerInternalConfig> TrainerInternalConfig::createFromMode(
    GradientMachine::CreateMode mode) {
  auto config = new TrainerInternalConfig();
  config->mode = mode;
  config->local = FLAGS_local;
  config->use_gpu = FLAGS_use_gpu;
  config->trainer_count = FLAGS_trainer_count;
  config->show_param_stats_period = FLAGS_show_parameter_stats_period;
  config->trainer_id = FLAGS_trainer_id;
  config->log_period = FLAGS_log_period;
  config->dot_period = FLAGS_dot_period;
  config->num_passes = FLAGS_num_passes;
  config->use_old_updater = FLAGS_use_old_updater;
  config->loadsave_parameters_in_pserver = FLAGS_loadsave_parameters_in_pserver;

  return std::unique_ptr<TrainerInternalConfig>(config);
}

}  // namespace paddle
