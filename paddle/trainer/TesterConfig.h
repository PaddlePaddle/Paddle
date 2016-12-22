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
#include "paddle/gserver/gradientmachines/GradientMachine.h"

#include "TrainerConfig.pb.h"

#include <stdlib.h>
#include <fstream>
#include "ParameterUpdater.h"

namespace paddle {

/**
 * TesterConfig
 * general configs for training
 */
struct TesterConfig {
  /**
   * indicate test period
   */
  int testPeriod;

  /**
   * indicate whether to save previous batch state
   */
  bool prevBatchState;

  /**
   * log period
   */
  int logPeriod;

  /**
   * loadsave parameters in pserver
   */
  bool loadsaveParametersInPserver;

  /**
   * feat file
   */
  std::string featFile;

  /**
   * predict output dir
   */
  std::string predictOutputDir;

  /**
   * trianer id
   */
  int trainerId;

  /**
   * distribute test
   */
  bool distributeTest;

  /**
   * training state
   */
  MachineState* trainState;

  /**
   * test state
   */
  MachineState* testState;

  /**
   * model list
   */
  std::string modelList;

  /**
   * test passes
   */
  int testPass;

  /**
   * num passes
   */
  int numPasses;

  /**
   * saving period
   */
  int savingPeriod;

  /**
   * test wait
   */
  int testWait;

  /**
   * init model path
   */
  std::string initModelPath;

  /**
   * save only one
   */
  bool saveOnlyOne;

  /**
   * testing mode
   */
  bool testing;

  /**
   * mode
   */
  int mode;

  /**
   * config loc
   */
  std::string config;
};

}  //  namespace paddle
