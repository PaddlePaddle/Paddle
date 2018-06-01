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

#include <fenv.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "capi_private.h"
#include "main.h"
#include "paddle/trainer/TrainerConfigHelper.h"
#include "paddle/utils/Excepts.h"
#include "paddle/utils/PythonUtil.h"

static void initPaddle(int argc, char** argv) {
  paddle::initMain(argc, argv);
  paddle::initPython(argc, argv);
}

extern "C" {
paddle_error paddle_init(int argc, char** argv) {
  static bool isInit = false;
  if (isInit) return kPD_NO_ERROR;

  std::vector<char*> realArgv;
  realArgv.reserve(argc + 1);
  realArgv.push_back(strdup(""));
  for (int i = 0; i < argc; ++i) {
    realArgv.push_back(argv[i]);
  }
  initPaddle(argc + 1, realArgv.data());
  free(realArgv[0]);
  isInit = true;
  return kPD_NO_ERROR;
}

paddle_error paddle_init_thread() {
  if (FLAGS_use_gpu) {
    hl_init(FLAGS_gpu_id);
  }
  return kPD_NO_ERROR;
}
}
