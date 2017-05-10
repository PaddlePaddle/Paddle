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

#include <fenv.h>
#include "paddle/pserver/ParameterServerController.h"
#include "paddle/utils/PythonUtil.h"

#include "ParamUtil.h"
#include "Trainer.h"

DEFINE_bool(start_pserver, false, "Whether to start pserver");
DECLARE_int32(gpu_id);
DEFINE_string(job, "train", "one of (train, test, checkgrad)");
DECLARE_int32(start_pass);
DECLARE_string(config);
DECLARE_string(init_model_path);
DECLARE_string(rdma_tcp);

using namespace paddle;  // NOLINT

int main(int argc, char** argv) {
  // write logs instantly (never buffer log messages)
  FLAGS_logbuflevel = -1;

  initMain(argc, argv);
  initPython(argc, argv);

  std::unique_ptr<ParameterServerController> parameterServerPtr(nullptr);
  if (FLAGS_start_pserver) {
    parameterServerPtr.reset(
        paddle::ParameterServerController::createFromGflags());
    parameterServerPtr->start();
  }
  Trainer trainer;
  auto config = TrainerConfigHelper::createFromFlags();
  CHECK(config != nullptr) << "no valid config";

  feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);
  trainer.init(config, FLAGS_job == "test");

  if (FLAGS_job == "train") {
    trainer.train();
  } else if (FLAGS_job == "checkgrad") {
    trainer.checkGradient();
  } else if (FLAGS_job == "test") {
    trainer.test();
  } else if (FLAGS_job == "time") {
    trainer.time();
  } else {
    LOG(FATAL) << "Unknown job type: " << FLAGS_job;
  }

  return 0;
}
