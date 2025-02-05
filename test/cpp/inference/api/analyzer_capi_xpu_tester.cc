/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <string>
#include <vector>

#include "paddle/common/enforce.h"
#include "paddle/fluid/inference/capi/paddle_c_api.h"
#include "test/cpp/inference/api/tester_helper.h"

namespace paddle {
namespace inference {
namespace analysis {

#ifdef PADDLE_WITH_XPU
TEST(PD_AnalysisConfig, use_xpu) {
  std::string model_dir = FLAGS_infer_model + "/mobilenet";
  PD_AnalysisConfig *config = PD_NewAnalysisConfig();
  PD_SwitchSpecifyInputNames(config, true);
  PD_SwitchIrDebug(config, true);
  PD_SetModel(config, model_dir.c_str(), nullptr);
  PD_SetOptimCacheDir(config, (FLAGS_infer_model + "/OptimCacheDir").c_str());
  const char *model_dir_ = PD_ModelDir(config);
  LOG(INFO) << model_dir_;
  PD_EnableXpu(config, 0xfffc00);
  bool use_xpu = PD_UseXpu(config);
  PADDLE_ENFORCE_EQ(use_xpu, true, common::errors::PreconditionNotMet("NO"));
  int device = PD_XpuDeviceId(config);
  PADDLE_ENFORCE_EQ(device, 0, common::errors::PreconditionNotMet("NO"));
  PD_SwitchIrOptim(config, true);
  bool ir_optim = PD_IrOptim(config);
  PADDLE_ENFORCE_EQ(ir_optim, true, common::errors::PreconditionNotMet("NO"));
  PD_EnableMemoryOptim(config);
  bool memory_optim_enable = PD_MemoryOptimEnabled(config);
  PADDLE_ENFORCE_EQ(
      memory_optim_enable, true, common::errors::PreconditionNotMet("NO"));
  PD_EnableProfile(config);
  bool profiler_enable = PD_ProfileEnabled(config);
  PADDLE_ENFORCE_EQ(
      profiler_enable, true, common::errors::PreconditionNotMet("NO"));
  PD_SetInValid(config);
  bool is_valid = PD_IsValid(config);
  PADDLE_ENFORCE_EQ(is_valid, false, common::errors::PreconditionNotMet("NO"));
  PD_DeleteAnalysisConfig(config);
}
#endif

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
