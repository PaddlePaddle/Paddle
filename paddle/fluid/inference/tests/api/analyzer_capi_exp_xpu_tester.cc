/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/inference/capi_exp/pd_inference_api.h"
#include "paddle/fluid/inference/tests/api/tester_helper.h"

namespace paddle {
namespace inference {
namespace analysis {

#ifdef PADDLE_WITH_XPU
TEST(PD_Config, use_xpu) {
  std::string model_dir = FLAGS_infer_model + "/mobilenet";
  PD_Config *config = PD_Config();
  PD_ConfigSwitchIrDebug(config, TRUE);
  PD_ConfigSetModelDir(config, model_dir.c_str());
  PD_ConfigSetOptimCacheDir(config,
                            (FLAGS_infer_model + "/OptimCacheDir").c_str());
  const char *model_dir_ = PD_ConfigGetModelDir(config);
  LOG(INFO) << model_dir_;
  PD_ConfigEnableXpu(config, 0xfffc00);
  bool use_xpu = PD_ConfigUseXpu(config);
  EXPECT_TRUE(use_xpu);
  int32_t device_id = PD_ConfigXpuDeviceId(config);
  EXPECT_EQ(devive_id, 0);
  PD_ConfigSwitchIrOptim(config, TRUE);
  bool ir_optim = PD_IrOptim(config);
  EXPECT_TRUE(ir_optim);
  PD_ConfigEnableMemoryOptim(config);
  bool memory_optim_enable = PD_ConfigMemoryOptimEnabled(config);
  EXPECT_TRUE(memory_optim_enable);
  PD_ConfigEnableProfile(config);
  bool profiler_enable = PD_ConfigProfileEnabled(config);
  EXPECT_TRUE(profiler_enable);
  PD_SetInValid(config);
  bool is_valid = PD_ConfigIsValid(config);
  EXPECT_FALSE(is_valid);
  PD_ConfigDestroy(config);
}
#endif

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
