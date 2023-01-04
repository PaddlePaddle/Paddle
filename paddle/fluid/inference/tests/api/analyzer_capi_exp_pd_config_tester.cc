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

TEST(PD_Config, interface) {
  std::string model_dir = FLAGS_infer_model + "/mobilenet";
  std::string prog_file = model_dir + "/__model__";
  std::string param_file = model_dir + "/__params__";
  std::string opt_cache_dir = FLAGS_infer_model + "/OptimCacheDir";

  PD_Config* config = PD_ConfigCreate();
  PD_ConfigSetModelDir(config, model_dir.c_str());
  std::string model_dir_ = PD_ConfigGetModelDir(config);
  EXPECT_EQ(model_dir, model_dir_);

  PD_ConfigSetModel(config, prog_file.c_str(), param_file.c_str());
  PD_ConfigSetProgFile(config, prog_file.c_str());
  PD_ConfigSetParamsFile(config, param_file.c_str());
  PD_ConfigSetOptimCacheDir(config, opt_cache_dir.c_str());
  std::string prog_file_ = PD_ConfigGetProgFile(config);
  std::string param_file_ = PD_ConfigGetParamsFile(config);
  EXPECT_EQ(prog_file, prog_file_);
  EXPECT_EQ(param_file, param_file_);

  PD_ConfigDisableFCPadding(config);
  bool fc_padding = PD_ConfigUseFcPadding(config);
  EXPECT_FALSE(fc_padding);

  PD_ConfigDisableGpu(config);
  PD_ConfigSwitchIrOptim(config, TRUE);
  bool ir_optim = PD_ConfigIrOptim(config);
  EXPECT_TRUE(ir_optim);

  PD_ConfigEnableMemoryOptim(config, true);
  bool memory_enabled = PD_ConfigMemoryOptimEnabled(config);
  EXPECT_TRUE(memory_enabled);

#ifndef PADDLE_WITH_LITE
  PD_ConfigEnableLiteEngine(
      config, PD_PRECISION_FLOAT32, TRUE, 0, nullptr, 0, nullptr);
  bool lite_enabled = PD_ConfigLiteEngineEnabled(config);
  EXPECT_TRUE(lite_enabled);
#endif

  PD_ConfigSwitchIrDebug(config, TRUE);
#ifdef PADDLE_WITH_MKLDNN
  const char* ops_name = "conv_2d";
  PD_ConfigEnableMKLDNN(config);
  PD_ConfigSetMkldnnOp(config, 1, &ops_name);
  PD_ConfigSetMkldnnCacheCapacity(config, 100);
  bool mkldnn_enabled = PD_ConfigMkldnnEnabled(config);
  EXPECT_TRUE(mkldnn_enabled);

  PD_ConfigSetCpuMathLibraryNumThreads(config, 10);
  int32_t cpu_threads = PD_ConfigGetCpuMathLibraryNumThreads(config);
  EXPECT_EQ(cpu_threads, 10);

  PD_ConfigEnableMkldnnQuantizer(config);
  bool mkldnn_qt_enabled = PD_ConfigMkldnnQuantizerEnabled(config);
  EXPECT_TRUE(mkldnn_qt_enabled);

  PD_ConfigEnableMkldnnBfloat16(config);
  PD_ConfigSetBfloat16Op(config, 1, &ops_name);
#endif

  PD_ConfigEnableONNXRuntime(config);
  bool onnxruntime_enabled = PD_ConfigONNXRuntimeEnabled(config);
#ifdef PADDLE_WITH_ONNXRUNTIME
  EXPECT_TRUE(onnxruntime_enabled);
#else
  EXPECT_FALSE(onnxruntime_enabled);
#endif
  PD_ConfigDisableONNXRuntime(config);
  bool onnxruntime_disabled = PD_ConfigONNXRuntimeEnabled(config);
  EXPECT_FALSE(onnxruntime_disabled);
  PD_ConfigEnableORTOptimization(config);

  PD_ConfigEnableProfile(config);
  bool profile_enabled = PD_ConfigProfileEnabled(config);
  EXPECT_TRUE(profile_enabled);

  PD_ConfigDisableGlogInfo(config);
  bool glog_diabled = PD_ConfigGlogInfoDisabled(config);
  EXPECT_TRUE(glog_diabled);

  PD_ConfigSetInvalid(config);
  bool is_valid = PD_ConfigIsValid(config);
  EXPECT_FALSE(is_valid);

  PD_ConfigPartiallyRelease(config);
  PD_ConfigDestroy(config);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
