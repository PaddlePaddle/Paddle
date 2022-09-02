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

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "paddle/fluid/inference/capi_exp/pd_config.h"
#include "paddle/fluid/inference/capi_exp/pd_inference_api.h"
#include "paddle/fluid/inference/capi_exp/pd_utils.h"
#include "paddle/fluid/inference/tests/api/tester_helper.h"

namespace paddle {
namespace inference {
namespace analysis {

void predictor_run() {
  std::string model_dir = FLAGS_infer_model;
  std::string prog_file = model_dir + "/model";
  std::string params_file = model_dir + "/params";
  PD_Config *config = PD_ConfigCreate();
  PD_ConfigDisableGpu(config);
  PD_ConfigSetCpuMathLibraryNumThreads(config, 10);
  PD_ConfigSwitchIrDebug(config, TRUE);
  PD_ConfigSetModel(config, prog_file.c_str(), params_file.c_str());
  PD_Cstr *config_summary = PD_ConfigSummary(config);
  LOG(INFO) << config_summary->data;

  PD_Predictor *predictor = PD_PredictorCreate(config);
  PD_Tensor *tensor = PD_PredictorGetInputHandle(predictor, "data");

  const int batch_size = 1;
  const int channels = 3;
  const int height = 318;
  const int width = 318;
  float *input = new float[batch_size * channels * height * width]();

  int32_t shape[4] = {batch_size, channels, height, width};
  PD_TensorReshape(tensor, 4, shape);
  PD_TensorCopyFromCpuFloat(tensor, input);
  EXPECT_TRUE(PD_PredictorRun(predictor));

  delete[] input;
  PD_TensorDestroy(tensor);
  PD_CstrDestroy(config_summary);
  PD_PredictorDestroy(predictor);
}

TEST(PD_PredictorRun, predictor_run) { predictor_run(); }

#ifdef PADDLE_WITH_MKLDNN
TEST(PD_Config, profile_mkldnn) {
  std::string model_dir = FLAGS_infer_model;
  std::string prog_file = model_dir + "/model";
  std::string params_file = model_dir + "/params";
  PD_Config *config = PD_ConfigCreate();
  PD_ConfigDisableGpu(config);
  PD_ConfigSetCpuMathLibraryNumThreads(config, 10);
  PD_ConfigSwitchIrDebug(config, TRUE);
  PD_ConfigEnableMKLDNN(config);
  bool mkldnn_enable = PD_ConfigMkldnnEnabled(config);
  EXPECT_TRUE(mkldnn_enable);
  PD_ConfigEnableMkldnnQuantizer(config);
  bool quantizer_enable = PD_ConfigMkldnnQuantizerEnabled(config);
  EXPECT_TRUE(quantizer_enable);
  PD_ConfigEnableMkldnnBfloat16(config);
  PD_ConfigSetMkldnnCacheCapacity(config, 0);
  PD_ConfigSetModel(config, prog_file.c_str(), params_file.c_str());
  PD_ConfigDestroy(config);
}
#endif

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
