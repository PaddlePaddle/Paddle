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

#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "paddle/fluid/inference/capi/c_api.h"
#include "paddle/fluid/inference/capi/c_api_internal.h"
#include "paddle/fluid/inference/tests/api/tester_helper.h"

namespace paddle {
namespace inference {
namespace analysis {

void PD_run() {
  PD_AnalysisConfig* config = PD_NewAnalysisConfig();
  std::string prog_file = "";
  std::string params_file = "";
  PD_SetModel(config, prog_file.c_str(), params_file.c_str());
  PD_SetProgFile(config, prog_file.c_str());
  PD_SetParamsFile(config, params_file.c_str());
  LOG(INFO) << PD_ProgFile(config);
  LOG(INFO) << PD_ParamsFile(config);
  PD_Tensor* input = PD_NewPaddleTensor();
  PD_PaddleBuf* buf = PD_NewPaddleBuf();
  LOG(INFO) << "PaddleBuf empty: " << PD_PaddleBufEmpty(buf);
  int batch = 1;
  int channel = 3;
  int height = 224;
  int width = 224;
  int shape[4] = [ batch, channel, height, width ];
  int shape_size = 4;
  float* data = new float[batch * channel * height * width];
  PD_PaddleBufReset(buf, static_cast<void*>(data),
                    sizeof(float) * (batch * channel * height * width));

  char name[6] = {'i', 'm', 'a', 'g', 'e', '\0'};
  PD_SetPaddleTensorName(input, name);
  PD_SetPaddleTensorDType(input, PD_FLOAT32);
  PD_SetPaddleTensorShape(input, shape, shape_size);
  PD_SetPaddleTensorData(input, buf);

  PD_Tensor* out_data;
  int* out_size;
  PD_PredictorZeroCopyRun(config, input, 1, out_data, &out_size);
}

TEST(PD_Tensor, PD_run) { PD_run(); }

TEST(SetModelBuffer, read) {}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
