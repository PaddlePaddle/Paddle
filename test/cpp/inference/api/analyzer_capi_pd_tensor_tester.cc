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

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "paddle/common/enforce.h"
#include "paddle/fluid/inference/capi/c_api_internal.h"
#include "paddle/fluid/inference/capi/paddle_c_api.h"
#include "test/cpp/inference/api/tester_helper.h"

namespace paddle {
namespace inference {
namespace analysis {

void PD_run() {
  PD_AnalysisConfig* config = PD_NewAnalysisConfig();
  std::string prog_file = FLAGS_infer_model + "/__model__";
  std::string params_file = FLAGS_infer_model + "/__params__";
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
  int height = 300;
  int width = 300;
  int shape[4] = {batch, channel, height, width};
  int shape_size = 4;
  float* data = new float[batch * channel * height * width];
  PD_PaddleBufReset(buf,
                    static_cast<void*>(data),
                    sizeof(float) * (batch * channel * height * width));

  char name[6] = {'i', 'm', 'a', 'g', 'e', '\0'};
  PD_SetPaddleTensorName(input, name);
  PD_SetPaddleTensorDType(input, PD_FLOAT32);
  PD_SetPaddleTensorShape(input, shape, shape_size);
  PD_SetPaddleTensorData(input, buf);

  PD_Tensor* out_data = PD_NewPaddleTensor();
  int out_size;
  PD_PredictorRun(config, input, 1, &out_data, &out_size, 1);
  LOG(INFO) << out_size;
  LOG(INFO) << PD_GetPaddleTensorName(out_data);
  LOG(INFO) << PD_GetPaddleTensorDType(out_data);
  PD_PaddleBuf* b = PD_GetPaddleTensorData(out_data);
  LOG(INFO) << PD_PaddleBufLength(b) / sizeof(float);
  float* result = static_cast<float*>(PD_PaddleBufData(b));
  LOG(INFO) << *result;
  PD_DeletePaddleTensor(input);
  int size;
  const int* out_shape = PD_GetPaddleTensorShape(out_data, &size);
  PADDLE_ENFORCE_EQ(
      size,
      2,
      common::errors::InvalidArgument("The Output shape's size is NOT match."));
  std::vector<int> ref_outshape_size({9, 6});
  for (int i = 0; i < 2; ++i) {
    PADDLE_ENFORCE_EQ(out_shape[i],
                      ref_outshape_size[i],
                      common::errors::InvalidArgument(
                          "The Output shape's size is NOT match."));
  }
  PD_DeletePaddleBuf(buf);
}

TEST(PD_Tensor, PD_run) { PD_run(); }

TEST(PD_Tensor, int32) {
  PD_Tensor* input = PD_NewPaddleTensor();
  PD_SetPaddleTensorDType(input, PD_INT32);
  LOG(INFO) << PD_GetPaddleTensorDType(input);
}

TEST(PD_Tensor, int64) {
  PD_Tensor* input = PD_NewPaddleTensor();
  PD_SetPaddleTensorDType(input, PD_INT64);
  LOG(INFO) << PD_GetPaddleTensorDType(input);
}

TEST(PD_Tensor, int8) {
  PD_Tensor* input = PD_NewPaddleTensor();
  PD_SetPaddleTensorDType(input, PD_UINT8);
  LOG(INFO) << PD_GetPaddleTensorDType(input);
}

std::string read_file(std::string filename) {
  std::ifstream file(filename);
  return std::string((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
}

void buffer_run() {
  PD_AnalysisConfig* config = PD_NewAnalysisConfig();
  std::string prog_file = FLAGS_infer_model + "/__model__";
  std::string params_file = FLAGS_infer_model + "/__params__";

  std::string prog_str = read_file(prog_file);
  std::string params_str = read_file(params_file);

  PD_SetModelBuffer(config,
                    prog_str.c_str(),
                    prog_str.size(),
                    params_str.c_str(),
                    params_str.size());
  LOG(INFO) << PD_ProgFile(config);
  LOG(INFO) << PD_ParamsFile(config);
  PADDLE_ENFORCE(PD_ModelFromMemory(config),
                 common::errors::PreconditionNotMet(
                     "PD_ModelFromMemory(config) is failed"));

  PD_Tensor* input = PD_NewPaddleTensor();
  PD_PaddleBuf* buf = PD_NewPaddleBuf();
  LOG(INFO) << "PaddleBuf empty: " << PD_PaddleBufEmpty(buf);
  int batch = 1;
  int channel = 3;
  int height = 300;
  int width = 300;
  int shape[4] = {batch, channel, height, width};
  int shape_size = 4;
  float* data = new float[batch * channel * height * width];
  PD_PaddleBufReset(buf,
                    static_cast<void*>(data),
                    sizeof(float) * (batch * channel * height * width));

  char name[6] = {'i', 'm', 'a', 'g', 'e', '\0'};
  PD_SetPaddleTensorName(input, name);
  PD_SetPaddleTensorDType(input, PD_FLOAT32);
  PD_SetPaddleTensorShape(input, shape, shape_size);
  PD_SetPaddleTensorData(input, buf);

  PD_Tensor* out_data = PD_NewPaddleTensor();
  int out_size;
  PD_PredictorRun(config, input, 1, &out_data, &out_size, 1);
  LOG(INFO) << out_size;
  LOG(INFO) << PD_GetPaddleTensorName(out_data);
  LOG(INFO) << PD_GetPaddleTensorDType(out_data);
  PD_PaddleBuf* b = PD_GetPaddleTensorData(out_data);
  LOG(INFO) << PD_PaddleBufLength(b) / sizeof(float);
  float* result = static_cast<float*>(PD_PaddleBufData(b));
  LOG(INFO) << *result;
  PD_DeletePaddleTensor(input);
  PD_DeletePaddleBuf(buf);
}

TEST(SetModelBuffer, read) { buffer_run(); }

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
