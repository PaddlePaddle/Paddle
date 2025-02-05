// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <string>
#include <vector>

#include "paddle/fluid/inference/capi/paddle_c_api.h"
#include "test/cpp/inference/api/tester_helper.h"

namespace paddle {
namespace inference {
namespace analysis {

void SetConfig(PD_AnalysisConfig *config) {
  auto model_dir = FLAGS_infer_model;
  PD_SetModel(config,
              (model_dir + "/__model__").c_str(),
              (model_dir + "/param").c_str());
  PD_SwitchSpecifyInputNames(config, true);
  PD_DisableGpu(config);
}

TEST(PD_ZeroCopyRun, zero_copy_run) {
  PD_AnalysisConfig *config = PD_NewAnalysisConfig();
  SetConfig(config);
  PD_Predictor *predictor = PD_NewPredictor(config);

  int input_num = PD_GetInputNum(predictor);
  printf("Input num: %d\n", input_num);
  int output_num = PD_GetOutputNum(predictor);
  printf("Output num: %d\n", output_num);

  PD_ZeroCopyTensor inputs[2];

  // inputs[0]: word
  PD_InitZeroCopyTensor(&inputs[0]);
  inputs[0].name = new char[5];
  snprintf(inputs[0].name,
           strlen(PD_GetInputName(predictor, 0)) + 1,
           "%s",
           PD_GetInputName(predictor, 0));

  inputs[0].data.capacity = sizeof(int64_t) * 11 * 1;
  inputs[0].data.length = inputs[0].data.capacity;
  inputs[0].data.data = malloc(inputs[0].data.capacity);
  std::vector<int64_t> ref_word(
      {12673, 9763, 905, 284, 45, 7474, 20, 17, 1, 4, 9});
  inputs[0].data.data = reinterpret_cast<void *>(ref_word.data());

  int shape0[] = {11, 1};
  inputs[0].shape.data = reinterpret_cast<void *>(shape0);
  inputs[0].shape.capacity = sizeof(shape0);
  inputs[0].shape.length = sizeof(shape0);
  inputs[0].dtype = PD_INT64;

  size_t lod0[] = {0, 11};
  inputs[0].lod.data = reinterpret_cast<void *>(lod0);
  inputs[0].lod.capacity = sizeof(size_t) * 2;
  inputs[0].lod.length = sizeof(size_t) * 2;

  PD_SetZeroCopyInput(predictor, &inputs[0]);

  // inputs[1]: mention
  PD_InitZeroCopyTensor(&inputs[1]);
  inputs[1].name = new char[8];
  snprintf(inputs[1].name,
           strlen(PD_GetInputName(predictor, 1)) + 1,
           "%s",
           PD_GetInputName(predictor, 1));

  inputs[1].data.capacity = sizeof(int64_t) * 11 * 1;
  inputs[1].data.length = inputs[1].data.capacity;
  inputs[1].data.data = malloc(inputs[1].data.capacity);
  std::vector<int64_t> ref_mention({27, 0, 0, 33, 34, 33, 0, 0, 0, 1, 2});
  inputs[1].data.data = reinterpret_cast<void *>(ref_mention.data());

  int shape1[] = {11, 1};
  inputs[1].shape.data = reinterpret_cast<void *>(shape1);
  inputs[1].shape.capacity = sizeof(shape1);
  inputs[1].shape.length = sizeof(shape1);
  inputs[1].dtype = PD_INT64;

  size_t lod1[] = {0, 11};
  inputs[1].lod.data = reinterpret_cast<void *>(lod1);
  inputs[1].lod.capacity = sizeof(size_t) * 2;
  inputs[1].lod.length = sizeof(size_t) * 2;

  PD_SetZeroCopyInput(predictor, &inputs[1]);

  PD_ZeroCopyRun(predictor);
  PD_ZeroCopyTensor output;
  PD_InitZeroCopyTensor(&output);
  output.name = new char[21];
  snprintf(output.name,
           strlen(PD_GetOutputName(predictor, 0)) + 1,
           "%s",
           PD_GetOutputName(predictor, 0));

  // not necessary, just for coverage tests
  output.lod.data = std::malloc(sizeof(size_t));

  PD_GetZeroCopyOutput(predictor, &output);
  PD_DestroyZeroCopyTensor(&output);
  PD_DeleteAnalysisConfig(config);
  PD_DeletePredictor(predictor);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
