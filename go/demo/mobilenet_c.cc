// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <paddle_c_api.h>
#include <stdio.h>
#include <stdlib.h>

void SetConfig(PD_AnalysisConfig *);
void ReadData(float *data, int size);

int main(int argc, char *argv[]) {
  PD_AnalysisConfig *config = PD_NewAnalysisConfig();
  SetConfig(config);
  PD_Predictor *predictor = PD_NewPredictor(config);

  int input_num = PD_GetInputNum(predictor);
  printf("Input num: %d\n", input_num);
  int output_num = PD_GetOutputNum(predictor);
  printf("Output num: %d\n", output_num);

  PD_ZeroCopyTensor input;
  PD_InitZeroCopyTensor(&input);
  input.name = const_cast<char *>(PD_GetInputName(predictor, 0));  // NOLINT
  input.data.capacity = sizeof(float) * 1 * 3 * 300 * 300;
  input.data.length = input.data.capacity;
  input.data.data = malloc(input.data.capacity);
  int shape[] = {1, 3, 300, 300};
  input.shape.data = static_cast<int *>(shape);
  input.shape.capacity = sizeof(shape);
  input.shape.length = sizeof(shape);
  input.dtype = PD_FLOAT32;
  ReadData((float *)input.data.data, 1 * 3 * 300 * 300);  // NOLINT
  float *data = (float *)input.data.data;                 // NOLINT
  PD_SetZeroCopyInput(predictor, &input);
  int *shape_ptr = (int *)input.shape.data;  // NOLINT

  PD_ZeroCopyRun(predictor);
  PD_ZeroCopyTensor output;
  PD_InitZeroCopyTensor(&output);
  output.name = const_cast<char *>(PD_GetOutputName(predictor, 0));  // NOLINT
  PD_GetZeroCopyOutput(predictor, &output);

  PD_DestroyZeroCopyTensor(&output);

  PD_DeleteAnalysisConfig(config);
  PD_DeletePredictor(predictor);
  return 0;
}

void SetConfig(PD_AnalysisConfig *config) {
  PD_SetModel(config, "data/model/__model__", "data/model/__params__");
  PD_SwitchUseFeedFetchOps(config, false);
  PD_SwitchSpecifyInputNames(config, true);
  PD_DisableGlogInfo(config);
  // PD_SwitchIrOptim(config, false);
}

void ReadData(float *data, int n) {
  FILE *fp = fopen("data/data.txt", "r");
  for (int i = 0; i < n; i++) {
    fscanf(fp, "%f", &data[i]);
  }
  fclose(fp);
}
