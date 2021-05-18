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
#include <pd_inference_api.h>
#include <stdio.h>
#include <stdlib.h>

void ReadData(float* data, int size);

int main(int argc, char* argv[]) {
  PD_Config* config = PD_ConfigCreate();
  PD_ConfigSetModel(config, "data/model/__model__", "data/model/__params__");
  PD_ConfigDisableGlogInfo(config);

  PD_Predictor* predictor = PD_PredictorCreate(config);
  // config has destroyed in PD_PredictorCreate
  config = NULL;

  int input_num = PD_PredictorGetInputNum(predictor);
  printf("Input num: %d\n", input_num);
  int output_num = PD_PredictorGetOutputNum(predictor);
  printf("Output num: %d\n", output_num);

  PD_OneDimArrayCstr* input_names = PD_PredictorGetInputNames(predictor);
  PD_Tensor* input_tensor =
      PD_PredictorGetInputHandle(predictor, input_names->data[0]);
  PD_OneDimArrayCstrDestroy(input_names);
  input_names = NULL;

  int32_t shape[] = {1, 3, 300, 300};
  float* data = (float*)malloc(sizeof(float) * 1 * 3 * 300 * 300);  // NOLINT
  ReadData(data, 1 * 3 * 300 * 300);                                // NOLINT
  PD_TensorReshape(input_tensor, 4, shape);
  PD_TensorCopyFromCpuFloat(input_tensor, data);
  free(data);
  data = NULL;
  PD_PredictorRun(predictor);

  PD_OneDimArrayCstr* output_names = PD_PredictorGetOutputNames(predictor);
  PD_Tensor* output_tensor =
      PD_PredictorGetOutputHandle(predictor, output_names->data[0]);
  PD_OneDimArrayCstrDestroy(output_names);
  output_names = nullptr;

  PD_OneDimArrayInt32* out_shape = PD_TensorGetShape(output_tensor);
  int32_t size = 1;
  for (size_t index = 0; index < out_shape->size; ++index) {
    size = size * out_shape->data[index];
  }
  PD_OneDimArrayInt32Destroy(out_shape);
  out_shape = NULL;

  data = (float*)malloc(sizeof(float) * size);  // NOLINT
  PD_TensorCopyToCpuFloat(output_tensor, data);
  free(data);
  data = NULL;

  PD_TensorDestroy(output_tensor);
  output_tensor = NULL;
  PD_TensorDestroy(input_tensor);
  input_tensor = NULL;
  PD_PredictorDestroy(predictor);
  predictor = NULL;

  return 0;
}

void ReadData(float* data, int n) {
  FILE* fp = fopen("data/data.txt", "r");
  for (int i = 0; i < n; i++) {
    fscanf(fp, "%f", &data[i]);
  }
  fclose(fp);
}
