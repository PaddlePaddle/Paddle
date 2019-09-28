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

#include <algorithm>
#include <map>
#include <vector>
#include "paddle/fluid/inference/capi/c_api.h"
#include "paddle/fluid/inference/capi/c_api_internal.h"

using paddle::ConvertToPaddleDType;
using paddle::ConvertToPlace;
using paddle::ConvertToPDDataType;
using paddle::ConvertToACPrecision;

extern "C" {

bool PD_PredictorRun(PD_Predictor* predictor, PD_Tensor* inputs, int in_size,
                     PD_Tensor* output_data, int** out_size, int batch_size) {
  std::vector<paddle::PaddleTensor> in;
  for (int i = 0; i < in_size; ++i) {
    in.emplace_back(inputs->tensor);
  }
  std::vector<paddle::PaddleTensor> out;
  if (predictor->predictor->Run(in, &out, batch_size)) {
    int osize = out.size();
    for (int i = 0; i < osize; ++i) {
      output_data[i].tensor = out[i];
    }
    *out_size = &osize;
    return true;
  }
  return false;
}

char** PD_GetPredictorInputNames(PD_Predictor* predictor, int** in_size) {
  std::vector<std::string> ret_names;
  ret_names = predictor->predictor->GetInputNames();
  int size = ret_names.size();
  *in_size = &size;
  char** names = new char*[size];
  for (int i = 0; i < size; ++i) {
    std::snprintf(names[i], ret_names[i].length() + 1, "%s",
                  ret_names[i].c_str());
  }
  return names;
}

InTensorShape* PD_GetPredictorInputTensorShape(PD_Predictor* predictor,
                                               int** size) {
  std::map<std::string, std::vector<int64_t>> input_tensor_shape =
      predictor->predictor->GetInputTensorShape();
  InTensorShape* ret_in_tensor_shape =
      new InTensorShape[input_tensor_shape.size()];
  int i = 0;
  for (auto item : input_tensor_shape) {
    std::snprintf(ret_in_tensor_shape[i].name, item.first.length() + 1, "%s",
                  item.first.c_str());
    std::vector<int64_t> tmp_shape = item.second;
    ret_in_tensor_shape[i].shape_size = tmp_shape.size();
    for (int j = 0; j < tmp_shape.size(); ++j) {
      ret_in_tensor_shape[i].tensor_shape[j] = tmp_shape[j];
    }
    ++i;
  }
  *size = &i;
  return ret_in_tensor_shape;
}

char** PD_GetPredictorOutputNames(PD_Predictor* predictor) {
  std::vector<std::string> ret_names;
  ret_names = predictor->predictor->GetOutputNames();
  int size = ret_names.size();
  char** names = new char*[size];
  for (int i = 0; i < size; ++i) {
    std::snprintf(names[i], ret_names[i].length() + 1, "%s",
                  ret_names[i].c_str());
  }
  return names;
}

PD_ZeroCopyTensor* PD_GetPredictorInputTensor(PD_Predictor* predictor,
                                              const char* name) {
  PD_ZeroCopyTensor* ret = new PD_ZeroCopyTensor;
  ret->tensor = predictor->predictor->GetInputTensor(std::string(name));
  return ret;
}

PD_ZeroCopyTensor* PD_GetPredictorOutputTensor(PD_Predictor* predictor,
                                               const char* name) {
  PD_ZeroCopyTensor* ret = new PD_ZeroCopyTensor;
  ret->tensor = predictor->predictor->GetOutputTensor(std::string(name));
  return ret;
}

bool PD_PredictorZeroCopyRun(PD_Predictor* predictor) {
  return predictor->predictor->ZeroCopyRun();
}
bool PD_PredictorZeroCopyRun1(const PD_AnalysisConfig* config, float* inputs,
                              int in_size, float** output, int** out_size,
                              int* shape, int shape_size) {
  auto predictor = paddle::CreatePaddlePredictor(config->config);
  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputTensor(input_names[0]);
  std::vector<int> tensor_shape;
  tensor_shape.assign(shape, shape + shape_size);
  input_t->Reshape(tensor_shape);
  input_t->copy_from_cpu(inputs);
  CHECK(predictor->ZeroCopyRun());

  return true;
}

void PD_DeletePredictor(PD_Predictor* predictor) {
  if (predictor) {
    delete predictor;
    predictor = nullptr;
  }
}

PD_Predictor* PD_ClonePredictor(const PD_Predictor* predictor) {
  PD_Predictor* cloned = new PD_Predictor;
  cloned->predictor = predictor->predictor->Clone();
  return cloned;
}

PD_Predictor* PD_NewPredictor(const PD_AnalysisConfig* config) {
  // auto predictor = paddle::CreatePaddlePredictor(config->config);

  auto predictor = new PD_Predictor;
  predictor->predictor = paddle::CreatePaddlePredictor(config->config);
  return predictor;

  // return static_cast<PD_Predictor*>(predictor);
}
}  // extern "C"
