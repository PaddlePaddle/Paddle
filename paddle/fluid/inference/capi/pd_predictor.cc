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
#include <numeric>
#include <vector>
#include "paddle/fluid/inference/capi/c_api.h"
#include "paddle/fluid/inference/capi/c_api_internal.h"

using paddle::ConvertToPaddleDType;
using paddle::ConvertToPDDataType;
using paddle::ConvertToACPrecision;

extern "C" {

bool PD_PredictorRun(const PD_AnalysisConfig* config, PD_Tensor* inputs,
                     int in_size, PD_Tensor** output_data, int* out_size,
                     int batch_size) {
  PADDLE_ENFORCE_NOT_NULL(config);
  static std::map<std::string, std::unique_ptr<paddle::PaddlePredictor>>
      predictors;
  if (!predictors.count(config->config.model_dir())) {
    predictors[config->config.model_dir()] =
        paddle::CreatePaddlePredictor(config->config);
  }
  auto& predictor = predictors[config->config.model_dir()];
  std::vector<paddle::PaddleTensor> in;
  for (int i = 0; i < in_size; ++i) {
    in.emplace_back(inputs->tensor);
  }
  std::vector<paddle::PaddleTensor> out;
  if (predictor->Run(in, &out, batch_size)) {
    int osize = out.size();
    *output_data = new PD_Tensor[osize];
    for (int i = 0; i < osize; ++i) {
      output_data[i]->tensor = out[i];
    }
    *out_size = osize;
    return true;
  }
  return false;
}

bool PD_PredictorZeroCopyRun(const PD_AnalysisConfig* config,
                             PD_ZeroCopyData* inputs, int in_size,
                             PD_ZeroCopyData** output, int** out_size) {
  PADDLE_ENFORCE_NOT_NULL(config);
  auto predictor = paddle::CreatePaddlePredictor(config->config);
  auto input_names = predictor->GetInputNames();
  VLOG(3) << "The inputs' size is " << input_names.size();
  PADDLE_ENFORCE_EQ(
      input_names.size(), in_size,
      "The number of input and the number of model's input must match. ");
  for (int i = 0; i < in_size; ++i) {
    auto input_t = predictor->GetInputTensor(inputs[i].name);
    std::vector<int> tensor_shape;
    tensor_shape.assign(inputs[i].shape,
                        inputs[i].shape + inputs[i].shape_size);
    input_t->Reshape(tensor_shape);
    switch (inputs[i].dtype) {
      case PD_FLOAT32:
        input_t->copy_from_cpu(static_cast<float*>(inputs[i].data));
        break;
      case PD_INT32:
        input_t->copy_from_cpu(static_cast<int32_t*>(inputs[i].data));
        break;
      case PD_INT64:
        input_t->copy_from_cpu(static_cast<int64_t*>(inputs[i].data));
        break;
      case PD_UINT8:
        input_t->copy_from_cpu(static_cast<uint8_t*>(inputs[i].data));
        break;
      default:
        CHECK(false) << "Unsupport data type.";
        break;
    }
  }
  CHECK(predictor->ZeroCopyRun());
  auto output_names = predictor->GetOutputNames();
  int osize = output_names.size();
  *out_size = &osize;
  *output = new PD_ZeroCopyData[osize];
  VLOG(3) << "The output size is " << osize;
  for (int i = 0; i < osize; ++i) {
    auto& output_i = (*output)[i];
    output_i.name = new char[output_names[i].length() + 1];
    snprintf(output_i.name, output_names[i].length() + 1, "%s",
             output_names[i].c_str());
    auto output_t = predictor->GetOutputTensor(output_names[i]);
    output_i.dtype = ConvertToPDDataType(output_t->type());
    std::vector<int> output_shape = output_t->shape();
    output_i.shape = new int[output_shape.size()];
    output_i.shape = output_shape.data();
    output_i.shape_size = output_shape.size();
    switch (output_i.dtype) {
      case PD_FLOAT32: {
        std::vector<float> out_data;
        int out_num = std::accumulate(output_shape.begin(), output_shape.end(),
                                      1, std::multiplies<int>());
        out_data.resize(out_num);
        output_t->copy_to_cpu(out_data.data());
        output_i.data = static_cast<void*>(out_data.data());
      } break;
      case PD_INT32: {
        std::vector<int32_t> out_data;
        int out_num = std::accumulate(output_shape.begin(), output_shape.end(),
                                      1, std::multiplies<int>());
        out_data.resize(out_num);
        output_t->copy_to_cpu(out_data.data());
        output_i.data = static_cast<void*>(out_data.data());
      } break;
      case PD_INT64: {
        std::vector<int64_t> out_data;
        int out_num = std::accumulate(output_shape.begin(), output_shape.end(),
                                      1, std::multiplies<int>());
        out_data.resize(out_num);
        output_t->copy_to_cpu(out_data.data());
        output_i.data = static_cast<void*>(out_data.data());
      } break;
      case PD_UINT8: {
        std::vector<uint8_t> out_data;
        int out_num = std::accumulate(output_shape.begin(), output_shape.end(),
                                      1, std::multiplies<int>());
        out_data.resize(out_num);
        output_t->copy_to_cpu(out_data.data());
        output_i.data = static_cast<void*>(out_data.data());
      } break;
      default:
        CHECK(false) << "Unsupport data type.";
        break;
    }
  }
  return true;
}
}  // extern "C"
