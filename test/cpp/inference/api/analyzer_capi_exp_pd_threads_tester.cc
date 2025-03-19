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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "paddle/common/flags.h"
#include "paddle/fluid/inference/capi_exp/pd_inference_api.h"

PD_DEFINE_string(infer_model, "", "model path");

namespace paddle {
namespace inference {
namespace analysis {

typedef struct RunParameter {
  PD_Predictor* predictor;
  int32_t* shapes;
  size_t shape_size;
  float* input_data;
  int32_t out_size;
  std::vector<float> out_data;
  int32_t thread_index;
} RunParameter;

void* run(void* thread_param) {
  struct RunParameter* param = (struct RunParameter*)thread_param;
  LOG(INFO) << "Thread " << param->thread_index << " start run!";
  PD_OneDimArrayCstr* input_names = PD_PredictorGetInputNames(param->predictor);
  PD_Tensor* tensor =
      PD_PredictorGetInputHandle(param->predictor, input_names->data[0]);
  PD_TensorReshape(tensor, param->shape_size, param->shapes);
  PD_TensorCopyFromCpuFloat(tensor, param->input_data);
  PD_PredictorRun(param->predictor);
  PD_OneDimArrayCstr* output_names =
      PD_PredictorGetOutputNames(param->predictor);
  PD_Tensor* output_tensor =
      PD_PredictorGetOutputHandle(param->predictor, output_names->data[0]);
  PD_OneDimArrayInt32* output_shape = PD_TensorGetShape(output_tensor);
  param->out_size = 1;
  for (size_t index = 0; index < output_shape->size; ++index) {
    param->out_size = param->out_size * output_shape->data[index];
  }
  PD_OneDimArrayInt32Destroy(output_shape);
  param->out_data.resize(param->out_size);
  PD_TensorCopyToCpuFloat(output_tensor, param->out_data.data());
  PD_TensorDestroy(output_tensor);
  PD_OneDimArrayCstrDestroy(output_names);
  PD_TensorDestroy(tensor);
  PD_OneDimArrayCstrDestroy(input_names);
  LOG(INFO) << "Thread " << param->thread_index << " end run!";
  return nullptr;
}
void threads_run(int thread_num) {
  auto model_dir = FLAGS_infer_model;
  PD_Config* config = PD_ConfigCreate();
  PD_ConfigSetModel(config,
                    (model_dir + "/__model__").c_str(),
                    (model_dir + "/__params__").c_str());
  PD_Predictor* predictor = PD_PredictorCreate(config);

  std::vector<pthread_t> threads(thread_num);
  std::vector<RunParameter> params(thread_num);

  std::array<int32_t, 4> shapes = {1, 3, 224, 224};
  std::vector<float> input(1 * 3 * 224 * 224, 0);
  for (int i = 0; i < thread_num; ++i) {
    params[i].predictor = PD_PredictorClone(predictor);
    params[i].shapes = shapes.data();
    params[i].shape_size = 4;
    params[i].input_data = input.data();
    params[i].out_size = 0;
    params[i].thread_index = i;
    pthread_create(&(threads[i]), nullptr, run, &(params[i]));
  }
  for (int i = 0; i < thread_num; ++i) {
    pthread_join(threads[i], nullptr);
  }
  ASSERT_GT(params[0].out_size, 0);

  for (int i = 1; i < thread_num; ++i) {
    ASSERT_EQ(params[i].out_size, params[0].out_size);
    for (int j = 0; j < params[i].out_size; ++j) {
      ASSERT_EQ(params[i].out_data[j], params[0].out_data[j]);
    }
  }
  for (int i = 0; i < thread_num; ++i) {
    PD_PredictorDestroy(params[i].predictor);
  }
  PD_PredictorDestroy(predictor);
}

TEST(PD_Predictor, PD_multi_threads_run) { threads_run(10); }

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
