// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <string>
#include <vector>

#include "paddle/common/flags.h"
#include "paddle/fluid/inference/capi_exp/pd_inference_api.h"

PD_DEFINE_string(infer_model, "", "model path");

namespace paddle {
namespace inference {
namespace analysis {

TEST(PD_PredictorRun, predictor_run) {
  auto model_dir = FLAGS_infer_model;
  PD_Config *config = PD_ConfigCreate();
  PD_ConfigSetModel(config,
                    (model_dir + "/__model__").c_str(),
                    (model_dir + "/param").c_str());
  PD_ConfigDisableGpu(config);

  PD_Predictor *predictor = PD_PredictorCreate(config);
  size_t input_num = PD_PredictorGetInputNum(predictor);
  LOG(INFO) << "Input num: " << input_num;
  size_t output_num = PD_PredictorGetOutputNum(predictor);
  LOG(INFO) << "Output num: " << output_num;

  PD_OneDimArrayCstr *input_names = PD_PredictorGetInputNames(predictor);
  EXPECT_EQ(input_names->size, 2u);
  LOG(INFO) << "Predictor start run!";
  PD_Tensor *inputs[2];  // NOLINT
  inputs[0] = PD_PredictorGetInputHandle(predictor, input_names->data[0]);
  inputs[1] = PD_PredictorGetInputHandle(predictor, input_names->data[1]);
  LOG(INFO) << "Predictor start run!";
  // inputs[0]: word, use lod memory in stack
  std::array<int32_t, 2> shape_0 = {11, 1};
  std::array<int64_t, 11 * 1> data_0 = {
      12673, 9763, 905, 284, 45, 7474, 20, 17, 1, 4, 9};
  std::array<size_t, 2> lod_layer_0 = {0, 11};
  PD_OneDimArraySize layer_0;
  layer_0.size = 2;
  layer_0.data = lod_layer_0.data();
  PD_OneDimArraySize *layer_0_ptr = &layer_0;
  PD_TwoDimArraySize lod_0;
  lod_0.size = 1;
  lod_0.data = &layer_0_ptr;
  PD_TensorReshape(inputs[0], 2, shape_0.data());
  PD_TensorCopyFromCpuInt64(inputs[0], data_0.data());
  PD_TensorSetLod(inputs[0], &lod_0);

  // inputs[1]: mention, use lod memory in heap
  std::array<int32_t, 2> shape_1 = {11, 1};
  std::array<int64_t, 11 * 1> data_1 = {27, 0, 0, 33, 34, 33, 0, 0, 0, 1, 2};
  PD_TwoDimArraySize *lod_1_ptr = new PD_TwoDimArraySize();
  lod_1_ptr->size = 1;
  lod_1_ptr->data = new PD_OneDimArraySize *[1];
  lod_1_ptr->data[0] = new PD_OneDimArraySize();
  lod_1_ptr->data[0]->size = 2;
  lod_1_ptr->data[0]->data = new size_t[2];
  lod_1_ptr->data[0]->data[0] = 0;
  lod_1_ptr->data[0]->data[1] = 11;

  PD_TensorReshape(inputs[1], 2, shape_1.data());
  PD_TensorCopyFromCpuInt64(inputs[1], data_1.data());
  PD_TensorSetLod(inputs[1], lod_1_ptr);
  // retrieve the lod memory
  delete[] lod_1_ptr->data[0]->data;
  delete lod_1_ptr->data[0];
  delete[] lod_1_ptr->data;
  delete lod_1_ptr;
  lod_1_ptr = nullptr;

  LOG(INFO) << "Predictor start run!";
  bool success = PD_PredictorRun(predictor);
  EXPECT_TRUE(success);
  LOG(INFO) << "Predictor run success!";
  PD_OneDimArrayCstr *output_names = PD_PredictorGetOutputNames(predictor);
  PD_Tensor *output =
      PD_PredictorGetOutputHandle(predictor, output_names->data[0]);
  PD_TwoDimArraySize *output_lod = PD_TensorGetLod(output);

  PD_TwoDimArraySizeDestroy(output_lod);
  PD_TensorDestroy(output);
  PD_OneDimArrayCstrDestroy(output_names);

  PD_TensorDestroy(inputs[0]);
  PD_TensorDestroy(inputs[1]);
  PD_OneDimArrayCstrDestroy(input_names);
  PD_PredictorDestroy(predictor);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
