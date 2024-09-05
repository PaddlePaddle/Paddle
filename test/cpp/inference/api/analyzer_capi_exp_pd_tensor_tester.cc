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
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "paddle/common/flags.h"
#include "paddle/fluid/inference/capi_exp/pd_inference_api.h"

PD_DEFINE_string(infer_model, "", "model path");

namespace paddle {
namespace inference {
namespace analysis {

void PD_run() {
  auto model_dir = FLAGS_infer_model;
  PD_Config* config = PD_ConfigCreate();
  PD_ConfigSetModel(config,
                    (model_dir + "/__model__").c_str(),
                    (model_dir + "/__params__").c_str());
  PD_Predictor* predictor = PD_PredictorCreate(config);
  PD_OneDimArrayCstr* input_names = PD_PredictorGetInputNames(predictor);
  PD_Tensor* tensor =
      PD_PredictorGetInputHandle(predictor, input_names->data[0]);

  std::array<int32_t, 4> shapes = {1, 3, 224, 224};
  std::vector<float> input(1 * 3 * 224 * 224, 0);
  int32_t size;
  PD_PlaceType place;
  PD_TensorReshape(tensor, 4, shapes.data());
  PD_TensorCopyFromCpuFloat(tensor, input.data());
  PD_TensorDataFloat(tensor, &place, &size);
  PD_TensorMutableDataFloat(tensor, place);

  PD_TwoDimArraySize lod;
  lod.size = 0;
  lod.data = nullptr;
  PD_TensorSetLod(tensor, &lod);

  PD_PredictorRun(predictor);

  std::vector<float> out_data;
  PD_OneDimArrayCstr* output_names = PD_PredictorGetOutputNames(predictor);
  PD_Tensor* output_tensor =
      PD_PredictorGetOutputHandle(predictor, output_names->data[0]);
  PD_OneDimArrayInt32* output_shape = PD_TensorGetShape(output_tensor);
  int32_t out_num = std::accumulate(output_shape->data,
                                    output_shape->data + output_shape->size,
                                    1,
                                    std::multiplies<>());
  out_data.resize(out_num);
  PD_TensorCopyToCpuFloat(output_tensor, out_data.data());
  LOG(INFO) << "Output tensor name is: " << PD_TensorGetName(output_tensor);
  PD_DataType data_type = PD_TensorGetDataType(output_tensor);
  EXPECT_EQ(data_type, PD_DATA_FLOAT32);

  PD_TwoDimArraySize* out_lod = PD_TensorGetLod(output_tensor);

  PD_TwoDimArraySizeDestroy(out_lod);
  PD_OneDimArrayInt32Destroy(output_shape);
  PD_TensorDestroy(output_tensor);
  PD_OneDimArrayCstrDestroy(output_names);
  PD_TensorDestroy(tensor);
  PD_OneDimArrayCstrDestroy(input_names);
  PD_PredictorDestroy(predictor);
}
TEST(PD_Tensor, PD_run) { PD_run(); }

TEST(PD_Tensor, int32) {
  auto model_dir = FLAGS_infer_model;
  PD_Config* config = PD_ConfigCreate();
  PD_ConfigSetModel(config,
                    (model_dir + "/__model__").c_str(),
                    (model_dir + "/__params__").c_str());
  PD_Predictor* predictor = PD_PredictorCreate(config);
  PD_OneDimArrayCstr* input_names = PD_PredictorGetInputNames(predictor);
  PD_Tensor* tensor =
      PD_PredictorGetInputHandle(predictor, input_names->data[0]);
  std::array<int32_t, 4> shapes = {1, 3, 224, 224};
  std::vector<int32_t> input(1 * 3 * 224 * 224, 0);
  int32_t size;
  PD_PlaceType place;
  PD_TensorReshape(tensor, 4, shapes.data());
  PD_TensorCopyFromCpuInt32(tensor, input.data());
  int32_t* data_ptr = PD_TensorDataInt32(tensor, &place, &size);
  EXPECT_EQ(place, PD_PLACE_CPU);
  EXPECT_EQ(size, 1 * 3 * 224 * 224);
  int32_t* mutable_data_ptr = PD_TensorMutableDataInt32(tensor, place);
  EXPECT_EQ(data_ptr, mutable_data_ptr);

  PD_DataType data_type = PD_TensorGetDataType(tensor);
  EXPECT_EQ(data_type, PD_DATA_INT32);
  PD_TensorCopyToCpuInt32(tensor, input.data());

  PD_TensorDestroy(tensor);
  PD_OneDimArrayCstrDestroy(input_names);
  PD_PredictorDestroy(predictor);
}

TEST(PD_Tensor, int64) {
  auto model_dir = FLAGS_infer_model;
  PD_Config* config = PD_ConfigCreate();
  PD_ConfigSetModel(config,
                    (model_dir + "/__model__").c_str(),
                    (model_dir + "/__params__").c_str());
  PD_Predictor* predictor = PD_PredictorCreate(config);
  PD_OneDimArrayCstr* input_names = PD_PredictorGetInputNames(predictor);
  PD_Tensor* tensor =
      PD_PredictorGetInputHandle(predictor, input_names->data[0]);
  std::array<int32_t, 4> shapes = {1, 3, 224, 224};
  std::vector<int64_t> input(1 * 3 * 224 * 224, 0);
  int32_t size;
  PD_PlaceType place;
  PD_TensorReshape(tensor, 4, shapes.data());
  PD_TensorCopyFromCpuInt64(tensor, input.data());
  int64_t* data_ptr = PD_TensorDataInt64(tensor, &place, &size);
  EXPECT_EQ(place, PD_PLACE_CPU);
  EXPECT_EQ(size, 1 * 3 * 224 * 224);
  int64_t* mutable_data_ptr = PD_TensorMutableDataInt64(tensor, place);
  EXPECT_EQ(data_ptr, mutable_data_ptr);

  PD_DataType data_type = PD_TensorGetDataType(tensor);
  EXPECT_EQ(data_type, PD_DATA_INT64);
  PD_TensorCopyToCpuInt64(tensor, input.data());

  PD_TensorDestroy(tensor);
  PD_OneDimArrayCstrDestroy(input_names);
  PD_PredictorDestroy(predictor);
}

TEST(PD_Tensor, uint8) {
  auto model_dir = FLAGS_infer_model;
  PD_Config* config = PD_ConfigCreate();
  PD_ConfigSetModel(config,
                    (model_dir + "/__model__").c_str(),
                    (model_dir + "/__params__").c_str());
  PD_Predictor* predictor = PD_PredictorCreate(config);
  PD_OneDimArrayCstr* input_names = PD_PredictorGetInputNames(predictor);
  PD_Tensor* tensor =
      PD_PredictorGetInputHandle(predictor, input_names->data[0]);
  std::array<int32_t, 4> shapes = {1, 3, 224, 224};
  std::array<uint8_t, 1 * 3 * 224 * 224> input = {0};
  int32_t size;
  PD_PlaceType place;
  PD_TensorReshape(tensor, 4, shapes.data());
  PD_TensorCopyFromCpuUint8(tensor, input.data());
  uint8_t* data_ptr = PD_TensorDataUint8(tensor, &place, &size);
  EXPECT_EQ(place, PD_PLACE_CPU);
  EXPECT_EQ(size, 1 * 3 * 224 * 224);
  uint8_t* mutable_data_ptr = PD_TensorMutableDataUint8(tensor, place);
  EXPECT_EQ(data_ptr, mutable_data_ptr);

  PD_DataType data_type = PD_TensorGetDataType(tensor);
  EXPECT_EQ(data_type, PD_DATA_UINT8);
  PD_TensorCopyToCpuUint8(tensor, input.data());

  PD_TensorDestroy(tensor);
  PD_OneDimArrayCstrDestroy(input_names);
  PD_PredictorDestroy(predictor);
}

std::string read_file(std::string filename) {
  std::ifstream file(filename);
  return std::string((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
}

TEST(PD_Tensor, from_buffer) {
  PD_Config* config = PD_ConfigCreate();
  std::string prog_file = FLAGS_infer_model + "/__model__";
  std::string params_file = FLAGS_infer_model + "/__params__";

  std::string prog_str = read_file(prog_file);
  std::string params_str = read_file(params_file);

  PD_ConfigSetModelBuffer(config,
                          prog_str.c_str(),
                          prog_str.size(),
                          params_str.c_str(),
                          params_str.size());

  bool model_from_memory = PD_ConfigModelFromMemory(config);
  EXPECT_TRUE(model_from_memory);
  PD_ConfigDestroy(config);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
