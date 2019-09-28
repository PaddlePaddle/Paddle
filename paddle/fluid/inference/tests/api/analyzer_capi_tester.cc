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
#include <string>
#include <vector>
#include "paddle/fluid/inference/capi/c_api.h"
#include "paddle/fluid/inference/capi/c_api_internal.h"
#include "paddle/fluid/inference/tests/api/tester_helper.h"

namespace paddle {
namespace inference {
namespace analysis {

struct Record {
  std::vector<float> data;
  std::vector<int32_t> shape;
};

Record ProcessALine(const std::string &line) {
  VLOG(3) << "process a line";
  std::vector<std::string> columns;
  split(line, '\t', &columns);
  CHECK_EQ(columns.size(), 2UL)
      << "data format error, should be <data>\t<shape>";

  Record record;
  std::vector<std::string> data_strs;
  split(columns[0], ' ', &data_strs);
  for (auto &d : data_strs) {
    record.data.push_back(std::stof(d));
  }

  std::vector<std::string> shape_strs;
  split(columns[1], ' ', &shape_strs);
  for (auto &s : shape_strs) {
    record.shape.push_back(std::stoi(s));
  }
  VLOG(3) << "data size " << record.data.size();
  VLOG(3) << "data shape size " << record.shape.size();
  return record;
}

const char *GetModelPath(std::string str) { return str.c_str(); }

void SetConfig(AnalysisConfig *cfg) {
  cfg->SetModel(FLAGS_infer_model + "/model/__model__",
                FLAGS_infer_model + "/model/__params__");
  cfg->DisableGpu();
  cfg->SwitchIrDebug();
  cfg->SwitchSpecifyInputNames(false);
  // TODO(TJ): fix fusion gru
  cfg->pass_builder()->DeletePass("fc_gru_fuse_pass");
}

void SetInput(std::vector<std::vector<PaddleTensor>> *inputs) {
  PADDLE_ENFORCE_EQ(FLAGS_test_all_data, 0, "Only have single batch of data.");
  std::string line;
  std::ifstream file(FLAGS_infer_data);
  std::getline(file, line);
  auto record = ProcessALine(line);

  PaddleTensor input;
  input.shape = record.shape;
  input.dtype = PaddleDType::FLOAT32;
  size_t input_size = record.data.size() * sizeof(float);
  input.data.Resize(input_size);
  memcpy(input.data.data(), record.data.data(), input_size);
  std::vector<PaddleTensor> input_slots;
  input_slots.assign({input});
  (*inputs).emplace_back(input_slots);
}

// Easy for profiling independently.
//  ocr, mobilenet and se_resnext50
void profile(bool use_mkldnn = false) {
  std::string model_dir1 =
      FLAGS_infer_model + "/mobilenet";  // + "/model/__model__";
  // std::string params_file1 = FLAGS_infer_model + "/model/__params__";
  // const char *model_dir = GetModelPath(FLAGS_infer_model +
  // "/model/__model__");
  // LOG(INFO) << model_dir;
  // const char *params_file =
  //     GetModelPath(FLAGS_infer_model + "/model/__params__");
  // LOG(INFO) << model_dir;
  PD_AnalysisConfig config;  // = PD_NewAnalysisConfig();
  // LOG(INFO) << PD_ModelDir(&config);
  PD_DisableGpu(&config);
  PD_SetCpuMathLibraryNumThreads(&config, 10);
  PD_SwitchUseFeedFetchOps(&config, false);
  PD_SwitchSpecifyInputNames(&config, true);
  PD_SwitchIrDebug(&config, true);
  // LOG(INFO) << "before here! ";
  PD_SetModel(&config, model_dir1.c_str());  //, params_file1.c_str());

  const int batch_size = 1;
  const int channels = 3;
  const int height = 224;
  const int width = 224;
  float input[batch_size * channels * height * width] = {0};

  int shape[4] = {batch_size, channels, height, width};

  int shape_size = 4;
  // AnalysisConfig cfg;
  // cfg.DisableGpu();
  // cfg.SwitchUseFeedFetchOps(false);
  // cfg.SetModel(model_dir, params_file);
  // auto predictor = CreatePaddlePredictor(cfg);

  // std::vector<std::vector<PaddleTensor>> inputs_all;
  // auto predictor = CreatePaddlePredictor(config.config);
  // SetFakeImageInput(&inputs_all, model_dir1, false, "__model__",
  // "__params__");

  // std::vector<PaddleTensor> outputs;
  // for (auto &input : inputs_all) {
  //   ASSERT_TRUE(predictor->Run(input, &outputs));
  // }

  // std::vector<std::vector<PaddleTensor>> inputs_all;
  // SetInput(&inputs_all);
  // std::vector<PaddleTensor> outputs;
  // for (auto &input : inputs_all) {
  //   ASSERT_TRUE(predictor->Run(input, &outputs));
  // }

  // std::vector<std::vector<PaddleTensor>> inputs;
  // SetInput(&inputs);
  // for (int i = 0; i < inputs[0][0].shape.size(); i++) {
  //   LOG(INFO) << (inputs[0][0].shape)[i];
  // }

  int in_size = 1;
  int *out_size;
  // float *output;
  PD_ZeroCopyData *inputs = new PD_ZeroCopyData;
  PD_ZeroCopyData *outputs = new PD_ZeroCopyData;
  inputs->data = static_cast<void *>(input);
  inputs->dtype = PD_FLOAT32;
  // inputs->name = new char[2];
  // inputs->name = "x";
  LOG(INFO) << sizeof(std::string("x")) / sizeof(char) + 1;
  snprintf(inputs->name, sizeof(std::string("x")) / sizeof(char) + 1, "%s",
           std::string("x").c_str());
  inputs->shape = shape;
  inputs->shape_size = shape_size;

  PD_PredictorZeroCopyRun(&config, inputs, in_size, &outputs, &out_size);

  // auto input_names = predictor->GetInputNames();
  // auto input_t = predictor->GetInputTensor(input_names[0]);
  // std::vector<int> tensor_shape;
  // tensor_shape.assign(shape, shape + shape_size);
  // input_t->Reshape(tensor_shape);
  // input_t->copy_from_cpu(input);
  // CHECK(predictor->ZeroCopyRun());

  // PD_Predictor *predictor = PD_NewPredictor(&config);
  // auto pre = CreatePaddlePredictor(config.config);
  // PD_Predictor *predictor = NULL;
  // predictor->predictor = pre.Clone();

  // int *size;
  // char **input_names = PD_GetPredictorInputNames(predictor, &size);
  // LOG(INFO) << input_names[0];
  // PD_DataType data_type = PD_FLOAT32;
  // PD_ZeroCopyTensor *tensor =
  //     PD_GetPredictorInputTensor(predictor, input_names[0]);
  // PD_ZeroCopyTensorReshape(tensor, shape, shape_size);
  // PD_ZeroCopyFromCpu(tensor, input, data_type);
  // LOG(INFO) << "before zerocopyrun.";
  // CHECK(PD_PredictorZeroCopyRun(predictor));

  /*PD_Tensor* ten = PD_NewPaddleTensor();
  ten->tensor = inputs_all[0][0];
  PD_Tensor* out = PD_NewPaddleTensor();
  int* outsize;
  int insize = 1;
  PD_PredictorRun(predictor, ten, insize, out, &outsize, 1);*/

  /*std::vector<PaddleTensor> outputs;
  for (auto& input : inputs_all) {
    ASSERT_TRUE(predictor->Run(input, &outputs));
  }*/
}

TEST(Analyzer_vis, profile) { profile(); }

// #ifdef PADDLE_WITH_MKLDNN
// TEST(Analyzer_vis, profile_mkldnn) { profile(true /* use_mkldnn */); }
// #endif

// Check the fuse status
// TEST(Analyzer_vis, fuse_statis) {
//   AnalysisConfig cfg;
//   SetConfig(&cfg);
//   int num_ops;
//   auto predictor = CreatePaddlePredictor<AnalysisConfig>(cfg);
//   GetFuseStatis(predictor.get(), &num_ops);
// }

// Compare result of NativeConfig and AnalysisConfig
void compare(bool use_mkldnn = false) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  if (use_mkldnn) {
    cfg.EnableMKLDNN();
    cfg.pass_builder()->AppendPass("fc_mkldnn_pass");
  }

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareNativeAndAnalysis(
      reinterpret_cast<const PaddlePredictor::Config *>(&cfg), input_slots_all);
}

// TEST(Analyzer_vis, compare) { compare(); }
// #ifdef PADDLE_WITH_MKLDNN
// TEST(Analyzer_vis, compare_mkldnn) { compare(true /* use_mkldnn */); }
// #endif

// // Compare Deterministic result
// TEST(Analyzer_vis, compare_determine) {
//   AnalysisConfig cfg;
//   SetConfig(&cfg);

//   std::vector<std::vector<PaddleTensor>> input_slots_all;
//   SetInput(&input_slots_all);
//   CompareDeterministic(reinterpret_cast<const PaddlePredictor::Config
//   *>(&cfg),
//                        input_slots_all);
// }

}  // namespace analysis
}  // namespace inference
}  // namespace paddle

/*namespace paddle {
namespace inference {

const char* GetModelPath(std::string a) { return a.c_str(); }


TEST(Analysis_capi, compare) {
  std::string a = FLAGS_infer_model;
  const char* model_dir =
      GetModelPath(FLAGS_infer_model + "/mobilenet/__model__");
  const char* params_file =
      GetModelPath(FLAGS_infer_model + "/mobilenet/__params__");
  LOG(INFO) << model_dir;
  PD_AnalysisConfig* config = PD_NewAnalysisConfig();
  PD_SetModel(config, model_dir, params_file);
  LOG(INFO) << PD_ModelDir(config);
  PD_DisableGpu(config);
  PD_SetCpuMathLibraryNumThreads(config, 10);
  PD_SwitchUseFeedFetchOps(config, false);
  PD_SwitchSpecifyInputNames(config, true);
  PD_SwitchIrDebug(config, true);
  LOG(INFO) << "before here! ";

  const int batch_size = 1;
  const int channels = 3;
  const int height = 224;
  const int width = 224;
  float input[batch_size * channels * height * width] = {0};

  int shape[4] = {batch_size, channels, height, width};

  AnalysisConfig c;
  c.SetModel(model_dir, params_file);
  LOG(INFO) << c.model_dir();
  c.DisableGpu();
  c.SwitchUseFeedFetchOps(false);
  int shape_size = 4;
  auto predictor = CreatePaddlePredictor(c);
  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputTensor(input_names[0]);
  std::vector<int> tensor_shape;
  tensor_shape.assign(shape, shape + shape_size);
  input_t->Reshape(tensor_shape);
  input_t->copy_from_cpu(input);
  CHECK(predictor->ZeroCopyRun());
}

}  // namespace inference
}  // namespace paddle*/
