#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <numeric>
#include <iostream>
#include <memory>
#include <chrono>
#include "paddle/fluid/inference/tests/api/tester_helper.h"
namespace paddle {
using paddle::contrib::AnalysisConfig;
DEFINE_string(dirname, "/home/chunwei/project2/models/paddle-transmodel", "Directory of the inference model.");
using Time = decltype(std::chrono::high_resolution_clock::now());
Time time() { return std::chrono::high_resolution_clock::now(); };
double time_diff(Time t1, Time t2) {
  typedef std::chrono::microseconds ms;
  auto diff = t2 - t1;
  ms counter = std::chrono::duration_cast<ms>(diff);
  return counter.count() / 1000.0;
}
void PrepareTRTConfig(AnalysisConfig *config, int batch_size) {
  config->prog_file = FLAGS_dirname + "/model";
  config->param_file = FLAGS_dirname + "/params";
  config->device = 0;
  config->fraction_of_gpu_memory = 0.05;
  //config->EnableTensorRtEngine(1 << 20, batch_size);
  config->pass_builder()->DeletePass("conv_bn_fuse_pass");
  config->pass_builder()->DeletePass("fc_fuse_pass");
  config->pass_builder()->TurnOnDebug();
}
/*
bool run_first(int batch_size, int repeat) {
  AnalysisConfig config(false);
  PrepareTRTConfig(&config, batch_size);
  config.EnableMemoryOptim(true);
  config.Build();
  int channels = 3;
  int height = 512;
  int width = 512;
  int input_num = batch_size * channels * height * width;
  // prepare inputs
  std::vector<PaddleTensor> inputs;
  float *input = new float[input_num];
  memset(input, 0, input_num * sizeof(float));
  PaddleTensor img;
  img.shape = {batch_size, channels, height, width};
  img.data = PaddleBuf(static_cast<void*>(input), input_num * sizeof(float));
  img.dtype = PaddleDType::FLOAT32;
  inputs.emplace_back(img);
  // create predictor
  auto predictor = CreatePaddlePredictor(config);
  std::vector<PaddleTensor> outputs;
  // warm up
  CHECK(predictor->Run(inputs, &outputs, batch_size));
  return true;
}
 */
bool test_map_cnn(int batch_size, int repeat) {
  /*
   std::cout << "111111111" << std::endl;
   run_first(batch_size, repeat);
   std::cout << "22222222" << std::endl;
 */
  AnalysisConfig config(true);
  config.enable_ir_optim = true;
  PrepareTRTConfig(&config, batch_size);
  config.EnableMemoryOptim();
  config.Build();
  int channels = 3;
  int height = 512;
  int width = 512;
  int input_num = batch_size * channels * height * width;
  // prepare inputs
  std::vector<PaddleTensor> inputs;
  float *input = new float[input_num];
  memset(input, 0, input_num * sizeof(float));
  PaddleTensor img;
  img.shape = {batch_size, channels, height, width};
  img.data = PaddleBuf(static_cast<void*>(input), input_num * sizeof(float));
  img.dtype = PaddleDType::FLOAT32;
  inputs.emplace_back(img);
  // create predictor
  auto predictor = CreatePaddlePredictor(config);
  std::vector<PaddleTensor> outputs;
  // warm up
  CHECK(predictor->Run(inputs, &outputs, batch_size));
  auto time1 = time();
  for (int i = 0; i < repeat; i++) {
    CHECK(predictor->Run(inputs, &outputs, batch_size));
  }
  auto time2 = time();
  std::cout <<"batch: " << batch_size << " predict cost: " << time_diff(time1, time2) / 10.0 << "ms" << std::endl;
  /*
  float* data_o = static_cast<float*>(outputs[0].data.data());
  for (size_t j = 0; j < outputs[0].data.length() / sizeof(float); ++j) {
   // data_o[j];
   // LOG(INFO) << "output[" << j << "]: " << data_o[j];
  }
  */
  return true;
}

TEST(memory, test) {
  for (int i = 0; i < 1; i++) {
    paddle::test_map_cnn(1 << i, 0);
  }
}
}  // namespace paddle
