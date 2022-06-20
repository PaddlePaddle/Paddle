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

#include "test_suite.h"  // NOLINT

DEFINE_string(modeldir, "", "Directory of the inference model.");

namespace paddle_infer {

paddle::test::Record PrepareInput(int batch_size) {
  // init input data
  int channel = 3;
  int width = 224;
  int height = 224;
  paddle::test::Record image_Record;
  int input_num = batch_size * channel * width * height;
  std::vector<float> input_data(input_num, 1);
  image_Record.data = input_data;
  image_Record.shape = std::vector<int>{batch_size, channel, width, height};
  image_Record.type = paddle::PaddleDType::FLOAT32;
  return image_Record;
}

TEST(gpu_tester_resnet50, analysis_gpu_bz1) {
  // init input data
  std::map<std::string, paddle::test::Record> my_input_data_map;
  my_input_data_map["inputs"] = PrepareInput(1);
  // init output data
  std::map<std::string, paddle::test::Record> infer_output_data,
      truth_output_data;
  // prepare groudtruth config
  paddle_infer::Config config, config_no_ir;
  config_no_ir.SetModel(FLAGS_modeldir + "/inference.pdmodel",
                        FLAGS_modeldir + "/inference.pdiparams");
  config_no_ir.SwitchIrOptim(false);
  // prepare inference config
  config.SetModel(FLAGS_modeldir + "/inference.pdmodel",
                  FLAGS_modeldir + "/inference.pdiparams");
  // get groudtruth by disbale ir
  paddle_infer::services::PredictorPool pred_pool_no_ir(config_no_ir, 1);
  SingleThreadPrediction(pred_pool_no_ir.Retrive(0), &my_input_data_map,
                         &truth_output_data, 1);
  // get infer results
  paddle_infer::services::PredictorPool pred_pool(config, 1);
  SingleThreadPrediction(pred_pool.Retrive(0), &my_input_data_map,
                         &infer_output_data);
  // check outputs
  CompareRecord(&truth_output_data, &infer_output_data);
  std::cout << "finish test" << std::endl;
}

TEST(tensorrt_tester_resnet50, trt_fp32_bz2) {
  // init input data
  std::map<std::string, paddle::test::Record> my_input_data_map;
  my_input_data_map["inputs"] = PrepareInput(2);
  // init output data
  std::map<std::string, paddle::test::Record> infer_output_data,
      truth_output_data;
  // prepare groudtruth config
  paddle_infer::Config config, config_no_ir;
  config_no_ir.SetModel(FLAGS_modeldir + "/inference.pdmodel",
                        FLAGS_modeldir + "/inference.pdiparams");
  config_no_ir.SwitchIrOptim(false);
  // prepare inference config
  config.SetModel(FLAGS_modeldir + "/inference.pdmodel",
                  FLAGS_modeldir + "/inference.pdiparams");
  config.EnableUseGpu(100, 0);
  config.EnableTensorRtEngine(
      1 << 20, 2, 3, paddle_infer::PrecisionType::kFloat32, false, false);
  // get groudtruth by disbale ir
  paddle_infer::services::PredictorPool pred_pool_no_ir(config_no_ir, 1);
  SingleThreadPrediction(pred_pool_no_ir.Retrive(0), &my_input_data_map,
                         &truth_output_data, 1);
  // get infer results
  paddle_infer::services::PredictorPool pred_pool(config, 1);
  SingleThreadPrediction(pred_pool.Retrive(0), &my_input_data_map,
                         &infer_output_data);
  // check outputs
  CompareRecord(&truth_output_data, &infer_output_data, 2e-4);
  std::cout << "finish test" << std::endl;
}

TEST(tensorrt_tester_resnet50, serial_diff_batch_trt_fp32) {
  int max_batch_size = 5;
  // prepare groudtruth config
  paddle_infer::Config config, config_no_ir;
  config_no_ir.SetModel(FLAGS_modeldir + "/inference.pdmodel",
                        FLAGS_modeldir + "/inference.pdiparams");
  config_no_ir.SwitchIrOptim(false);
  paddle_infer::services::PredictorPool pred_pool_no_ir(config_no_ir, 1);
  // prepare inference config
  config.SetModel(FLAGS_modeldir + "/inference.pdmodel",
                  FLAGS_modeldir + "/inference.pdiparams");
  config.EnableUseGpu(100, 0);
  config.EnableTensorRtEngine(1 << 20, max_batch_size, 3,
                              paddle_infer::PrecisionType::kFloat32, false,
                              false);
  paddle_infer::services::PredictorPool pred_pool(config, 1);

  for (int i = 1; i < max_batch_size; i++) {
    // init input data
    std::map<std::string, paddle::test::Record> my_input_data_map;
    my_input_data_map["inputs"] = PrepareInput(i);
    // init output data
    std::map<std::string, paddle::test::Record> infer_output_data,
        truth_output_data;
    // get groudtruth by disbale ir
    SingleThreadPrediction(pred_pool_no_ir.Retrive(0), &my_input_data_map,
                           &truth_output_data, 1);
    // get infer results
    SingleThreadPrediction(pred_pool.Retrive(0), &my_input_data_map,
                           &infer_output_data);
    // check outputs
    CompareRecord(&truth_output_data, &infer_output_data, 1e-4);
  }
  std::cout << "finish test" << std::endl;
}

TEST(tensorrt_tester_resnet50, multi_thread4_trt_fp32_bz2) {
  int thread_num = 4;
  // init input data
  std::map<std::string, paddle::test::Record> my_input_data_map;
  my_input_data_map["inputs"] = PrepareInput(2);
  // init output data
  std::map<std::string, paddle::test::Record> infer_output_data,
      truth_output_data;
  // prepare groudtruth config
  paddle_infer::Config config, config_no_ir;
  config_no_ir.SetModel(FLAGS_modeldir + "/inference.pdmodel",
                        FLAGS_modeldir + "/inference.pdiparams");
  config_no_ir.SwitchIrOptim(false);
  // prepare inference config
  config.SetModel(FLAGS_modeldir + "/inference.pdmodel",
                  FLAGS_modeldir + "/inference.pdiparams");
  config.EnableUseGpu(100, 0);
  config.EnableTensorRtEngine(
      1 << 20, 2, 3, paddle_infer::PrecisionType::kFloat32, false, false);
  // get groudtruth by disbale ir
  paddle_infer::services::PredictorPool pred_pool_no_ir(config_no_ir, 1);
  SingleThreadPrediction(pred_pool_no_ir.Retrive(0), &my_input_data_map,
                         &truth_output_data, 1);

  // get infer results from multi threads
  std::vector<std::thread> threads;
  services::PredictorPool pred_pool(config, thread_num);
  for (int i = 0; i < thread_num; ++i) {
    threads.emplace_back(paddle::test::SingleThreadPrediction,
                         pred_pool.Retrive(i), &my_input_data_map,
                         &infer_output_data, 2);
  }

  // thread join & check outputs
  for (int i = 0; i < thread_num; ++i) {
    LOG(INFO) << "join tid : " << i;
    threads[i].join();
    CompareRecord(&truth_output_data, &infer_output_data, 2e-4);
  }

  std::cout << "finish multi-thread test" << std::endl;
}

TEST(tensorrt_tester_resnet50, trt_int8_bz2) {
  // init input data
  std::map<std::string, paddle::test::Record> my_input_data_map;
  my_input_data_map["inputs"] = PrepareInput(2);
  // init output data
  std::map<std::string, paddle::test::Record> infer_output_data,
      truth_output_data;
  // prepare inference config
  paddle_infer::Config config;
  config.SetModel(FLAGS_modeldir + "/inference.pdmodel",
                  FLAGS_modeldir + "/inference.pdiparams");
  config.EnableUseGpu(100, 0);
  config.EnableTensorRtEngine(1 << 20, 2, 3, paddle_infer::PrecisionType::kInt8,
                              true, true);

  // get first time prediction int8 results
  paddle_infer::services::PredictorPool pred_pool(config, 1);
  SingleThreadPrediction(pred_pool.Retrive(0), &my_input_data_map,
                         &truth_output_data, 1);

  // get repeat 5 times prediction int8 results
  SingleThreadPrediction(pred_pool.Retrive(0), &my_input_data_map,
                         &infer_output_data, 5);

  // check outputs
  CompareRecord(&truth_output_data, &infer_output_data);
  std::cout << "finish test" << std::endl;
}

TEST(DISABLED_tensorrt_tester_resnet50, profile_multi_thread_trt_fp32) {
  int batch_size = 2;
  int thread_num = 4;
  int repeat_time = 1000;
  // init input data
  std::map<std::string, paddle::test::Record> my_input_data_map;
  my_input_data_map["inputs"] = PrepareInput(batch_size);
  // init output data
  std::map<std::string, paddle::test::Record> infer_output_data;
  // prepare inference config
  paddle_infer::Config config;
  config.SetModel(FLAGS_modeldir + "/inference.pdmodel",
                  FLAGS_modeldir + "/inference.pdiparams");
  config.EnableUseGpu(100, 0);
  config.EnableTensorRtEngine(
      1 << 20, 2, 3, paddle_infer::PrecisionType::kFloat32, false, false);
  // get infer results from multi threads
  services::PredictorPool pred_pool(config, thread_num);
  std::vector<std::future<double>> calcs;
  for (int i = 0; i < thread_num; ++i) {
    calcs.push_back(std::async(&paddle::test::SingleThreadProfile,
                               pred_pool.Retrive(i), &my_input_data_map,
                               repeat_time));
  }
  double total_time_ = 0.0;
  for (auto&& fut : calcs) {
    total_time_ += fut.get();
  }
  std::cout << total_time_ << std::endl;

  std::cout << "finish multi-thread profile" << std::endl;
}

}  // namespace paddle_infer

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}
