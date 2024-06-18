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

paddle::test::Record PrepareInput(int batch_size, int image_shape = 640) {
  // init input data
  int channel = 3;
  int width = image_shape;
  int height = image_shape;
  paddle::test::Record image_Record;
  int input_num = batch_size * channel * width * height;
  std::vector<float> input_data(input_num, 1);
  image_Record.data = input_data;
  image_Record.shape = std::vector<int>{batch_size, channel, width, height};
  image_Record.type = paddle::PaddleDType::FLOAT32;
  return image_Record;
}

void PrepareDynamicShape(paddle_infer::Config* config, int max_batch_size = 4) {
  // set dynamic shape range
  std::map<std::string, std::vector<int>> min_input_shape = {
      {"x", {1, 3, 224, 224}},
      {"conv2d_124.tmp_0", {1, 256, 56, 56}},
      {"nearest_interp_v2_2.tmp_0", {1, 256, 56, 56}},
      {"nearest_interp_v2_3.tmp_0", {1, 64, 56, 56}},
      {"nearest_interp_v2_4.tmp_0", {1, 64, 56, 56}},
      {"nearest_interp_v2_5.tmp_0", {1, 64, 56, 56}}};
  std::map<std::string, std::vector<int>> max_input_shape = {
      {"x", {max_batch_size, 3, 448, 448}},
      {"conv2d_124.tmp_0", {max_batch_size, 256, 112, 112}},
      {"nearest_interp_v2_2.tmp_0", {max_batch_size, 256, 112, 112}},
      {"nearest_interp_v2_3.tmp_0", {max_batch_size, 64, 112, 112}},
      {"nearest_interp_v2_4.tmp_0", {max_batch_size, 64, 112, 112}},
      {"nearest_interp_v2_5.tmp_0", {max_batch_size, 64, 112, 112}}};
  std::map<std::string, std::vector<int>> opt_input_shape = {
      {"x", {1, 3, 256, 256}},
      {"conv2d_124.tmp_0", {1, 256, 64, 64}},
      {"nearest_interp_v2_2.tmp_0", {1, 256, 64, 64}},
      {"nearest_interp_v2_3.tmp_0", {1, 64, 64, 64}},
      {"nearest_interp_v2_4.tmp_0", {1, 64, 64, 64}},
      {"nearest_interp_v2_5.tmp_0", {1, 64, 64, 64}}};
  config->SetTRTDynamicShapeInfo(
      min_input_shape, max_input_shape, opt_input_shape);
}

TEST(gpu_tester_det_mv3_db, analysis_gpu_bz4) {
  // init input data
  std::map<std::string, paddle::test::Record> my_input_data_map;
  my_input_data_map["x"] = PrepareInput(4, 640);
  // init output data
  std::map<std::string, paddle::test::Record> infer_output_data,
      truth_output_data;
  // prepare ground truth config
  paddle_infer::Config config, config_no_ir;
  config_no_ir.SetModel(FLAGS_modeldir + "/inference.pdmodel",
                        FLAGS_modeldir + "/inference.pdiparams");
  config_no_ir.SwitchIrOptim(false);
  // prepare inference config
  config.SetModel(FLAGS_modeldir + "/inference.pdmodel",
                  FLAGS_modeldir + "/inference.pdiparams");
  // get ground truth by disable ir
  paddle_infer::services::PredictorPool pred_pool_no_ir(config_no_ir, 1);
  SingleThreadPrediction(
      pred_pool_no_ir.Retrieve(0), &my_input_data_map, &truth_output_data, 1);
  // get infer results
  paddle_infer::services::PredictorPool pred_pool(config, 1);
  SingleThreadPrediction(
      pred_pool.Retrieve(0), &my_input_data_map, &infer_output_data);
  // check outputs
  CompareRecord(&truth_output_data, &infer_output_data, 1e-4);
  std::cout << "finish test" << std::endl;
}

TEST(tensorrt_tester_det_mv3_db, multi_thread2_trt_fp32_dynamic_shape_bz2) {
  int thread_num = 2;  // thread > 2 may OOM
  // init input data
  std::map<std::string, paddle::test::Record> my_input_data_map;
  my_input_data_map["x"] = PrepareInput(2, 256);
  // init output data
  std::map<std::string, paddle::test::Record> infer_output_data,
      truth_output_data;
  // prepare ground truth config
  paddle_infer::Config config, config_no_ir;
  config_no_ir.SetModel(FLAGS_modeldir + "/inference.pdmodel",
                        FLAGS_modeldir + "/inference.pdiparams");
  config_no_ir.SwitchIrOptim(false);
  // prepare inference config
  config.SetModel(FLAGS_modeldir + "/inference.pdmodel",
                  FLAGS_modeldir + "/inference.pdiparams");
  config.EnableUseGpu(100, 0);
  config.EnableTensorRtEngine(
      1 << 20, 4, 3, paddle_infer::PrecisionType::kFloat32, false, false);
  PrepareDynamicShape(&config, 4);
  // get ground truth by disable ir
  paddle_infer::services::PredictorPool pred_pool_no_ir(config_no_ir, 1);
  SingleThreadPrediction(
      pred_pool_no_ir.Retrieve(0), &my_input_data_map, &truth_output_data, 1);

  // get infer results from multi threads
  std::vector<std::thread> threads;
  services::PredictorPool pred_pool(config, thread_num);
  for (int i = 0; i < thread_num; ++i) {
    threads.emplace_back(paddle::test::SingleThreadPrediction,
                         pred_pool.Retrieve(i),
                         &my_input_data_map,
                         &infer_output_data,
                         2);
  }

  // thread join & check outputs
  for (int i = 0; i < thread_num; ++i) {
    LOG(INFO) << "join tid : " << i;
    threads[i].join();
    CompareRecord(&truth_output_data, &infer_output_data, 1e-4);
  }

  std::cout << "finish multi-thread test" << std::endl;
}

TEST(mkldnn_tester_det_mv3_db, multi_thread2_mkl_fp32_bz2) {
  int thread_num = 2;  // thread > 2 may OOM
  // init input data
  std::map<std::string, paddle::test::Record> my_input_data_map;
  my_input_data_map["x"] = PrepareInput(2, 640);
  // init output data
  std::map<std::string, paddle::test::Record> infer_output_data,
      truth_output_data;
  // prepare ground truth config
  paddle_infer::Config config, config_no_ir;
  config_no_ir.SetModel(FLAGS_modeldir + "/inference.pdmodel",
                        FLAGS_modeldir + "/inference.pdiparams");
  config_no_ir.SwitchIrOptim(false);
  // prepare inference config
  config.SetModel(FLAGS_modeldir + "/inference.pdmodel",
                  FLAGS_modeldir + "/inference.pdiparams");
  config.DisableGpu();
  config.EnableMKLDNN();
  config.SetMkldnnCacheCapacity(10);
  config.SetCpuMathLibraryNumThreads(10);
  // get ground truth by disable ir
  paddle_infer::services::PredictorPool pred_pool_no_ir(config_no_ir, 1);
  SingleThreadPrediction(
      pred_pool_no_ir.Retrieve(0), &my_input_data_map, &truth_output_data, 1);

  // get infer results from multi threads
  std::vector<std::thread> threads;
  services::PredictorPool pred_pool(config, thread_num);
  for (int i = 0; i < thread_num; ++i) {
    threads.emplace_back(paddle::test::SingleThreadPrediction,
                         pred_pool.Retrieve(i),
                         &my_input_data_map,
                         &infer_output_data,
                         2);
  }

  // thread join & check outputs
  for (int i = 0; i < thread_num; ++i) {
    LOG(INFO) << "join tid : " << i;
    threads[i].join();
    CompareRecord(&truth_output_data, &infer_output_data, 1e-4);
  }

  std::cout << "finish multi-thread test" << std::endl;
}

}  // namespace paddle_infer

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}
