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
      {"x", {1, 3, 50, 50}},
      {"conv2d_92.tmp_0", {1, 120, 20, 20}},
      {"conv2d_91.tmp_0", {1, 24, 10, 10}},
      {"conv2d_59.tmp_0", {1, 96, 20, 20}},
      {"nearest_interp_v2_1.tmp_0", {1, 256, 10, 10}},
      {"nearest_interp_v2_2.tmp_0", {1, 256, 20, 20}},
      {"conv2d_124.tmp_0", {1, 256, 20, 20}},
      {"nearest_interp_v2_3.tmp_0", {1, 64, 20, 20}},
      {"nearest_interp_v2_4.tmp_0", {1, 64, 20, 20}},
      {"nearest_interp_v2_5.tmp_0", {1, 64, 20, 20}},
      {"elementwise_add_7", {1, 56, 2, 2}},
      {"nearest_interp_v2_0.tmp_0", {1, 256, 2, 2}}};
  std::map<std::string, std::vector<int>> max_input_shape = {
      {"x", {max_batch_size, 3, 2000, 2000}},
      {"conv2d_92.tmp_0", {max_batch_size, 120, 400, 400}},
      {"conv2d_91.tmp_0", {max_batch_size, 24, 200, 200}},
      {"conv2d_59.tmp_0", {max_batch_size, 96, 400, 400}},
      {"nearest_interp_v2_1.tmp_0", {max_batch_size, 256, 200, 200}},
      {"nearest_interp_v2_2.tmp_0", {max_batch_size, 256, 400, 400}},
      {"conv2d_124.tmp_0", {max_batch_size, 256, 400, 400}},
      {"nearest_interp_v2_3.tmp_0", {max_batch_size, 64, 400, 400}},
      {"nearest_interp_v2_4.tmp_0", {max_batch_size, 64, 400, 400}},
      {"nearest_interp_v2_5.tmp_0", {max_batch_size, 64, 400, 400}},
      {"elementwise_add_7", {max_batch_size, 56, 400, 400}},
      {"nearest_interp_v2_0.tmp_0", {max_batch_size, 256, 400, 400}}};
  std::map<std::string, std::vector<int>> opt_input_shape = {
      {"x", {1, 3, 640, 640}},
      {"conv2d_92.tmp_0", {1, 120, 160, 160}},
      {"conv2d_91.tmp_0", {1, 24, 80, 80}},
      {"conv2d_59.tmp_0", {1, 96, 160, 160}},
      {"nearest_interp_v2_1.tmp_0", {1, 256, 80, 80}},
      {"nearest_interp_v2_2.tmp_0", {1, 256, 160, 160}},
      {"conv2d_124.tmp_0", {1, 256, 160, 160}},
      {"nearest_interp_v2_3.tmp_0", {1, 64, 160, 160}},
      {"nearest_interp_v2_4.tmp_0", {1, 64, 160, 160}},
      {"nearest_interp_v2_5.tmp_0", {1, 64, 160, 160}},
      {"elementwise_add_7", {1, 56, 40, 40}},
      {"nearest_interp_v2_0.tmp_0", {1, 256, 40, 40}}};
  config->SetTRTDynamicShapeInfo(min_input_shape, max_input_shape,
                                 opt_input_shape);
}

TEST(test_det_mv3_db, analysis_gpu_bz4) {
  // init input data
  std::map<std::string, paddle::test::Record> my_input_data_map;
  my_input_data_map["x"] = PrepareInput(4, 640);
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
  CompareRecord(&truth_output_data, &infer_output_data, 1e-4);
  std::cout << "finish test" << std::endl;
}

TEST(test_det_mv3_db, multi_thread2_trt_fp32_dynamic_shape_bz2) {
  int thread_num = 2;  // thread > 2 may OOM
  // init input data
  std::map<std::string, paddle::test::Record> my_input_data_map;
  my_input_data_map["x"] = PrepareInput(2, 640);
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
      1 << 20, 2, 3, paddle_infer::PrecisionType::kFloat32, true, false);
  PrepareDynamicShape(&config, 4);
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
    CompareRecord(&truth_output_data, &infer_output_data, 1e-4);
  }

  std::cout << "finish multi-thread test" << std::endl;
}

TEST(test_det_mv3_db, multi_thread2_mkl_fp32_bz2) {
  int thread_num = 2;  // thread > 2 may OOM
  // init input data
  std::map<std::string, paddle::test::Record> my_input_data_map;
  my_input_data_map["x"] = PrepareInput(2, 640);
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
  config.DisableGpu();
  config.EnableMKLDNN();
  config.SetMkldnnCacheCapacity(10);
  config.SetCpuMathLibraryNumThreads(10);
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
    CompareRecord(&truth_output_data, &infer_output_data, 1e-4);
  }

  std::cout << "finish multi-thread test" << std::endl;
}

}  // namespace paddle_infer

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::google::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}
