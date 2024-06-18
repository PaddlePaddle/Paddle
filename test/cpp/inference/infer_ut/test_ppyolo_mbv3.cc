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

std::map<std::string, paddle::test::Record> PrepareInput(int batch_size) {
  // init input data
  int channel = 3;
  int width = 320;
  int height = 320;
  paddle::test::Record image, im_shape, scale_factor;
  int input_num = batch_size * channel * width * height;
  int shape_num = batch_size * 2;
  std::vector<float> image_data(input_num, 1);
  for (int i = 1; i < input_num + 1; ++i) {
    image_data[i] = i % 10 * 0.5;
  }
  std::vector<float> im_shape_data(shape_num, 1);
  std::vector<float> scale_factor_data(shape_num, 1);

  image.data = std::vector<float>(image_data.begin(), image_data.end());
  image.shape = std::vector<int>{batch_size, channel, width, height};
  image.type = paddle::PaddleDType::FLOAT32;

  im_shape.data =
      std::vector<float>(im_shape_data.begin(), im_shape_data.end());
  im_shape.shape = std::vector<int>{batch_size, 2};
  im_shape.type = paddle::PaddleDType::FLOAT32;

  scale_factor.data =
      std::vector<float>(scale_factor_data.begin(), scale_factor_data.end());
  scale_factor.shape = std::vector<int>{batch_size, 2};
  scale_factor.type = paddle::PaddleDType::FLOAT32;

  std::map<std::string, paddle::test::Record> input_data_map;
  input_data_map.insert({"image", image});
  input_data_map.insert({"im_shape", im_shape});
  input_data_map.insert({"scale_factor", scale_factor});

  return input_data_map;
}

TEST(tensorrt_tester_ppyolo_mbv3, multi_thread4_trt_fp32_bz2) {
  int thread_num = 4;
  // init input data
  auto input_data_map = PrepareInput(2);
  // init output data
  std::map<std::string, paddle::test::Record> infer_output_data,
      truth_output_data;
  // prepare ground truth config
  paddle_infer::Config config, config_no_ir;
  config_no_ir.SetModel(FLAGS_modeldir + "/model.pdmodel",
                        FLAGS_modeldir + "/model.pdiparams");
  config_no_ir.EnableUseGpu(100, 0);
  config_no_ir.SwitchIrOptim(false);
  // prepare inference config
  config.SetModel(FLAGS_modeldir + "/model.pdmodel",
                  FLAGS_modeldir + "/model.pdiparams");
  config.EnableUseGpu(100, 0);
  config.EnableTensorRtEngine(
      1 << 25, 2, 3, paddle_infer::PrecisionType::kFloat32, false, false);
  LOG(INFO) << config.Summary();
  // get ground truth by disable ir
  paddle_infer::services::PredictorPool pred_pool_no_ir(config_no_ir, 1);
  SingleThreadPrediction(
      pred_pool_no_ir.Retrieve(0), &input_data_map, &truth_output_data, 1);

  // get infer results from multi threads
  std::vector<std::thread> threads;
  services::PredictorPool pred_pool(config, thread_num);
  for (int i = 0; i < thread_num; ++i) {
    threads.emplace_back(paddle::test::SingleThreadPrediction,
                         pred_pool.Retrieve(i),
                         &input_data_map,
                         &infer_output_data,
                         2);
  }

  // thread join & check outputs
  for (int i = 0; i < thread_num; ++i) {
    LOG(INFO) << "join tid : " << i;
    threads[i].join();
    CompareRecord(&truth_output_data, &infer_output_data, 0.18);
    // TODO(OliverLPH): precision set to 1e-2 since input is fake, change to
    // real input later
  }

  std::cout << "finish multi-thread test" << std::endl;
}

TEST(DISABLED_mkldnn_tester_ppyolo_mbv3, multi_thread4_mkl_bz2) {
  // TODO(OliverLPH): onednn multi thread will fail
  int thread_num = 4;
  // init input data
  auto input_data_map = PrepareInput(2);
  // init output data
  std::map<std::string, paddle::test::Record> infer_output_data,
      truth_output_data;
  // prepare ground truth config
  paddle_infer::Config config, config_no_ir;
  config_no_ir.SetModel(FLAGS_modeldir + "/model.pdmodel",
                        FLAGS_modeldir + "/model.pdiparams");
  config_no_ir.DisableGpu();
  config_no_ir.SwitchIrOptim(false);
  // prepare inference config
  config.SetModel(FLAGS_modeldir + "/model.pdmodel",
                  FLAGS_modeldir + "/model.pdiparams");
  config.DisableGpu();
  config.EnableMKLDNN();
  config.SetMkldnnCacheCapacity(10);
  config.SetCpuMathLibraryNumThreads(10);
  LOG(INFO) << config.Summary();
  // get ground truth by disable ir
  paddle_infer::services::PredictorPool pred_pool_no_ir(config_no_ir, 1);
  SingleThreadPrediction(
      pred_pool_no_ir.Retrieve(0), &input_data_map, &truth_output_data, 1);

  // get infer results from multi threads
  std::vector<std::thread> threads;
  services::PredictorPool pred_pool(config, thread_num);
  for (int i = 0; i < thread_num; ++i) {
    threads.emplace_back(paddle::test::SingleThreadPrediction,
                         pred_pool.Retrieve(i),
                         &input_data_map,
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
