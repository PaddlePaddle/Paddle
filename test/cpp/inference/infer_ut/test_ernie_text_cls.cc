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

template <typename T>
T cRandom(int min, int max) {
  unsigned int seed = 100;
  return (min +
          static_cast<T>(max * rand_r(&seed) / static_cast<T>(RAND_MAX + 1)));
}

std::map<std::string, paddle::test::Record> PrepareInput(int batch_size) {
  // init input data
  int digit_length = 115;
  paddle::test::Record input_ids, segment_ids;
  int input_num = batch_size * digit_length;
  std::vector<int64_t> input_data(input_num, 1);
  std::vector<int64_t> segment_data(input_num, 0);
  srand((unsigned)time(NULL));
  for (int x = 0; x < input_data.size(); x++) {
    input_data[x] = cRandom<int>(1, 100);
  }
  input_ids.data = std::vector<float>(input_data.begin(), input_data.end());
  input_ids.shape = std::vector<int>{batch_size, digit_length};
  input_ids.type = paddle::PaddleDType::INT64;

  segment_ids.data =
      std::vector<float>(segment_data.begin(), segment_data.end());
  segment_ids.shape = std::vector<int>{batch_size, digit_length};
  segment_ids.type = paddle::PaddleDType::INT64;

  std::map<std::string, paddle::test::Record> my_input_data_map;
  my_input_data_map.insert({"input_ids", input_ids});
  my_input_data_map.insert({"token_type_ids", segment_ids});

  return my_input_data_map;
}

TEST(gpu_tester_ernie_text_cls, analysis_gpu_bz2_buffer) {
  // init input data
  auto my_input_data_map = PrepareInput(2);
  // init output data
  std::map<std::string, paddle::test::Record> infer_output_data,
      truth_output_data;
  // prepare ground truth config
  paddle_infer::Config config, config_no_ir;
  config_no_ir.SetModel(FLAGS_modeldir + "/inference.pdmodel",
                        FLAGS_modeldir + "/inference.pdiparams");
  config_no_ir.SwitchIrOptim(false);

  // prepare inference config from buffer
  std::string prog_file = FLAGS_modeldir + "/inference.pdmodel";
  std::string params_file = FLAGS_modeldir + "/inference.pdiparams";
  std::string prog_str = paddle::test::read_file(prog_file);
  std::string params_str = paddle::test::read_file(params_file);
  config.SetModelBuffer(
      prog_str.c_str(), prog_str.size(), params_str.c_str(), params_str.size());
  // get ground truth by disable ir
  paddle_infer::services::PredictorPool pred_pool_no_ir(config_no_ir, 1);
  SingleThreadPrediction(
      pred_pool_no_ir.Retrieve(0), &my_input_data_map, &truth_output_data, 1);
  // get infer results
  paddle_infer::services::PredictorPool pred_pool(config, 1);
  SingleThreadPrediction(
      pred_pool.Retrieve(0), &my_input_data_map, &infer_output_data);
  // check outputs
  CompareRecord(&truth_output_data, &infer_output_data);
  std::cout << "finish test" << std::endl;
}

TEST(mkldnn_tester_ernie_text_cls, multi_thread4_mkl_fp32_bz2) {
  int thread_num = 4;
  // init input data
  auto my_input_data_map = PrepareInput(2);
  // init output data
  std::map<std::string, paddle::test::Record> infer_output_data,
      truth_output_data;
  // prepare ground truth config
  paddle_infer::Config config, config_no_ir;
  config_no_ir.SetModel(FLAGS_modeldir + "/inference.pdmodel",
                        FLAGS_modeldir + "/inference.pdiparams");
  config.DisableGpu();
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
    CompareRecord(&truth_output_data, &infer_output_data);
  }

  std::cout << "finish multi-thread test" << std::endl;
}

}  // namespace paddle_infer

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}
