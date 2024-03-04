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
DEFINE_string(int8dir, "", "Directory of the quant inference model.");
DEFINE_string(datadir, "", "Directory of the infer data.");

namespace paddle_infer {

paddle::test::Record PrepareInput(int batch_size) {
  // init input data
  int channel = 3;
  int width = 224;
  int height = 224;
  paddle::test::Record image_Record;
  int input_num = batch_size * channel * width * height;

  // load from binary data
  std::ifstream fs(FLAGS_datadir, std::ifstream::binary);
  EXPECT_TRUE(fs.is_open());
  CHECK(fs.is_open());

  float* input = new float[input_num];
  memset(input, 0, input_num * sizeof(float));
  auto input_data_tmp = input;
  for (int i = 0; i < input_num; ++i) {
    fs.read(reinterpret_cast<char*>(input_data_tmp), sizeof(*input_data_tmp));
    input_data_tmp++;
  }
  int label = 0;
  fs.read(reinterpret_cast<char*>(&label), sizeof(label));
  fs.close();

  std::vector<float> input_data{input, input + input_num};
  image_Record.data = input_data;
  image_Record.shape = std::vector<int>{batch_size, channel, width, height};
  image_Record.type = paddle::PaddleDType::FLOAT32;
  image_Record.label = label;
  return image_Record;
}

TEST(DISABLED_tensorrt_tester_resnet50_quant, multi_thread4_trt_int8_bz1) {
  int thread_num = 4;
  // init input data
  std::map<std::string, paddle::test::Record> input_data_map;
  input_data_map["image"] = PrepareInput(1);
  // init output data
  std::map<std::string, paddle::test::Record> infer_output_data;
  // prepare inference config
  paddle_infer::Config config;
  config.SetModel(FLAGS_int8dir);
  config.EnableUseGpu(1000, 0);
  config.EnableTensorRtEngine(
      1 << 20, 10, 3, paddle_infer::PrecisionType::kInt8, false, false);
  // get infer results from multi threads
  std::vector<std::thread> threads;
  services::PredictorPool pred_pool(config, thread_num);
  for (int i = 0; i < thread_num; ++i) {
    threads.emplace_back(paddle::test::SingleThreadPrediction,
                         pred_pool.Retrieve(i),
                         &input_data_map,
                         &infer_output_data,
                         5);
  }

  // thread join & check outputs
  for (int i = 0; i < thread_num; ++i) {
    LOG(INFO) << "join tid : " << i;
    threads[i].join();

    // check outputs
    std::vector<int> index(1000);
    std::iota(index.begin(), index.end(), 0);
    auto out_data =
        infer_output_data["save_infer_model/scale_0.tmp_0"].data.data();
    std::sort(index.begin(), index.end(), [out_data](size_t i1, size_t i2) {
      return out_data[i1] > out_data[i2];
    });
    // compare inference & ground truth label
    ASSERT_EQ(index[0], input_data_map["image"].label);
  }

  std::cout << "finish test" << std::endl;
}

TEST(DISABLED_tensorrt_tester_resnet50_quant, multi_thread_multi_instance) {
  int thread_num = 4;
  // init input data
  std::map<std::string, paddle::test::Record> input_data_fp32, input_data_quant;
  input_data_quant["image"] = PrepareInput(1);
  input_data_fp32["inputs"] = PrepareInput(1);

  // init output data
  std::map<std::string, paddle::test::Record> infer_output_data;
  // prepare inference config
  paddle_infer::Config config_fp32, config_quant;
  config_fp32.SetModel(FLAGS_modeldir + "/inference.pdmodel",
                       FLAGS_modeldir + "/inference.pdiparams");
  config_fp32.EnableUseGpu(1000, 0);
  config_fp32.EnableTensorRtEngine(
      1 << 20, 10, 3, paddle_infer::PrecisionType::kFloat32, false, false);

  config_quant.SetModel(FLAGS_int8dir);
  config_quant.EnableUseGpu(1000, 0);
  config_quant.EnableTensorRtEngine(
      1 << 20, 10, 3, paddle_infer::PrecisionType::kInt8, false, false);

  // get infer results from multi threads
  std::vector<std::thread> threads;
  services::PredictorPool pred_pool_fp32(config_fp32, thread_num);
  services::PredictorPool pred_pool_quant(config_quant, thread_num);
  for (int i = 0; i < thread_num; ++i) {
    if (i % 2 == 0) {
      threads.emplace_back(paddle::test::SingleThreadPrediction,
                           pred_pool_fp32.Retrieve(i),
                           &input_data_fp32,
                           &infer_output_data,
                           5);
    } else {
      threads.emplace_back(paddle::test::SingleThreadPrediction,
                           pred_pool_quant.Retrieve(i),
                           &input_data_quant,
                           &infer_output_data,
                           5);
    }
  }

  // thread join & check outputs
  for (int i = 0; i < thread_num; ++i) {
    LOG(INFO) << "join tid : " << i;
    std::vector<int> index(1000);
    threads[i].join();
    if (i % 2 == 0) {
      // check outputs
      std::iota(index.begin(), index.end(), 0);
      auto out_data =
          infer_output_data["save_infer_model/scale_0.tmp_0"].data.data();
      std::sort(index.begin(), index.end(), [out_data](size_t i1, size_t i2) {
        return out_data[i1] > out_data[i2];
      });
      // compare inference & ground truth label
      ASSERT_EQ(index[0], input_data_fp32["inputs"].label);
    } else {
      // check outputs
      std::iota(index.begin(), index.end(), 0);
      auto out_data =
          infer_output_data["save_infer_model/scale_0.tmp_0"].data.data();
      std::sort(index.begin(), index.end(), [out_data](size_t i1, size_t i2) {
        return out_data[i1] > out_data[i2];
      });
      // compare inference & ground truth label
      ASSERT_EQ(index[0], input_data_quant["image"].label);
    }
  }
}

}  // namespace paddle_infer

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}
