/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/common/flags.h"
#include "test/cpp/inference/api/tester_helper.h"

namespace paddle {
namespace inference {

void ErnieInputData(const int &total_batch_size,
                    const bool enable_fp16,
                    std::vector<PaddleTensor> *inputs) {
  const int input_num = total_batch_size * 128 * 1;
  std::vector<int64_t> placeholder_012(input_num, 1);
  std::vector<float> placeholder_3(input_num, 1);

  for (int i = 0; i < 4; i++) {
    PaddleTensor in;
    in.name = "placeholder_" + std::to_string(i);
    in.shape = {total_batch_size, 128, 1};
    if (i < 3) {
      in.data = PaddleBuf(static_cast<void *>(placeholder_012.data()),
                          input_num * sizeof(int64_t));
      in.dtype = PaddleDType::INT64;
    } else {
      in.data = PaddleBuf(static_cast<void *>(placeholder_3.data()),
                          input_num * sizeof(float));
      in.dtype = PaddleDType::FLOAT32;
      if (enable_fp16) {
        ConvertFP32toFP16(in);
      }
    }
    inputs->push_back(std::move(in));
  }
}

void Resnet50InputData(const int &total_batch_size,
                       const bool enable_fp16,
                       std::vector<paddle::PaddleTensor> *inputs) {
  const int input_num = total_batch_size * 3 * 318 * 318;
  std::vector<float> input(input_num, 1);
  PaddleTensor in;
  in.shape = {total_batch_size, 3, 318, 318};
  in.data =
      PaddleBuf(static_cast<void *>(input.data()), input_num * sizeof(float));
  in.dtype = PaddleDType::FLOAT32;
  if (enable_fp16) {
    ConvertFP32toFP16(in);
  }
  inputs->push_back(std::move(in));
}

// performance profile
TEST(Analyzer_ipu_fp16, performance_profile) {
  AnalysisConfig config;
  std::vector<PaddleTensor> inputs;
  std::vector<std::vector<PaddleTensor>> outputs;

  int total_batch_size = FLAGS_ipu_micro_batch_size * FLAGS_ipu_replica_num;
  if (FLAGS_ipu_enable_pipelining) {
    // if device_num > 1 and pipelining is enabled, the total batch size =
    // micro_batch_size * device_num(batches_per_step) * replica_num
    total_batch_size = FLAGS_ipu_micro_batch_size * FLAGS_ipu_batches_per_step *
                       FLAGS_ipu_replica_num;
  }

  if (FLAGS_model_name == "Resnet50") {
    config.SetModel(FLAGS_infer_model + "/model/model",
                    FLAGS_infer_model + "/model/params");
    Resnet50InputData(total_batch_size, FLAGS_ipu_enable_fp16, &inputs);
  } else if (FLAGS_model_name == "Ernie") {
    config.SetModel(FLAGS_infer_model + "/model/");
    ErnieInputData(total_batch_size, FLAGS_ipu_enable_fp16, &inputs);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Only support Resnet50 and Ernie Currently"));
  }
  // ipu_device_num, ipu_micro_batch_size, ipu_enable_pipelining,
  // ipu_batches_per_step
  config.EnableIpu(FLAGS_ipu_device_num,
                   FLAGS_ipu_micro_batch_size,
                   FLAGS_ipu_enable_pipelining,
                   FLAGS_ipu_batches_per_step);
  // ipu_enable_fp16, ipu_replica_num, ipu_available_memory_proportion,
  // ipu_enable_half_partial
  config.SetIpuConfig(FLAGS_ipu_enable_fp16,
                      FLAGS_ipu_replica_num,
                      FLAGS_ipu_available_memory_proportion,
                      FLAGS_ipu_enable_half_partial);

  TestPrediction(reinterpret_cast<const PaddlePredictor::Config *>(&config),
                 {inputs},
                 &outputs,
                 1);
}

}  // namespace inference
}  // namespace paddle
