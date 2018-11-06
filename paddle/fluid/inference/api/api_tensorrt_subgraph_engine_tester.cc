// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "paddle/fluid/inference/analysis/analyzer.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"

namespace paddle {

using contrib::MixedRTConfig;

DEFINE_string(dirname, "", "Directory of the inference model.");

void CompareTensorRTWithFluid(bool enable_tensorrt) {
  FLAGS_IA_enable_tensorrt_subgraph_engine = enable_tensorrt;

  //# 1. Create PaddlePredictor with a config.
  NativeConfig config0;
  config0.model_dir = FLAGS_dirname;
  config0.use_gpu = true;
  config0.fraction_of_gpu_memory = 0.3;
  config0.device = 0;

  MixedRTConfig config1;
  config1.model_dir = FLAGS_dirname;
  config1.use_gpu = true;
  config1.fraction_of_gpu_memory = 0.3;
  config1.device = 0;
  config1.max_batch_size = 10;

  auto predictor0 = CreatePaddlePredictor<NativeConfig>(config0);
  auto predictor1 = CreatePaddlePredictor<MixedRTConfig>(config1);

  for (int batch_id = 0; batch_id < 1; batch_id++) {
    //# 2. Prepare input.
    std::vector<int64_t> data(20);
    for (int i = 0; i < 20; i++) data[i] = i;

    PaddleTensor tensor;
    tensor.shape = std::vector<int>({10, 1});
    tensor.data = PaddleBuf(data.data(), data.size() * sizeof(int64_t));
    tensor.dtype = PaddleDType::INT64;

    // For simplicity, we set all the slots with the same data.
    std::vector<PaddleTensor> slots(4, tensor);

    //# 3. Run
    std::vector<PaddleTensor> outputs0;
    std::vector<PaddleTensor> outputs1;
    CHECK(predictor0->Run(slots, &outputs0));
    CHECK(predictor1->Run(slots, &outputs1, 10));

    //# 4. Get output.
    ASSERT_EQ(outputs0.size(), 1UL);
    ASSERT_EQ(outputs1.size(), 1UL);

    const size_t num_elements = outputs0.front().data.length() / sizeof(float);
    const size_t num_elements1 = outputs1.front().data.length() / sizeof(float);
    EXPECT_EQ(num_elements, num_elements1);

    auto *data0 = static_cast<float *>(outputs0.front().data.data());
    auto *data1 = static_cast<float *>(outputs1.front().data.data());

    ASSERT_GT(num_elements, 0UL);
    for (size_t i = 0; i < std::min(num_elements, num_elements1); i++) {
      EXPECT_NEAR(data0[i], data1[i], 1e-3);
    }
  }
}

TEST(paddle_inference_api_tensorrt_subgraph_engine, without_tensorrt) {
  CompareTensorRTWithFluid(false);
}

TEST(paddle_inference_api_tensorrt_subgraph_engine, with_tensorrt) {
  CompareTensorRTWithFluid(true);
}

}  // namespace paddle
