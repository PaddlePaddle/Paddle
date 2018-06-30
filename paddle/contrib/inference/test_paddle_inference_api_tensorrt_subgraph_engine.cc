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
#include "paddle/contrib/inference/paddle_inference_api.h"

namespace paddle {

DEFINE_string(dirname, "", "Directory of the inference model.");

void Main(bool use_gpu) {
  //# 1. Create PaddlePredictor with a config.
  TensorRTConfig config;
  config.model_dir = FLAGS_dirname + "word2vec.inference.model";
  config.use_gpu = use_gpu;
  config.fraction_of_gpu_memory = 0.15;
  config.device = 0;
  auto predictor =
      CreatePaddlePredictor<TensorRTConfig,
                            PaddleEngineKind::kAutoMixedTensorRT>(config);

  for (int batch_id = 0; batch_id < 3; batch_id++) {
    //# 2. Prepare input.
    int64_t data[4] = {1, 2, 3, 4};

    PaddleTensor tensor{.name = "",
                        .shape = std::vector<int>({4, 1}),
                        .data = PaddleBuf(data, sizeof(data)),
                        .dtype = PaddleDType::INT64};

    // For simplicity, we set all the slots with the same data.
    std::vector<PaddleTensor> slots(4, tensor);

    //# 3. Run
    std::vector<PaddleTensor> outputs;
    CHECK(predictor->Run(slots, &outputs));

    //# 4. Get output.
    ASSERT_EQ(outputs.size(), 1UL);
    LOG(INFO) << "output buffer size: " << outputs.front().data.length();
    const size_t num_elements = outputs.front().data.length() / sizeof(float);
    // The outputs' buffers are in CPU memory.
    for (size_t i = 0; i < std::min(5UL, num_elements); i++) {
      LOG(INFO) << static_cast<float*>(outputs.front().data.data())[i];
    }
  }
}

TEST(paddle_inference_api_tensorrt_subgraph_engine, main) { Main(true); }

}  // namespace paddle