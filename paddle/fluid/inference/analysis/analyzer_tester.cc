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

#include "paddle/fluid/inference/analysis/analyzer.h"
#include <google/protobuf/text_format.h>
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/inference/analysis/ut_helper.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"

namespace paddle {
namespace inference {
namespace analysis {

TEST(Analyzer, analysis_without_tensorrt) {
  FLAGS_IA_enable_tensorrt_subgraph_engine = false;
  Argument argument;
  argument.fluid_model_dir.reset(new std::string(FLAGS_inference_model_dir));
  Analyzer analyser;
  analyser.Run(&argument);
}

TEST(Analyzer, analysis_with_tensorrt) {
  FLAGS_IA_enable_tensorrt_subgraph_engine = true;
  Argument argument;
  argument.fluid_model_dir.reset(new std::string(FLAGS_inference_model_dir));
  Analyzer analyser;
  analyser.Run(&argument);
}

void TestWord2vecPrediction(const std::string& model_path) {
  NativeConfig config;
  config.model_dir = model_path;
  config.use_gpu = false;
  config.device = 0;
  auto predictor =
      ::paddle::CreatePaddlePredictor<NativeConfig, PaddleEngineKind::kNative>(
          config);

  // One single batch

  int64_t data[4] = {1, 2, 3, 4};
  PaddleTensor tensor;
  tensor.shape = std::vector<int>({4, 1});
  tensor.data = PaddleBuf(data, sizeof(data));
  tensor.dtype = PaddleDType::INT64;

  // For simplicity, we set all the slots with the same data.
  std::vector<PaddleTensor> slots(4, tensor);
  std::vector<PaddleTensor> outputs;
  CHECK(predictor->Run(slots, &outputs));

  PADDLE_ENFORCE(outputs.size(), 1UL);
  // Check the output buffer size and result of each tid.
  PADDLE_ENFORCE(outputs.front().data.length(), 33168UL);
  float result[5] = {0.00129761, 0.00151112, 0.000423564, 0.00108815,
                     0.000932706};
  const size_t num_elements = outputs.front().data.length() / sizeof(float);
  // The outputs' buffers are in CPU memory.
  for (size_t i = 0; i < std::min(5UL, num_elements); i++) {
    LOG(INFO) << "data: "
              << static_cast<float*>(outputs.front().data.data())[i];
    PADDLE_ENFORCE(static_cast<float*>(outputs.front().data.data())[i],
                   result[i]);
  }
}

// Turn on the IR pass supportion, run a real inference and check the result.
TEST(Analyzer, SupportIRPass) {
  FLAGS_IA_enable_ir = true;
  FLAGS_IA_enable_tensorrt_subgraph_engine = false;
  FLAGS_IA_output_storage_path = "./analysis.out";

  Argument argument(FLAGS_inference_model_dir);
  argument.model_output_store_path.reset(new std::string("./analysis.out"));

  Analyzer analyzer;
  analyzer.Run(&argument);

  // Should get the transformed model stored to ./analysis.out
  ASSERT_TRUE(PathExists("./analysis.out"));

  // Inference from this path.
  TestWord2vecPrediction("./analysis.out");
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle

USE_PASS(fc_fuse_pass);
USE_PASS(graph_viz_pass);
USE_PASS(infer_clean_graph_pass);
