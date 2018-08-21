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
#include <glog/logging.h>  // use glog instead of PADDLE_ENFORCE to avoid importing other paddle header files.
#include <gtest/gtest.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/inference/analysis/analyzer.h"
#include "paddle/fluid/inference/analysis/argument.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"

DEFINE_string(modeldir, "", "Directory of the inference model.");

using namespace paddle;
using namespace paddle::inference::analysis;

void PrepareInputs(std::vector<PaddleTensor>* inputs);

void Main(int max_batch) {
  FLAGS_IA_enabe_ir = true;
  FLAGS_IA_enable_tensorrt_subgraph_engine = false;
  FLAGS_IA_output_storage_path = "./analysis.out";

  Argument argument(FLAGS_modeldir);
  argument.model_output_store_path.reset(new std::string("./analysis.out"));

  Analyzer analyzer;
  analyzer.Run(&argument);

  // Should get the transformed model stored to ./analysis.out
  ASSERT_TRUE(PathExists("./analysis.out"));

  NativeConfig config;
  config.prog_file = "./analysis.out/__model__";
  config.param_file = "./analysis.out/__params__";
  config.use_gpu = false;
  config.device = 0;

  std::vector<std::vector<int>> shapes({{4},
                                        {1, 50, 12},
                                        {1, 50, 19},
                                        {1, 50, 1},
                                        {4, 50, 1},
                                        {1, 50, 1},
                                        {5, 50, 1},
                                        {7, 50, 1},
                                        {3, 50, 1}});

  std::vector<PaddleTensor> inputs;
  for (auto& shape : shapes) {
    // For simplicity, max_batch as the batch_size
    // shape.insert(shape.begin(), max_batch);
    shape.insert(shape.begin(), 1);
    PaddleTensor feature{
        .name = "",
        .shape = shape,
        .lod = std::vector<std::vector<size_t>>(),
        .data = PaddleBuf(sizeof(float) *
                          std::accumulate(shape.begin(), shape.end(), 1,
                                          [](int a, int b) { return a * b; })),
        .dtype = PaddleDType::FLOAT32};
    inputs.emplace_back(std::move(feature));
  }

  auto predictor =
      CreatePaddlePredictor<NativeConfig, PaddleEngineKind::kNative>(config);

  // { batch begin

  std::vector<PaddleTensor> outputs;

  CHECK(predictor->Run(inputs, &outputs));

  // } batch end
}

TEST(ditu, main) { Main(1); }
