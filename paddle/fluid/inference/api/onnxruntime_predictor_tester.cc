// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/api/onnxruntime_predictor.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <string>
#include <thread>  // NOLINT
#include <vector>
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_api.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/tests/api/tester_helper.h"
#include "paddle/fluid/inference/utils/io_utils.h"
#include "paddle/fluid/platform/cpu_info.h"

DEFINE_string(dirname, "", "dirname to tests.");

namespace paddle {

TEST(ONNXRuntimePredictor, onnxruntime_on) {
  AnalysisConfig config;
  config.SetModel(FLAGS_dirname + "/inference.pdmodel",
                  FLAGS_dirname + "/inference.pdiparams");
  config.EnableONNXRuntime();
  config.EnableORTOptimization();
  config.SetCpuMathLibraryNumThreads(2);
  LOG(INFO) << config.Summary();

  auto _predictor =
      CreatePaddlePredictor<AnalysisConfig,
                            paddle::PaddleEngineKind::kONNXRuntime>(config);
  ASSERT_TRUE(_predictor);
  auto* predictor = static_cast<ONNXRuntimePredictor*>(_predictor.get());

  ASSERT_TRUE(predictor);
  ASSERT_TRUE(!predictor->Clone());
  ASSERT_TRUE(predictor->scope_);
  ASSERT_TRUE(predictor->sub_scope_);
  ASSERT_EQ(predictor->scope_->parent(), nullptr);
  ASSERT_EQ(predictor->sub_scope_->parent(), predictor->scope_.get());
  // Dummy Input Data
  std::vector<int64_t> input_shape = {-1, 3, 224, 224};
  std::vector<float> input_data(1 * 3 * 224 * 224, 1.0);
  std::vector<float> out_data;
  out_data.resize(1000);

  // testing all interfaces
  auto input_names = predictor->GetInputNames();
  auto output_names = predictor->GetOutputNames();
  auto get_input_shape = predictor->GetInputTensorShape();

  ASSERT_EQ(input_names.size(), 1UL);
  ASSERT_EQ(output_names.size(), 1UL);
  ASSERT_EQ(input_names[0], "inputs");
  ASSERT_EQ(output_names[0], "save_infer_model/scale_0.tmp_1");
  ASSERT_EQ(get_input_shape["inputs"], input_shape);

  auto input_tensor = predictor->GetInputTensor(input_names[0]);
  input_tensor->Reshape({1, 3, 224, 224});
  auto output_tensor = predictor->GetOutputTensor(output_names[0]);

  input_tensor->CopyFromCpu(input_data.data());
  ASSERT_TRUE(predictor->ZeroCopyRun());
  output_tensor->CopyToCpu(out_data.data());

  predictor->TryShrinkMemory();
}

}  // namespace paddle
