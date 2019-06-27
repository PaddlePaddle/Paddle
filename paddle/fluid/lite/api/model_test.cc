// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <glog/logging.h>
#include <string>
#include <vector>
#include "paddle/fluid/lite/api/paddle_api.h"
#include "paddle/fluid/lite/api/paddle_use_kernels.h"
#include "paddle/fluid/lite/api/paddle_use_ops.h"
#include "paddle/fluid/lite/api/paddle_use_passes.h"
#include "paddle/fluid/lite/api/test_helper.h"
#include "paddle/fluid/lite/core/cpu_info.h"
#include "paddle/fluid/lite/utils/string.h"

namespace paddle {
namespace lite_api {

void OutputOptModel(const std::string& load_model_dir,
                    const std::string& save_optimized_model_dir,
                    const std::vector<int64_t>& input_shape) {
  lite_api::CxxConfig config;
  config.set_model_dir(load_model_dir);
  config.set_preferred_place(Place{TARGET(kX86), PRECISION(kFloat)});
  config.set_valid_places({
      Place{TARGET(kX86), PRECISION(kFloat)},
      Place{TARGET(kARM), PRECISION(kFloat)},
  });
  auto predictor = lite_api::CreatePaddlePredictor(config);

  auto input_tensor = predictor->GetInput(0);
  input_tensor->Resize(input_shape);
  auto* data = input_tensor->mutable_data<float>();
  int input_num = 1;
  for (int i = 0; i < input_shape.size(); ++i) {
    input_num *= input_shape[i];
  }
  for (int i = 0; i < input_num; ++i) {
    data[i] = i;
  }
  predictor->Run();
  // delete old optimized model
  int ret = system(
      paddle::lite::string_format("rm -rf %s", save_optimized_model_dir.c_str())
          .c_str());
  if (ret == 0) {
    LOG(INFO) << "delete old optimized model " << save_optimized_model_dir;
  }
  predictor->SaveOptimizedModel(save_optimized_model_dir);
  LOG(INFO) << "Load model from " << load_model_dir;
  LOG(INFO) << "Save optimized model to " << save_optimized_model_dir;
}

#ifdef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
void Run(const std::vector<int64_t>& input_shape, const std::string& model_dir,
         const int repeat, const int thread_num, const int warmup_times = 10) {
  lite::DeviceInfo::Init();
  lite::DeviceInfo::Global().SetRunMode(lite::LITE_POWER_HIGH, thread_num);
  lite_api::MobileConfig config;
  config.set_model_dir(model_dir);

  auto predictor = lite_api::CreatePaddlePredictor(config);

  auto input_tensor = predictor->GetInput(0);
  input_tensor->Resize(input_shape);
  float* input_data = input_tensor->mutable_data<float>();
  int input_num = 1;
  for (int i = 0; i < input_shape.size(); ++i) {
    input_num *= input_shape[i];
  }
  for (int i = 0; i < input_num; ++i) {
    input_data[i] = i;
  }

  for (int i = 0; i < warmup_times; ++i) {
    predictor->Run();
  }

  auto start = lite::GetCurrentUS();
  for (int i = 0; i < repeat; ++i) {
    predictor->Run();
  }
  auto end = lite::GetCurrentUS();

  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << model_dir << ", threads num " << thread_num
            << ", warmup: " << warmup_times << ", repeats: " << repeat
            << ", spend " << (end - start) / repeat / 1000.0
            << " ms in average.";

  auto output = predictor->GetOutput(0);
  const float* out = output->data<float>();
  LOG(INFO) << "out " << out[0];
  LOG(INFO) << "out " << out[1];
  auto output_shape = output->shape();
  int output_num = 1;
  for (int i = 0; i < output_shape.size(); ++i) {
    output_num *= output_shape[i];
  }
  LOG(INFO) << "output_num: " << output_num;
}
#endif

}  // namespace lite_api
}  // namespace paddle

int main(int argc, char** argv) {
  if (argc < 4) {
    LOG(INFO) << "usage: " << argv[0] << " <model_dir> <repeat> <thread_num>";
    exit(0);
  }
  std::string load_model_dir = argv[1];
  std::string save_optimized_model_dir = load_model_dir + "opt2";

#ifdef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
  int repeat = std::stoi(argv[2]);
  int thread_num = std::stoi(argv[3]);
#endif

  std::vector<int64_t> input_shape{1, 3, 224, 224};

  // Output optimized model
  paddle::lite_api::OutputOptModel(load_model_dir, save_optimized_model_dir,
                                   input_shape);

#ifdef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
  // Run inference using optimized model
  paddle::lite_api::Run(input_shape, save_optimized_model_dir, repeat,
                        thread_num);
#endif

  return 0;
}
