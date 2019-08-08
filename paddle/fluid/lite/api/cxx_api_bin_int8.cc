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

#include "paddle/fluid/lite/api/cxx_api.h"
#include <chrono>  // NOLINT
#include "paddle/fluid/lite/api/paddle_use_kernels.h"
#include "paddle/fluid/lite/api/paddle_use_ops.h"
#include "paddle/fluid/lite/api/paddle_use_passes.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {

using Time = decltype(std::chrono::high_resolution_clock::now());
Time time() { return std::chrono::high_resolution_clock::now(); }
double time_diff(Time t1, Time t2) {
  typedef std::chrono::microseconds ms;
  auto diff = t2 - t1;
  ms counter = std::chrono::duration_cast<ms>(diff);
  return counter.count() / 1000.0;
}

void Run(const char* model_dir, int repeat) {
#ifdef LITE_WITH_ARM
  DeviceInfo::Init();
  DeviceInfo::Global().SetRunMode(LITE_POWER_HIGH, 1);
#endif
  lite::Predictor predictor;
  std::vector<Place> valid_places({
      Place{TARGET(kHost), PRECISION(kFloat)},
      Place{TARGET(kARM), PRECISION(kFloat)},
      Place{TARGET(kARM), PRECISION(kInt8)},
  });

  predictor.Build(model_dir, Place{TARGET(kARM), PRECISION(kInt8)},
                  valid_places);

  auto* input_tensor = predictor.GetInput(0);
  input_tensor->Resize(DDim(std::vector<DDim::value_type>({1, 3, 224, 224})));
  auto* data = input_tensor->mutable_data<float>();
  for (int i = 0; i < input_tensor->dims().production(); i++) {
    data[i] = 1;
  }

  for (int i = 0; i < 10; i++) predictor.Run();
  auto time1 = time();
  for (int i = 0; i < repeat; i++) predictor.Run();
  auto time2 = time();
  std::cout << " predict cost: " << time_diff(time1, time2) / repeat << "ms"
            << std::endl;

  auto* out = predictor.GetOutput(0);
  LOG(INFO) << out << " memory size " << out->data_size();
  LOG(INFO) << "dims " << out->dims();
  LOG(INFO) << "out data size: " << out->data_size();
  /*
  float sum = 0.;
  for (int i = 0; i < out->data_size(); i++) {
     LOG(INFO) << "out " << out->data<float>()[i];
     sum += out->data<float>()[i];
  }
  LOG(INFO) << sum;
  */
}

}  // namespace lite
}  // namespace paddle

int main(int argc, char** argv) {
  CHECK_EQ(argc, 3) << "usage: ./cmd <model_dir> <repeat>";
  paddle::lite::Run(argv[1], std::stoi(argv[2]));

  return 0;
}
