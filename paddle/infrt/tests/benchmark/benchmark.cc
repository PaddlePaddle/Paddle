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

#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include "llvm/Support/raw_ostream.h"
#include "paddle/infrt/api/infrt_api.h"
#include "paddle/infrt/backends/host/phi_allocator.h"
#include "paddle/infrt/common/buffer.h"
#include "paddle/infrt/common/dtype.h"
#include "paddle/infrt/tests/timer.h"

using infrt::InfRtConfig;
using infrt::InfRtPredictor;
using infrt::CreateInfRtPredictor;

DEFINE_int32(layers, 0, "");
DEFINE_int32(num, 0, "");

namespace infrt {

void benchmark(size_t layers, size_t num) {
  const std::string tag =
      "l" + std::to_string(layers) + "n" + std::to_string(num);
  const std::string model_name = tag + ".pdmodel";
  const std::string param_name = tag + ".pdiparams";
  const std::string prefix =
      "/shixiaowei02/Paddle-InfRT/Paddle/tools/infrt/fake_models/elt_add/";

  InfRtConfig config;
  config.set_model_dir(prefix + model_name);
  config.set_param_dir(prefix + param_name);

  std::vector<std::string> shared_libs;

  std::unique_ptr<InfRtPredictor> predictor = CreateInfRtPredictor(config);

  ::infrt::backends::CpuPhiAllocator cpu_allocator;
  ::phi::DenseTensor* input = predictor->GetInput(0);
  input->Resize({static_cast<int>(num), static_cast<int>(num)});
  input->AllocateFrom(&cpu_allocator, ::phi::DataType::FLOAT32);
  auto* input_data = reinterpret_cast<float*>(input->data());
  for (int i = 0; i < input->numel(); i++) input_data[i] = 1.0;

  predictor->Run();

  ::phi::DenseTensor* output = predictor->GetOutput(0);
  float* output_data = reinterpret_cast<float*>(output->data());
  float sum;
  for (int64_t i = 0; i < output->numel(); ++i) {
    sum += output_data[i];
  }
  std::cout << "sum = " << sum << '\n';

  tests::BenchmarkStats timer;

  for (size_t i = 0; i < 9; ++i) {
    predictor->Run();
  }

  for (size_t j = 0; j < 100; ++j) {
    timer.Start();
    predictor->Run();
    timer.Stop();
  }
  std::cout << "\nlayers " << layers << ", num " << num << '\n';
  std::cout << "framework " << timer.Summerize({0.5});
  // auto* output = predictor->GetOutput(0);

  // TODO(Shixiaowei02): Automatic result validation for training then
  // inference.
  // auto* output_data = reinterpret_cast<float*>(output->data());

  // CHECK(output->dims() == ::phi::DDim({16, 10}));
}

/*
  for (auto layers : {1, 10, 50}) {
    for (auto num : {1, 100, 1000, 10000}) {
      benchmark(layers, num);
    }
  }
*/

TEST(InfRtPredictor, predictor) { benchmark(FLAGS_layers, FLAGS_num); }

}  // namespace infrt
