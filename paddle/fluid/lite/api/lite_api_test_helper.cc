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

#include "paddle/fluid/lite/api/lite_api_test_helper.h"
#include <vector>

DEFINE_string(model_dir, "", "");
DEFINE_string(optimized_model, "", "");

namespace paddle {
namespace lite {

const lite::Tensor* RunHvyModel() {
  lite::Predictor predictor;
#ifndef LITE_WITH_CUDA
  std::vector<Place> valid_places({Place{TARGET(kHost), PRECISION(kFloat)},
                                   Place{TARGET(kX86), PRECISION(kFloat)}});
#else
  std::vector<Place> valid_places({
      Place{TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW)},
      Place{TARGET(kCUDA), PRECISION(kFloat), DATALAYOUT(kNCHW)},
      Place{TARGET(kCUDA), PRECISION(kAny), DATALAYOUT(kNCHW)},
      Place{TARGET(kHost), PRECISION(kAny), DATALAYOUT(kNCHW)},
      Place{TARGET(kCUDA), PRECISION(kAny), DATALAYOUT(kAny)},
      Place{TARGET(kHost), PRECISION(kAny), DATALAYOUT(kAny)},
  });
#endif

  predictor.Build(FLAGS_model_dir,
                  Place{TARGET(kX86), PRECISION(kFloat)},  // origin cuda
                  valid_places);

  auto* input_tensor = predictor.GetInput(0);
  input_tensor->Resize(DDim(std::vector<DDim::value_type>({100, 100})));
  auto* data = input_tensor->mutable_data<float>();
  for (int i = 0; i < 100 * 100; i++) {
    data[i] = i;
  }

  // LOG(INFO) << "input " << *input_tensor;

  predictor.Run();

  const auto* out = predictor.GetOutput(0);
  return out;
}

}  // namespace lite
}  // namespace paddle
