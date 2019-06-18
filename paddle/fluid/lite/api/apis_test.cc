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

/*
 * We test multiple apis here.
 */
#include <gtest/gtest.h>
#include <sstream>
#include <vector>
#include "paddle/fluid/lite/api/cxx_api.h"
#include "paddle/fluid/lite/api/light_api.h"
#include "paddle/fluid/lite/core/mir/pass_registry.h"
#include "paddle/fluid/lite/core/mir/use_passes.h"
#include "paddle/fluid/lite/kernels/use_kernels.h"
#include "paddle/fluid/lite/operators/use_ops.h"

DEFINE_string(model_dir, "", "");
DEFINE_string(optimized_model, "", "");

namespace paddle {
namespace lite {

void SetConstInput(lite::Tensor* x) {
  x->Resize(DDim(std::vector<DDim::value_type>({100, 100})));
  auto* data = x->mutable_data<float>();
  for (int i = 0; i < 100 * 100; i++) {
    data[i] = i;
  }
}

bool CompareTensors(const std::string& name, const ExecutorLite& cxx_api,
                    const LightPredictor& light_api) {
  const auto* a = cxx_api.GetTensor(name);
  const auto* b = light_api.GetTensor(name);
  return TensorCompareWith(*a, *b);
}

#ifndef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
TEST(CXXApi_LightApi, save_and_load_model) {
  lite::ExecutorLite cxx_api;
  lite::LightPredictor light_api;

  // CXXAPi
  {
    std::vector<Place> valid_places({Place{TARGET(kHost), PRECISION(kFloat)},
                                     Place{TARGET(kX86), PRECISION(kFloat)}});
    cxx_api.Build(FLAGS_model_dir, Place{TARGET(kCUDA), PRECISION(kFloat)},
                  valid_places);

    auto* x = cxx_api.GetInput(0);
    SetConstInput(x);

    cxx_api.Run();

    LOG(INFO) << "Save optimized model to " << FLAGS_optimized_model;
    cxx_api.SaveModel(FLAGS_optimized_model);
  }

  // LightApi
  {
    light_api.Build(FLAGS_optimized_model);

    auto* x = light_api.GetInput(0);
    SetConstInput(x);

    light_api.Run();
  }

  const auto* cxx_out = cxx_api.GetOutput(0);
  const auto* light_out = light_api.GetOutput(0);
  ASSERT_TRUE(TensorCompareWith(*cxx_out, *light_out));

  std::vector<std::string> tensors_with_order({
      "a", "fc_0.w_0", "fc_0.tmp_0", "scale_0.tmp_0",
  });

  for (const auto& tensor_name : tensors_with_order) {
    ASSERT_TRUE(CompareTensors(tensor_name, cxx_api, light_api));
  }
}
#endif  // LITE_WITH_LIGHT_WEIGHT_FRAMEWORK

}  // namespace lite
}  // namespace paddle
