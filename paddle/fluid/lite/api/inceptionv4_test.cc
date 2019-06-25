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

#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <vector>
#include "paddle/fluid/lite/api/cxx_api.h"
#include "paddle/fluid/lite/api/paddle_use_kernels.h"
#include "paddle/fluid/lite/api/paddle_use_ops.h"
#include "paddle/fluid/lite/api/paddle_use_passes.h"
#include "paddle/fluid/lite/api/test_helper.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {

#ifdef LITE_WITH_ARM
TEST(InceptionV4, test) {
  DeviceInfo::Init();
  DeviceInfo::Global().SetRunMode(LITE_POWER_HIGH, FLAGS_threads);
  lite::Predictor predictor;
  std::vector<Place> valid_places({Place{TARGET(kHost), PRECISION(kFloat)},
                                   Place{TARGET(kARM), PRECISION(kFloat)}});

  predictor.Build(FLAGS_model_dir, Place{TARGET(kARM), PRECISION(kFloat)},
                  valid_places);

  auto* input_tensor = predictor.GetInput(0);
  input_tensor->Resize(DDim(std::vector<DDim::value_type>({1, 3, 224, 224})));
  auto* data = input_tensor->mutable_data<float>();
  auto item_size = input_tensor->dims().production();
  for (int i = 0; i < item_size; i++) {
    data[i] = 1;
  }

  for (int i = 0; i < FLAGS_warmup; ++i) {
    predictor.Run();
  }

  auto start = GetCurrentUS();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    predictor.Run();
  }

  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << FLAGS_model_dir << ", threads num " << FLAGS_threads
            << ", warmup: " << FLAGS_warmup << ", repeats: " << FLAGS_repeats
            << ", spend " << (GetCurrentUS() - start) / FLAGS_repeats / 1000.0
            << " ms in average.";

  // std::vector<float> results({0.00078033, 0.00083865, 0.00060029, 0.00057083,
  //                            0.00070094, 0.00080584, 0.00044525, 0.00074907,
  //                            0.00059774, 0.00063654});
  //
  std::vector<std::vector<float>> results;
  // i = 1
  results.emplace_back(std::vector<float>(
      {0.0011684548,  0.0010390386,  0.0011301535,  0.0010133048,
       0.0010259597,  0.0010982729,  0.00093195855, 0.0009141837,
       0.00096620916, 0.00089982944, 0.0010064574,  0.0010474789,
       0.0009782845,  0.0009230255,  0.0010548076,  0.0010974824,
       0.0010612885,  0.00089107914, 0.0010112736,  0.00097655767}));
  auto* out = predictor.GetOutput(0);
  ASSERT_EQ(out->dims().size(), 2);
  ASSERT_EQ(out->dims()[0], 1);
  ASSERT_EQ(out->dims()[1], 1000);

  int step = 50;
  for (int i = 0; i < results.size(); ++i) {
    for (int j = 0; j < results[i].size(); ++j) {
      EXPECT_NEAR(out->data<float>()[j * step + (out->dims()[1] * i)],
                  results[i][j], 1e-6);
    }
  }
}
#endif

}  // namespace lite
}  // namespace paddle
