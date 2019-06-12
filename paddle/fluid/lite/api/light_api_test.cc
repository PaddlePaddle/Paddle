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

#include "paddle/fluid/lite/api/light_api.h"
#include <gflags/gflags.h>
#include <gtest/gtest.h>

DEFINE_string(optimized_model, "", "");

namespace paddle {
namespace lite {

TEST(LightAPI, load) {
  LightPredictor predictor;
  predictor.Build(FLAGS_optimized_model);

  auto* input_tensor = predictor.GetInput(0);
  input_tensor->Resize(DDim(std::vector<int64_t>({100, 100})));
  auto* data = input_tensor->mutable_data<float>();
  for (int i = 0; i < 100 * 100; i++) {
    data[i] = i;
  }

  predictor.Run();

  const auto* output = predictor.GetOutput(0);
  const float* raw_output = output->data<float>();

  for (int i = 0; i < 10; i++) {
    LOG(INFO) << "out " << raw_output[i];
  }
}

}  // namespace lite
}  // namespace paddle

USE_LITE_OP(mul);
USE_LITE_OP(fc);
USE_LITE_OP(scale);
USE_LITE_OP(feed);
USE_LITE_OP(fetch);
USE_LITE_OP(io_copy);

USE_LITE_KERNEL(feed, kHost, kAny, kAny, def);
USE_LITE_KERNEL(fetch, kHost, kAny, kAny, def);

#ifdef LITE_WITH_X86
USE_LITE_KERNEL(relu, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(mul, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(fc, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(scale, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(square, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_sub, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_add, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(softmax, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(dropout, kX86, kFloat, kNCHW, def);
#endif
