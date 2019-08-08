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
#include "paddle/fluid/lite/api/lite_api_test_helper.h"
#include "paddle/fluid/lite/api/paddle_use_kernels.h"
#include "paddle/fluid/lite/api/paddle_use_ops.h"
#include "paddle/fluid/lite/api/paddle_use_passes.h"
#include "paddle/fluid/lite/core/compatible_tensor.h"
#include "paddle/fluid/lite/core/op_registry.h"

// for googlenet
DEFINE_string(model_dir, "", "");

namespace paddle {
namespace lite {
#ifdef LITE_WITH_X86
TEST(CXXApi, test_lite_googlenet) {
  lite::Predictor predictor;
  std::vector<Place> valid_places({Place{TARGET(kHost), PRECISION(kFloat)},
                                   Place{TARGET(kX86), PRECISION(kFloat)}});

  //  LOG(INFO)<<"FLAGS_eval_googlenet_dir:"<<FLAGS_test_lite_googlenet_dir;
  std::string model_dir = FLAGS_model_dir;
  predictor.Build(model_dir, Place{TARGET(kX86), PRECISION(kFloat)},
                  valid_places);

  auto* input_tensor = predictor.GetInput(0);
  input_tensor->Resize(DDim(std::vector<DDim::value_type>({1, 3, 224, 224})));
  auto* data = input_tensor->mutable_data<float>();
  for (int i = 0; i < input_tensor->dims().production(); i++) {
    data[i] = 1;
  }
  predictor.Run();

  auto* out = predictor.GetOutput(0);
  std::vector<float> results(
      {0.00034298553, 0.0008200012, 0.0005046297, 0.000839279,
       0.00052616704, 0.0003447803, 0.0010877076, 0.00081762316,
       0.0003941339,  0.0011430943, 0.0008892841, 0.00080191303,
       0.0004442384,  0.000658702,  0.0026721435, 0.0013686896,
       0.0005618166,  0.0006556497, 0.0006984528, 0.0014619455});
  for (size_t i = 0; i < results.size(); ++i) {
    EXPECT_NEAR(out->data<float>()[i * 51], results[i], 1e-5);
  }
  ASSERT_EQ(out->dims().size(), 2);
  ASSERT_EQ(out->dims()[0], 1);
  ASSERT_EQ(out->dims()[1], 1000);
}
#endif
}  // namespace lite
}  // namespace paddle
