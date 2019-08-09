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
#include "paddle/fluid/lite/api/test_helper.h"
#include "paddle/fluid/lite/core/compatible_tensor.h"
#include "paddle/fluid/lite/core/op_registry.h"
// for googlenet

namespace paddle {
namespace lite {

TEST(Step_rnn, test_step_rnn_lite_x86) {
  lite::Predictor predictor;
  std::vector<Place> valid_places({Place{TARGET(kHost), PRECISION(kFloat)},
                                   Place{TARGET(kX86), PRECISION(kInt64)},
                                   Place{TARGET(kX86), PRECISION(kFloat)}});

  //  LOG(INFO)<<"FLAGS_eval_googlenet_dir:"<<FLAGS_test_lite_googlenet_dir;
  std::string model_dir = FLAGS_model_dir;
  std::vector<std::string> passes(
      {/*"lite_fc_fuse_pass",*/ "static_kernel_pick_pass",
       "variable_place_inference_pass", "type_target_cast_pass",
       "variable_place_inference_pass", "io_copy_kernel_pick_pass",
       "variable_place_inference_pass", "runtime_context_assign_pass"});
  predictor.Build(model_dir, Place{TARGET(kX86), PRECISION(kFloat)},
                  valid_places, passes);

  std::vector<std::string> target_names = {
      "item_type_id",   "mthid_id",         "source_id_id",
      "layout_id",      "mark_id",          "category_id",
      "subcategory_id", "score_segment_id", "item_attention_id",
      "queue_num_id",   "micro_video_id",   "vertical_type_id"};
  for (size_t i = 0; i < target_names.size(); i++) {
    auto* input_tensor = predictor.GetInput(i);
    int size = 0;
    if (i == 6 || i == 8) {
      input_tensor->Resize(
          lite::DDim(std::vector<lite::DDim::value_type>({5, 1})));
      input_tensor->raw_tensor().set_lod({{0, 5}});
      size = 5;
    } else {
      input_tensor->Resize(
          lite::DDim(std::vector<lite::DDim::value_type>({1, 1})));
      input_tensor->raw_tensor().set_lod({{0, 1}});
      size = 1;
    }
    auto* data = input_tensor->mutable_data<int64_t>();
    for (int i = 0; i < size; i++) data[i] = 1;
  }

  for (int i = 0; i < FLAGS_warmup; ++i) {
    predictor.Run();
  }

  auto start = GetCurrentUS();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    predictor.Run();
  }

  //  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << ", warmup: " << FLAGS_warmup << ", repeats: " << FLAGS_repeats
            << ", spend " << (GetCurrentUS() - start) / FLAGS_repeats / 1000.0
            << " ms in average.";

  std::vector<std::vector<float>> results;
  // i = 1
  results.emplace_back(std::vector<float>({0.471981, 0.528019}));
  auto* out = predictor.GetOutput(0);
  ASSERT_EQ(out->dims().size(), 2);
  ASSERT_EQ(out->dims()[0], 1);
  ASSERT_EQ(out->dims()[1], 2);

  for (int i = 0; i < results.size(); ++i) {
    for (int j = 0; j < results[i].size(); ++j) {
      LOG(INFO) << "output[" << i << "]"
                << "[" << j
                << "]:" << out->data<float>()[j + (out->dims()[1] * i)];
      // EXPECT_NEAR(out->data<float>()[j + (out->dims()[1] * i)],
      // results[i][j],
      //            1e-6);
    }
  }
}

}  // namespace lite
}  // namespace paddle
