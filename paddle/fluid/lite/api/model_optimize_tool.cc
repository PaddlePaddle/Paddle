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
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "paddle/fluid/lite/api/paddle_api.h"
#include "paddle/fluid/lite/api/paddle_use_kernels.h"
#include "paddle/fluid/lite/api/paddle_use_ops.h"
#include "paddle/fluid/lite/api/paddle_use_passes.h"
#include "paddle/fluid/lite/utils/string.h"

DEFINE_string(model_dir, "", "path of the model");
DEFINE_string(optimize_out, "", "path of the output optimized model");
DEFINE_string(valid_targets, "ARM",
              "The targets this model optimized for, should be one of (arm, "
              "opencl, x86), splitted by space");
DEFINE_bool(int8_mode, false, "Support Int8 quantitative mode");

namespace paddle {
namespace lite_api {

void Main() {
  lite_api::CxxConfig config;
  config.set_model_dir(FLAGS_model_dir);

  std::vector<Place> valid_places;
  auto target_reprs = lite::Split(FLAGS_valid_targets, " ");
  for (auto& target_repr : target_reprs) {
    if (target_repr == "arm") {
      valid_places.emplace_back(TARGET(kARM));
    } else if (target_repr == "opencl") {
      valid_places.emplace_back(TARGET(kOpenCL));
    } else if (target_repr == "x86") {
      valid_places.emplace_back(TARGET(kX86));
    } else {
      LOG(FATAL) << lite::string_format(
          "Wrong target '%s' found, please check the command flag "
          "'valid_targets'",
          target_repr.c_str());
    }
  }

  CHECK(!valid_places.empty())
      << "At least one target should be set, should set the "
         "command argument 'valid_targets'";
  if (FLAGS_int8_mode) {
    LOG(WARNING) << "Int8 mode is only support by ARM target";
    valid_places.push_back(Place{TARGET(kARM), PRECISION(kInt8)});
    config.set_preferred_place(Place{TARGET(kARM), PRECISION(kInt8)});
  }
  config.set_valid_places(valid_places);

  auto predictor = lite_api::CreatePaddlePredictor(config);
  predictor->SaveOptimizedModel(FLAGS_optimize_out);
}

}  // namespace lite_api
}  // namespace paddle

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, false);
  paddle::lite_api::Main();
  return 0;
}
