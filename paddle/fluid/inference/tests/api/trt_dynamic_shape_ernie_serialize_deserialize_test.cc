/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <dirent.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <unistd.h>
#include "gflags/gflags.h"

#include "paddle/fluid/inference/tests/api/trt_dynamic_shape_ernie_serialize_deserialize_test.h"

namespace paddle {
namespace inference {

TEST(AnalysisPredictor, no_fp16) {
  std::vector<float> result = {0.597841, 0.219972, 0.182187};
  trt_ernie(false, result);
}

}  // namespace inference
}  // namespace paddle
