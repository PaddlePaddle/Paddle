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

#include <ios>
#include <fstream>
#include <gtest/gtest.h>

#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"

#include "paddle/fluid/inference/lite/engine.h"
#include "paddle/fluid/inference/utils/singleton.h"

namespace paddle {
namespace lite {

namespace {

std::string read_file(const std::string &file) {
  std::ifstream ifs(file.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
  std::ifstream::pos_type file_size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  std::vector<char> bytes(file_size);
  ifs.read(bytes.data(), file_size);
  return std::string(bytes.data(), file_size);
}

} // namespace


TEST(EngineManager, Create) {
  const std::string unique_key("engine_0");
  const std::string model_dir = "/shixiaowei02/models/tmp/__model__";

  inference::lite::EngineConfig config;
  config.model = read_file(model_dir);
  config.param = "";
  config.prefer_place = {TARGET(kCUDA), PRECISION(kFloat)};
  config.valid_places = {
    paddle::lite::Place({TARGET(kHost), PRECISION(kFloat)}),
#ifdef PADDLE_WITH_CUDA
    paddle::lite::Place({TARGET(kCUDA), PRECISION(kFloat)}),
#endif
  };

  inference::Singleton<inference::lite::EngineManager>::Global()
    .Create(unique_key, config);
  /*
  paddle::lite::Predictor* engine = inference::Singleton<inference::lite::EngineManager>::Global()
          .Get(Attr<std::string>(unique_key));
  */
}

}  // namespace lite
}  // namespace paddle
