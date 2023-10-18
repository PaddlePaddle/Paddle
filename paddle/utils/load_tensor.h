/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <cstdint>

#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/version.h"
#include "paddle/fluid/framework/lod_tensor.h"

namespace paddle {

void LoadTensor(const std::string& file_path, phi::DenseTensor* out) {
  std::ifstream fin(file_path, std::ios::binary);
  PADDLE_ENFORCE_EQ(static_cast<bool>(fin),
                    true,
                    phi::errors::Unavailable(
                        "Load operator fail to open file %s, please check "
                        "whether the model file is complete or damaged.",
                        file_path));
  PADDLE_ENFORCE_NOT_NULL(out,
                          phi::errors::InvalidArgument(
                              "The variable to be loaded cannot be found."));

  framework::DeserializeFromStream(fin, out);
}

}  // namespace paddle
