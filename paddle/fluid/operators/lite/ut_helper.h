/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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

#include <gtest/gtest.h>
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/inference/analysis/helper.h"

namespace paddle {
namespace inference {
namespace lite {

/*
 * Get a random float value between [low, high]
 */
float random(float low, float high) {
  // static std::random_device rd;
  static std::mt19937 mt(100);
  std::uniform_real_distribution<double> dist(low, high);
  return dist(mt);
}

void RandomizeTensor(framework::LoDTensor* tensor,
                     const platform::Place& place) {
  auto dims = tensor->dims();
  size_t num_elements = analysis::AccuDims(dims, dims.size());
  PADDLE_ENFORCE_GT(num_elements, 0);

  platform::CPUPlace cpu_place;
  framework::LoDTensor temp_tensor;
  temp_tensor.Resize(dims);
  auto* temp_data = temp_tensor.mutable_data<float>(cpu_place);

  for (size_t i = 0; i < num_elements; i++) {
    *(temp_data + i) = random(0., 1.);
  }

  TensorCopySync(temp_tensor, place, tensor);
}
}  // namespace lite
}  // namespace inference
}  // namespace paddle
