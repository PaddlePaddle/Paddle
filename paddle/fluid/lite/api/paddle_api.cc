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

#include "paddle/fluid/lite/api/paddle_api.h"
#include "paddle/fluid/lite/api/cxx_api.h"
#include "paddle/fluid/lite/api/light_api.h"

namespace paddle {
namespace lite_api {

Tensor::Tensor(void *raw) : raw_tensor_(raw) {}

// TODO(Superjomn) refine this by using another `const void* const_raw`;
Tensor::Tensor(const void *raw) { raw_tensor_ = const_cast<void *>(raw); }

lite::Tensor *tensor(void *x) { return static_cast<lite::Tensor *>(x); }
const lite::Tensor *ctensor(void *x) {
  return static_cast<const lite::Tensor *>(x);
}

void Tensor::Resize(const shape_t &shape) {
  tensor(raw_tensor_)->Resize(shape);
}

template <>
const float *Tensor::data() const {
  return ctensor(raw_tensor_)->data<float>();
}
template <>
const int8_t *Tensor::data() const {
  return ctensor(raw_tensor_)->data<int8_t>();
}

template <>
float *Tensor::mutable_data() const {
  return tensor(raw_tensor_)->mutable_data<float>();
}
template <>
int8_t *Tensor::mutable_data() const {
  return tensor(raw_tensor_)->mutable_data<int8_t>();
}

shape_t Tensor::shape() const {
  return ctensor(raw_tensor_)->dims().Vectorize();
}

void PaddlePredictor::SaveOptimizedModel(const std::string &model_dir) {
  LOG(ERROR)
      << "The SaveOptimizedModel API is only supported by CxxConfig predictor.";
}

template <typename ConfigT>
std::shared_ptr<PaddlePredictor> CreatePaddlePredictor(const ConfigT &) {
  return std::shared_ptr<PaddlePredictor>();
}

}  // namespace lite_api
}  // namespace paddle
