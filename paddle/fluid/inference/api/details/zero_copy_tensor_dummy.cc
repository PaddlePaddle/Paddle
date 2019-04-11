// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/api/paddle_inference_api.h"

namespace paddle {

void ZeroCopyTensor::Reshape(const std::vector<int> &shape) {}

template <typename T>
T *ZeroCopyTensor::mutable_data(PaddlePlace place) {
  return nullptr;
}

template <typename T>
T *ZeroCopyTensor::data(PaddlePlace *place, int *size) const {
  return nullptr;
}

template float *ZeroCopyTensor::data<float>(PaddlePlace *place,
                                            int *size) const;
template int64_t *ZeroCopyTensor::data<int64_t>(PaddlePlace *place,
                                                int *size) const;
template float *ZeroCopyTensor::mutable_data(PaddlePlace place);
template int64_t *ZeroCopyTensor::mutable_data(PaddlePlace place);

void *ZeroCopyTensor::FindTensor() const { return nullptr; }

std::vector<int> ZeroCopyTensor::shape() const { return {}; }

void ZeroCopyTensor::SetLoD(const std::vector<std::vector<size_t>> &x) {}

std::vector<std::vector<size_t>> ZeroCopyTensor::lod() const {
  return std::vector<std::vector<size_t>>();
}

}  // namespace paddle
