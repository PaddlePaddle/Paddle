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

#include "paddle/fluid/inference/api/paddle_api.h"
#include "paddle/fluid/inference/api/paddle_infer_declare.h"

namespace paddle_infer {

void Tensor::Reshape(const std::vector<int> &shape) {}

template <typename T>
T *Tensor::mutable_data(PlaceType place) {
  return nullptr;
}

template <typename T>
T *Tensor::data(PlaceType *place, int *size) const {
  return nullptr;
}

template PD_INFER_DECL float *Tensor::data<float>(PlaceType *place,
                                                  int *size) const;
template PD_INFER_DECL int64_t *Tensor::data<int64_t>(PlaceType *place,
                                                      int *size) const;
template float *Tensor::mutable_data(PlaceType place);
template int64_t *Tensor::mutable_data(PlaceType place);

template <typename T>
void *Tensor::FindTensor() const {
  return nullptr;
}

std::vector<int> Tensor::shape() const { return {}; }

void Tensor::SetLoD(const std::vector<std::vector<size_t>> &x) {}

std::vector<std::vector<size_t>> Tensor::lod() const {
  return std::vector<std::vector<size_t>>();
}

}  // namespace paddle_infer
