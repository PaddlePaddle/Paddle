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

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {

void ZeroCopyTensor::Reshape(const std::vector<int> &shape) {
  PADDLE_ENFORCE(!name_.empty(),
                 "Need to SetName first, so that the corresponding tensor can "
                 "be retrieved.");
  PADDLE_ENFORCE(input_or_output_,
                 "Can't reshape the output tensor, it is readonly");
  PADDLE_ENFORCE(scope_);
  auto *scope = static_cast<framework::Scope *>(scope_);
  auto *var = scope->FindVar(name_);
  PADDLE_ENFORCE(var, "No tensor called [%s] in the runtime scope", name_);
  auto *tensor = var->GetMutable<framework::LoDTensor>();
  tensor->Resize(framework::make_ddim(shape));
}

template <typename T>
T *ZeroCopyTensor::mutable_data(PaddlePlace place) {
  auto *tensor = static_cast<framework::LoDTensor *>(FindTensor());
  switch (static_cast<int>(place)) {
    case static_cast<int>(PaddlePlace::kCPU): {
      return tensor->mutable_data<T>(platform::CPUPlace());
    }
    case static_cast<int>(PaddlePlace::kGPU): {
      return tensor->mutable_data<T>(platform::CUDAPlace());
    }
    default:
      PADDLE_THROW("Unsupported place: %d", static_cast<int>(place));
      break;
  }
  return nullptr;
}

template <typename T>
T *ZeroCopyTensor::data(PaddlePlace *place, int *size) const {
  auto *tensor = static_cast<framework::LoDTensor *>(FindTensor());
  auto *res = tensor->data<T>();

  if (platform::is_cpu_place(tensor->place())) {
    *place = PaddlePlace::kCPU;
  } else if (platform::is_gpu_place(tensor->place())) {
    *place = PaddlePlace::kGPU;
  } else {
    *place = PaddlePlace::kUNK;
  }

  *size = tensor->numel();
  return res;
}

template float *ZeroCopyTensor::data<float>(PaddlePlace *place,
                                            int *size) const;
template int64_t *ZeroCopyTensor::data<int64_t>(PaddlePlace *place,
                                                int *size) const;
template float *ZeroCopyTensor::mutable_data<float>(PaddlePlace place);
template int64_t *ZeroCopyTensor::mutable_data<int64_t>(PaddlePlace place);

void *ZeroCopyTensor::FindTensor() const {
  PADDLE_ENFORCE(!name_.empty(),
                 "Need to SetName first, so that the corresponding tensor can "
                 "be retrieved.");
  PADDLE_ENFORCE(scope_);
  auto *scope = static_cast<framework::Scope *>(scope_);
  auto *var = scope->FindVar(name_);
  PADDLE_ENFORCE(var, "No tensor called [%s] in the runtime scope", name_);
  auto *tensor = var->GetMutable<framework::LoDTensor>();
  return tensor;
}

std::vector<int64_t> ZeroCopyTensor::shape() const {
  auto *tensor = static_cast<framework::LoDTensor *>(FindTensor());
  PADDLE_ENFORCE(tensor, "not found tensor called %s in the scope", name_);
  return framework::vectorize(tensor->dims());
}

void ZeroCopyTensor::SetLoD(const std::vector<std::vector<size_t>> &x) {
  auto *tensor = static_cast<framework::LoDTensor *>(FindTensor());
  framework::LoD lod;
  for (auto &level : x) {
    lod.emplace_back(level);
  }
  tensor->set_lod(lod);
}

std::vector<std::vector<size_t>> ZeroCopyTensor::lod() const {
  std::vector<std::vector<size_t>> res;
  auto *tensor = static_cast<framework::LoDTensor *>(FindTensor());
  for (auto &level : tensor->lod()) {
    res.emplace_back(level);
  }
  return res;
}

}  // namespace paddle
