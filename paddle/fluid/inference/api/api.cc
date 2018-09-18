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

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle_inference_api.h"

namespace paddle {

int PaddleDtypeSize(PaddleDType dtype) {
  switch (dtype) {
    case PaddleDType::FLOAT32:
      return sizeof(float);
    case PaddleDType::INT64:
      return sizeof(int64_t);
    default:
      assert(false);
      return -1;
  }
}

PaddleBuf::PaddleBuf(PaddleBuf &&other)
    : data_(other.data_),
      length_(other.length_),
      memory_owned_(other.memory_owned_) {
  other.memory_owned_ = false;
  other.data_ = nullptr;
  other.length_ = 0;
}

PaddleBuf::PaddleBuf(const PaddleBuf &other) { *this = other; }

PaddleBuf &PaddleBuf::operator=(const PaddleBuf &other) {
  if (!other.memory_owned_) {
    data_ = other.data_;
    length_ = other.length_;
    memory_owned_ = other.memory_owned_;
  } else {
    Resize(other.length());
    memcpy(data_, other.data(), other.length());
    length_ = other.length();
    memory_owned_ = true;
  }
  return *this;
}

PaddleBuf &PaddleBuf::operator=(PaddleBuf &&other) {
  // only the buffer with external memory can be copied
  data_ = other.data_;
  length_ = other.length_;
  memory_owned_ = other.memory_owned_;
  other.data_ = nullptr;
  other.length_ = 0;
  other.memory_owned_ = false;
  return *this;
}

void PaddleBuf::Resize(size_t length) {
  // Only the owned memory can be reset, the external memory can't be changed.
  if (length_ >= length) return;
  if (memory_owned_) {
    Free();
    data_ = malloc(length);
    length_ = length;
    memory_owned_ = true;
  } else {
    PADDLE_THROW("The memory is allocated externally, can not Resized");
  }
}

void PaddleBuf::Reset(void *data, size_t length) {
  Free();
  memory_owned_ = false;
  data_ = data;
  length_ = length;
}

void PaddleBuf::Free() {
  if (memory_owned_ && data_) {
    PADDLE_ENFORCE_GT(length_, 0);
    free(static_cast<char *>(data_));
    data_ = nullptr;
    length_ = 0;
  }
}

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
    case static_cast<int>(kCPU): {
      return tensor->mutable_data<T>(platform::CPUPlace());
    }
    case static_cast<int>(kGPU): {
      return tensor->mutable_data<T>(platform::CUDAPlace());
    }
    default:
      PADDLE_THROW("Unsupported place: %d", static_cast<int>(place));
      break;
  }
  return nullptr;
}

template <typename T>
T *ZeroCopyTensor::data(PaddlePlace *place, int *size) {
  auto *tensor = static_cast<framework::LoDTensor *>(FindTensor());
  auto *res = tensor->data<T>();

  if (platform::is_cpu_place(tensor->place())) {
    *place = kCPU;
  } else if (platform::is_gpu_place(tensor->place())) {
    *place = kGPU;
  } else {
    *place = kUnknown;
  }

  *size = tensor->numel();
  return res;
}

template float *ZeroCopyTensor::data<float>(PaddlePlace *place, int *size);
template int64_t *ZeroCopyTensor::data<int64_t>(PaddlePlace *place, int *size);
template float *ZeroCopyTensor::mutable_data(PaddlePlace place);
template int64_t *ZeroCopyTensor::mutable_data(PaddlePlace place);

void *ZeroCopyTensor::FindTensor() {
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

std::vector<int64_t> ZeroCopyTensor::shape() {
  auto *tensor = static_cast<framework::LoDTensor *>(FindTensor());
  return framework::vectorize(tensor->dims());
}

}  // namespace paddle
