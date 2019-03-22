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
#include "paddle/fluid/memory/memcpy.h"
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

#define EAGER_GET_TENSOR    \
  if (!tensor_) {           \
    tensor_ = FindTensor(); \
  }                         \
  auto *tensor = static_cast<framework::LoDTensor *>(tensor_);

template <typename T>
T *ZeroCopyTensor::mutable_data(PaddlePlace place) {
  EAGER_GET_TENSOR;
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
  EAGER_GET_TENSOR;
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

PaddleDType ZeroCopyTensor::type() {
  EAGER_GET_TENSOR;
  auto type = tensor->type();
  if (type == framework::proto::VarType::FP32) {
    return PaddleDType::FLOAT32;
  } else if (type == framework::proto::VarType::INT64) {
    return PaddleDType::INT64;
  } else if (type == framework::proto::VarType::INT32) {
    return PaddleDType::INT32;
  } else {
    LOG(ERROR) << "unknown type, only support float32 and int64 now.";
  }
  return PaddleDType::FLOAT32;
}

template <typename T>
void ZeroCopyTensor::copy_from_cpu(const T *data) {
  EAGER_GET_TENSOR;
  PADDLE_ENFORCE_GE(
      tensor->numel(), 0,
      "You should call ZeroCopyTensor::Reshape(const std::vector<int> &shape)"
      "function before copy data from cpu.");
  size_t ele_size = tensor->numel() * sizeof(T);

  if (place_ == PaddlePlace::kCPU) {
    auto *t_data = tensor->mutable_data<T>(platform::CPUPlace());
    std::memcpy(static_cast<void *>(t_data), data, ele_size);
  } else {
#ifdef PADDLE_WITH_CUDA
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    platform::CUDAPlace gpu_place(device_);
    auto *t_data = tensor->mutable_data<T>(gpu_place);
    auto *dev_ctx =
        static_cast<const platform::CUDADeviceContext *>(pool.Get(gpu_place));

    memory::Copy(gpu_place, static_cast<void *>(t_data), platform::CPUPlace(),
                 data, ele_size, dev_ctx->stream());
#else
    PADDLE_THROW("Not compile with CUDA, should not reach here.");
#endif
  }
}

template <typename T>
void ZeroCopyTensor::copy_to_cpu(T *data) {
  EAGER_GET_TENSOR;
  auto ele_num = tensor->numel();
  auto *t_data = tensor->data<T>();
  auto t_place = tensor->place();

  if (platform::is_cpu_place(t_place)) {
    std::memcpy(static_cast<void *>(data), t_data, ele_num * sizeof(T));
  } else {
#ifdef PADDLE_WITH_CUDA
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto gpu_place = boost::get<platform::CUDAPlace>(t_place);
    auto *dev_ctx =
        static_cast<const platform::CUDADeviceContext *>(pool.Get(gpu_place));
    memory::Copy(platform::CPUPlace(), static_cast<void *>(data), gpu_place,
                 t_data, ele_num * sizeof(T), dev_ctx->stream());
    cudaDeviceSynchronize();
#else
    PADDLE_THROW("Not compile with CUDA, should not reach here.");
#endif
  }
}
template void ZeroCopyTensor::copy_from_cpu<float>(const float *data);
template void ZeroCopyTensor::copy_from_cpu<int64_t>(const int64_t *data);
template void ZeroCopyTensor::copy_from_cpu<int32_t>(const int32_t *data);
template void ZeroCopyTensor::copy_to_cpu<float>(float *data);
template void ZeroCopyTensor::copy_to_cpu<int64_t>(int64_t *data);
template void ZeroCopyTensor::copy_to_cpu<int32_t>(int32_t *data);

template float *ZeroCopyTensor::data<float>(PaddlePlace *place,
                                            int *size) const;
template int64_t *ZeroCopyTensor::data<int64_t>(PaddlePlace *place,
                                                int *size) const;
template int32_t *ZeroCopyTensor::data<int32_t>(PaddlePlace *place,
                                                int *size) const;
template float *ZeroCopyTensor::mutable_data<float>(PaddlePlace place);
template int64_t *ZeroCopyTensor::mutable_data<int64_t>(PaddlePlace place);
template int32_t *ZeroCopyTensor::mutable_data<int32_t>(PaddlePlace place);

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

std::vector<int> ZeroCopyTensor::shape() const {
  EAGER_GET_TENSOR;
  PADDLE_ENFORCE(tensor_, "not found tensor called %s in the scope", name_);
  return framework::vectorize2int(tensor->dims());
}

void ZeroCopyTensor::SetLoD(const std::vector<std::vector<size_t>> &x) {
  EAGER_GET_TENSOR;
  framework::LoD lod;
  for (auto &level : x) {
    lod.emplace_back(level);
  }
  tensor->set_lod(lod);
}

std::vector<std::vector<size_t>> ZeroCopyTensor::lod() const {
  EAGER_GET_TENSOR;
  std::vector<std::vector<size_t>> res;
  for (auto &level : tensor->lod()) {
    res.emplace_back(level);
  }
  return res;
}

}  // namespace paddle
