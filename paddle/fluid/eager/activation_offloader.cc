// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/activation_offloader.h"
#include "glog/logging.h"
#include "paddle/common/flags.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/memory/stats.h"

COMMON_DECLARE_bool(offload_inplace_tensor);
COMMON_DECLARE_bool(print_offload_info);

namespace egr {

template <typename T>
static size_t GetMemorySize(const T &tensor_ptr) {
  if (tensor_ptr == nullptr) return 0;
  const auto &holder = tensor_ptr->Holder();
  return holder != nullptr ? holder->size() : 0;
}

static std::shared_ptr<phi::DenseTensor> GetDenseTensorImpl(
    const paddle::Tensor &tensor, size_t *memory_size = nullptr) {
  auto dense_tensor =
      std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
  size_t size = GetMemorySize(dense_tensor);
  if (memory_size) *memory_size = size;
  return size == 0 ? nullptr : dense_tensor;
}

static size_t GetAllocatedMemory(phi::GPUPlace place) {
  return paddle::memory::DeviceMemoryStatCurrentValue("Allocated",
                                                      place.device);
}

template <typename T>
static std::string GetTensorMetaString(const T &tensor_ptr) {
  std::stringstream ss;
  if (tensor_ptr == nullptr) {
    ss << "tensor with null";
  } else if (!tensor_ptr->initialized()) {
    ss << "tensor with shape: [" << tensor_ptr->dims()
       << "] , dtype: [NOT_INITIALIZED]"
       << " , place: [NOT_INITIALIZED]"
       << " , memory_size: 0"
       << " , data_ptr: null";
  } else {
    ss << "tensor with shape: [" << tensor_ptr->dims()
       << "] , dtype: " << tensor_ptr->type()
       << " , place: " << tensor_ptr->place()
       << " , memory_size: " << GetMemorySize(tensor_ptr)
       << " , data_ptr: " << tensor_ptr->data() << " , inplace_version: "
       << tensor_ptr->InplaceVersionCounter().CurrentVersion();
  }
  return ss.str();
}

ReloadFunctor::ReloadFunctor(std::weak_ptr<phi::DenseTensor> tensor,
                             ActivationOffloaderWithPlace *offloader)
    : tensor_(tensor), offloader_(offloader) {}

void ReloadFunctor::Reload() {
  offloader_->Remove(tensor_);
  auto dense_tensor = tensor_.lock();
  size_t memory_size = GetMemorySize(dense_tensor);
  if (memory_size == 0) return;
  auto dst_place = offloader_->Place();
  if (dense_tensor->place() != dst_place) {
    if (FLAGS_print_offload_info) {
      LOG(INFO) << "Reload " << dense_tensor->place() << " -> " << dst_place
                << " , " << GetTensorMetaString(dense_tensor);
    }
    PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
    auto dst_holder = phi::memory_utils::AllocShared(dst_place, memory_size);
    phi::memory_utils::Copy(dst_holder->place(),
                            dst_holder->ptr(),
                            dense_tensor->place(),
                            dense_tensor->data(),
                            memory_size,
                            nullptr);
    dense_tensor->set_offset(0);
    dense_tensor->ResetHolder(std::move(dst_holder));
  }
}

ActivationOffloaderWithPlace::ActivationOffloaderWithPlace(phi::GPUPlace place)
    : place_(place) {}

void ActivationOffloaderWithPlace::SetSkipTensors(
    const std::vector<paddle::Tensor> &tensors) {
  skip_tensors_.clear();
  for (auto &t : tensors) {
    auto dense_tensor = GetDenseTensorImpl(t);
    if (dense_tensor != nullptr && dense_tensor->place() == place_) {
      PADDLE_ENFORCE_EQ(dense_tensor->meta().is_contiguous(),
                        true,
                        common::errors::InvalidArgument(
                            "Only contiguous tensor is supported."));
      VLOG(10) << "SetSkip " << GetTensorMetaString(dense_tensor);
      skip_tensors_.insert(std::move(dense_tensor));
    }
  }
  activations_.clear();
}

paddle::optional<ReloadFunctor> ActivationOffloaderWithPlace::Add(
    const paddle::Tensor &activation) {
  size_t memory_size;
  auto dense_tensor = GetDenseTensorImpl(activation, &memory_size);
  if (memory_size == 0) return paddle::none;
  if (skip_tensors_.count(dense_tensor) > 0) return paddle::none;
  if (dense_tensor->place() != place_) return paddle::none;
  if (!dense_tensor->meta().is_contiguous()) {
    VLOG(7) << "Offload skip non-contiguous tensor "
            << GetTensorMetaString(dense_tensor)
            << " allocated: " << GetAllocatedMemory(place_);
    return paddle::none;
  }
  if (dense_tensor->offset() != 0) {
    VLOG(7) << "Offload skip non-zero offset tensor "
            << GetTensorMetaString(dense_tensor)
            << " allocated: " << GetAllocatedMemory(place_);
    return paddle::none;
  }
  if (!FLAGS_offload_inplace_tensor &&
      dense_tensor->InplaceVersionCounter().CurrentVersion() > 0) {
    VLOG(7) << "Offload skip inplace tensor "
            << GetTensorMetaString(dense_tensor)
            << " allocated: " << GetAllocatedMemory(place_);
    return paddle::none;
  }

  VLOG(10) << "Add " << GetTensorMetaString(dense_tensor)
           << " allocated: " << GetAllocatedMemory(place_);
  ++activations_[dense_tensor];
  return ReloadFunctor(dense_tensor, this);
}

size_t ActivationOffloaderWithPlace::Offload(size_t size) {
  if (size == 0) return 0;

  Shrink();

  std::map<std::pair<size_t, const void *>, std::weak_ptr<phi::DenseTensor>>
      activation_map;
  for (auto &pair : activations_) {
    auto dense_tensor = pair.first.lock();
    auto ref_cnt = dense_tensor.use_count() - 1;
    auto cnt = static_cast<decltype(ref_cnt)>(pair.second);
    PADDLE_ENFORCE_GE(
        cnt,
        1,
        common::errors::InvalidArgument("Invalid reference count %d", cnt));
    if (ref_cnt > cnt) {
      VLOG(7) << "Cannot offload tensor because its reference is not unique: "
              << GetTensorMetaString(dense_tensor)
              << " , allocated: " << GetAllocatedMemory(place_)
              << " , desired_ref_cnt: " << cnt
              << " , actual_ref_cnt: " << ref_cnt;
      continue;
    } else if (cnt > 1) {
      VLOG(7) << "Tensor with ref_cnt " << cnt << ": "
              << GetTensorMetaString(dense_tensor)
              << " , allocated: " << GetAllocatedMemory(place_)
              << " , desired_ref_cnt: " << cnt
              << " , actual_ref_cnt: " << ref_cnt;
    }
    size_t memory_size = GetMemorySize(dense_tensor);
    if (memory_size > 0) {
      activation_map.insert(
          {std::make_pair(memory_size, dense_tensor->data()), pair.first});
    }
  }

  size_t offload_cnt = 0;

  auto offload_tensor = [this, &activation_map, &offload_cnt, &size](
                            phi::DenseTensor *tensor,
                            size_t memory_size) -> size_t {
    if (memory_size == 0) return 0;
    if (FLAGS_print_offload_info) {
      LOG(INFO) << "Start to offload " << GetTensorMetaString(tensor)
                << " , allocated: " << GetAllocatedMemory(place_)
                << " , activation_number: " << activation_map.size()
                << " , desired_size: " << size;
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
    auto dst_holder =
        phi::memory_utils::AllocShared(phi::GPUPinnedPlace(), memory_size);
    phi::memory_utils::Copy(dst_holder->place(),
                            dst_holder->ptr(),
                            tensor->place(),
                            tensor->data(),
                            memory_size,
                            nullptr);
    tensor->set_offset(0);
    tensor->ResetHolder(std::move(dst_holder));
    auto end_time = std::chrono::high_resolution_clock::now();
    double time_cost = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           end_time - start_time)
                           .count() /
                       1e9;
    ++offload_cnt;
    if (FLAGS_print_offload_info) {
      LOG(INFO) << "End to offload " << GetTensorMetaString(tensor)
                << " , time_cost: " << time_cost
                << " , allocated: " << GetAllocatedMemory(place_)
                << " , activation_number: "
                << activation_map.size() - offload_cnt
                << " , desired_size: " << size;
    }
    return memory_size;
  };

  size_t offloaded_memory_size = 0;
  auto iter = activation_map.lower_bound(
      std::pair<size_t, const void *>(size, nullptr));
  if (iter != activation_map.end()) {
    offloaded_memory_size +=
        offload_tensor(iter->second.lock().get(), iter->first.first);
    activations_.erase(iter->second);
  } else {
    for (auto iter = activation_map.rbegin(); iter != activation_map.rend();
         ++iter) {
      offloaded_memory_size +=
          offload_tensor(iter->second.lock().get(), iter->first.first);
      activations_.erase(iter->second);
      if (offloaded_memory_size >= size) {
        break;
      }
    }
  }
  return offloaded_memory_size;
}

void ActivationOffloaderWithPlace::Remove(
    const std::weak_ptr<phi::DenseTensor> &tensor) {
  auto iter = activations_.find(tensor);
  if (iter == activations_.end()) return;
  --(iter->second);
  if (iter->second == 0) {
    activations_.erase(iter);
    VLOG(10) << "Remove " << GetTensorMetaString(tensor.lock());
  }
}

void ActivationOffloaderWithPlace::Shrink() {
  for (auto iter = activations_.begin(); iter != activations_.end();) {
    if (iter->first.expired()) {
      activations_.erase(iter++);
    } else {
      ++iter;
    }
  }
}

size_t ActivationOffloaderWithPlace::CachedSize() const {
  size_t size = 0;
  for (auto &t : activations_) {
    if (auto shared_t = t.first.lock()) {
      const auto &holder = shared_t->Holder();
      if (holder != nullptr) {
        size += holder->size();
      }
    }
  }
  return size;
}

void ActivationOffloader::SetSkipTensors(
    const std::vector<paddle::Tensor> &tensors) {
  std::map<ActivationOffloaderWithPlace *, std::vector<paddle::Tensor>>
      offload_map;
  for (auto &t : tensors) {
    auto dense_tensor = GetDenseTensorImpl(t);
    if (dense_tensor != nullptr && dense_tensor->initialized()) {
      auto *offloader = GetOrCreateOffloader(dense_tensor->place());
      if (offloader != nullptr) {
        offload_map[offloader].push_back(t);
      }
    }
  }

  for (auto &pair : offloaders_) {
    auto *offloader = pair.second.get();
    offloader->SetSkipTensors(offload_map[offloader]);
  }
}

paddle::optional<ReloadFunctor> ActivationOffloader::Add(
    const paddle::Tensor &activation) {
  auto dense_tensor = GetDenseTensorImpl(activation);
  if (dense_tensor != nullptr) {
    auto *offloader = GetOrCreateOffloader(dense_tensor->place());
    if (offloader != nullptr) {
      return offloader->Add(activation);
    }
  }
  return paddle::none;
}

ActivationOffloaderWithPlace *ActivationOffloader::GetOrCreateOffloader(
    phi::Place place) {
  if (!phi::is_gpu_place(place)) return nullptr;
  auto gpu_place = static_cast<phi::GPUPlace>(place);
  auto &offloader = offloaders_[gpu_place];
  if (offloader == nullptr) {
    offloader.reset(new ActivationOffloaderWithPlace(gpu_place));
  }
  return offloader.get();
}

size_t ActivationOffloader::Offload(phi::Place place, size_t size) {
  auto *offloader = GetOrCreateOffloader(place);
  return offloader != nullptr ? offloader->Offload(size) : 0;
}

size_t ActivationOffloader::CachedSize() const {
  size_t size = 0;
  for (auto &pair : offloaders_) {
    size += pair.second->CachedSize();
  }
  return size;
}

ActivationOffloader *ActivationOffloader::Instance() {
  static ActivationOffloader offloader;
  return &offloader;
}

}  // namespace egr
