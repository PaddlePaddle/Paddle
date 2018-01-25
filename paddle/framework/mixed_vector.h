/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include <initializer_list>
#include <vector>

#include "paddle/memory/memcpy.h"
#include "paddle/memory/memory.h"
#include "paddle/platform/device_context.h"
#include "paddle/platform/enforce.h"
#include "paddle/platform/place.h"

namespace paddle {
namespace framework {

/**
 * @brief Vector support both cpu and gpu.
 * host vector lifetime is same with Vector
 * device vector is lazily malloc and modified.
 */

template <typename T>
class Vector : public std::vector<T> {
 public:
  /* NOTE(dzhwinter):
   * Data always store and modified on Host.
   * If the data is modified when use cuda_data interface,
   * You need to call the CopyFromCUDA explicitly to synchronize data.
   *
   */
  enum class kDataPosition {
    kDataOnHost = 0,
    kDataOnDevice = 1,
  };

 public:
  using std::vector<T>::vector;

  Vector() {}
  Vector(const std::vector<T> &v) : std::vector<T>(v) {}  // NOLINT

  virtual ~Vector() {
#ifdef PADDLE_WITH_CUDA
    if (cuda_ptr_ != nullptr) {
      memory::Free<platform::CUDAPlace>(place_, static_cast<void *>(cuda_ptr_));
    }
#endif
  }

  T *cuda_data() {
    if (position_ == kDataPosition::kDataOnHost) {
      CopyToCUDA();
    }
    PADDLE_ENFORCE_NOT_NULL(
        cuda_ptr_, "No data or Insufficient CUDA memory to allocation");
    return static_cast<T *>(cuda_ptr_);
  }

  T *data() {
    if (position_ == kDataPosition::kDataOnDevice) {
      CopyFromCUDA();
    }
    return std::vector<T>::data();
  }

  const T *data() const { return std::vector<T>::data(); }

  void CopyToCUDA();

  void CopyFromCUDA();

  void CopyToPeer(platform::Place);

 private:
  void *cuda_ptr_ = nullptr;
  size_t cuda_size_ = 0;
  kDataPosition position_ = kDataPosition::kDataOnHost;
  platform::CUDAPlace place_;
};

template <typename T>
void Vector<T>::CopyToCUDA() {
#ifdef PADDLE_WITH_CUDA
  if (cuda_ptr_ == nullptr) {
    cuda_ptr_ =
        memory::Alloc<platform::CUDAPlace>(place_, this->size() * sizeof(T));
  }
  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto *cuda_ctx = pool.GetByPlace(place_);

  memory::Copy(place_, static_cast<void *>(cuda_ptr_), platform::CPUPlace(),
               static_cast<const void *>(this->data()),
               this->size() * sizeof(T), cuda_ctx->stream());
  cuda_ctx->Wait();

  cuda_size_ = this->size();
  position_ = kDataPosition::kDataOnDevice;
#endif
}

template <typename T>
void Vector<T>::CopyFromCUDA() {
#ifdef PADDLE_WITH_CUDA
  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto *cuda_ctx = pool.GetByPlace(place_);
  if (cuda_ptr_ == nullptr) {
    LOG(WARNING) << "No uncommited cuda data.";
    return;
  }
  this->resize(cuda_size_);
  memory::Copy(platform::CPUPlace(), static_cast<void *>(this->data()), place_,
               static_cast<const void *>(cuda_ptr_), this->size() * sizeof(T),
               cuda_ctx->stream());
  cuda_ctx->Wait();

  position_ = kDataPosition::kDataOnHost;
#endif
}

template <typename T>
void Vector<T>::CopyToPeer(platform::Place peer_place) {
  if (platform::is_cpu_place(peer_place)) {
    return;
  }
#ifdef PADDLE_WITH_CUDA
  auto *cuda_ctx = platform::DeviceContextPool::Instance().GetByPlace(place_);
  void *peer_cuda_ptr_ = memory::Alloc<platform::CUDAPlace>(
      boost::get<platform::CUDAPlace>(peer_place), this->size() * sizeof(T));
  memory::Copy(boost::get<platform::CUDAPlace>(peer_place),
               static_cast<void *>(peer_cuda_ptr_), place_,
               static_cast<const void *>(cuda_ptr_), this->size() * sizeof(T),
               cuda_ctx->stream());
  cuda_ctx->Wait();
  memory::Free<platform::CUDAPlace>(place_, static_cast<void *>(cuda_ptr_));
  place_ = boost::get<platform::CUDAPlace>(peer_place);
  cuda_ptr_ = peer_cuda_ptr_;
#endif
}

template class Vector<int>;
template class Vector<unsigned>;
template class Vector<size_t>;
template class Vector<int64_t>;

}  // namespace framework
}  // namespace paddle
