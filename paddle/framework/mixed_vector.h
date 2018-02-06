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
  using std::vector<T>::vector;

  Vector() {}
  Vector(const std::vector<T> &v) : std::vector<T>(v) {}  // NOLINT

  inline platform::Place place() const { return place_; }

  /*! Return a pointer to constant memory block. */
  inline const T *data(platform::Place place) const;

  /*! Return a pointer to mutable memory block. */
  inline T *mutable_data(platform::Place place);

  // TODO(dzhwinter): below interfaces should be removed
  /* Get device vector */
  T *cuda_data() {
    CopyToCUDA();
    PADDLE_ENFORCE_NOT_NULL(
        cuda_ptr_, "No data or Insufficient CUDA memory to allocation");
    return static_cast<T *>(cuda_ptr_.get());
  }

  /* Get host vector */
  T *data() { return std::vector<T>::data(); }
  const T *data() const { return std::vector<T>::data(); }

  /* Synchronize host vector to device vector */
  void CopyToCUDA();
  /* Synchronize device vector to host vector */
  void CopyFromCUDA();
  /* Switch device vector location */
  void CopyToPeer(platform::Place);

 private:
  std::shared_ptr<void> cuda_ptr_;
  size_t cuda_size_ = 0;  // device vector numel
  platform::CUDAPlace place_;
};

template <typename T>
inline const T *Vector<T>::data(platform::Place place) const {
  if (platform::is_cpu_place(place)) {
    return std::vector<T>::data();
  } else if (platform::is_gpu_place(place)) {
    if (cuda_ptr_ == nullptr) {
      return nullptr;
    }
    if (platform::is_same_place(place, place_)) {
      return static_cast<const T *>(cuda_ptr_.get());
    } else {
      PADDLE_THROW(
          "Unmatched place. Please use `mutable_data` copy lod to the target "
          "Place first.");
    }
  } else {
    PADDLE_THROW("Unsupport Place.");
  }
}

template <typename T>
inline T *Vector<T>::mutable_data(platform::Place place) {
  if (platform::is_cpu_place(place)) {
    return std::vector<T>::data();
  } else if (platform::is_gpu_place(place)) {
    if (!platform::is_same_place(place, place_)) {
      place_ = boost::get<platform::CUDAPlace>(place);
    }
#ifdef PADDLE_WITH_CUDA
    if (cuda_size_ < this->size() || cuda_ptr_ == nullptr) {
      cuda_ptr_.reset(
          memory::Alloc<platform::CUDAPlace>(place_, this->size() * sizeof(T)),
          memory::PlainDeleter<void, platform::CUDAPlace>(place_));
    }
    cuda_size_ = this->size();
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto *ctx = pool.GetByPlace(place_);
    memory::Copy(place_, cuda_ptr_.get(), platform::CPUPlace(),
                 static_cast<const void *>(this->data()),
                 this->size() * sizeof(T), ctx->stream());
    ctx->Wait();
    return static_cast<T *>(cuda_ptr_.get());
#else
    return nullptr;
#endif
  } else {
    PADDLE_THROW("Unsupport Place.");
  }
}

template <typename T>
void Vector<T>::CopyToCUDA() {
#ifdef PADDLE_WITH_CUDA
  if (cuda_size_ < this->size() || cuda_ptr_ == nullptr) {
    cuda_ptr_.reset(
        memory::Alloc<platform::CUDAPlace>(place_, this->size() * sizeof(T)),
        memory::PlainDeleter<void, platform::CUDAPlace>(place_));
  }
  cuda_size_ = this->size();
  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto *ctx = pool.GetByPlace(place_);
  memory::Copy(place_, cuda_ptr_.get(), platform::CPUPlace(),
               static_cast<const void *>(this->data()),
               this->size() * sizeof(T), ctx->stream());
  ctx->Wait();
#endif
}

template <typename T>
void Vector<T>::CopyFromCUDA() {
#ifdef PADDLE_WITH_CUDA
  if (cuda_ptr_ == nullptr) {
    LOG(WARNING) << "No uncommitted cuda data.";
    return;
  }
  this->resize(cuda_size_);
  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto *ctx = pool.GetByPlace(place_);
  memory::Copy(platform::CPUPlace(), static_cast<void *>(this->data()), place_,
               static_cast<const void *>(cuda_ptr_.get()),
               this->size() * sizeof(T), ctx->stream());
  ctx->Wait();
#endif
}

}  // namespace framework
}  // namespace paddle
