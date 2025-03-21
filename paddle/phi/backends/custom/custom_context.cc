/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#define EIGEN_USE_GPU
#include "paddle/phi/backends/custom/custom_context.h"

#include <algorithm>
#include <array>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "glog/logging.h"
#include "paddle/common/exception.h"
#include "paddle/phi/backends/context_pool.h"

#include "paddle/phi/common/place.h"
#include "paddle/phi/backends/device_guard.h"
#include "paddle/phi/backends/stream.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/memory/allocation/allocator_facade.h"

#include "unsupported/Eigen/CXX11/Tensor"

#include "paddle/phi/core/enforce.h"

#ifndef DEVICEPROP_TYPE
#define DEVICEPROP_TYPE void*
#endif

namespace phi {

namespace internal {
  
class EigenGpuStreamDevice : public Eigen::StreamInterface {
 public:
  EigenGpuStreamDevice()
      : stream_(nullptr),
        allocator_(nullptr),
        device_prop_(nullptr),
        scratch_(nullptr),
        semaphore_(nullptr),
        allocations_() {
    Eigen::initializeDeviceProp();
  }
  ~EigenGpuStreamDevice() override = default;

  void Reinitialize(STREAM_TYPE cuda_stream,
                    Allocator* allocator,
                    GPUPlace place) {
    stream_ = cuda_stream;
    place_ = place;
    allocator_ = allocator;
    device_prop_ = &Eigen::m_deviceProperties[place.device];
  }

  const STREAM_TYPE& stream() const override { return stream_; }

  const DEVICEPROP_TYPE& deviceProperties() const override {
    return *device_prop_;
  }

  void* allocate(size_t num_bytes) const override {
    if (UNLIKELY(num_bytes == 0)) {
      return nullptr;
    }
    auto buf = allocator_->Allocate(num_bytes);
    VLOG(4) << "Eigen allocated at " << buf->ptr() << " requested "
            << num_bytes;
    void* retv = buf->ptr();
    {
      std::lock_guard<std::mutex> lock(mtx_);
      allocations_.emplace(retv, std::move(buf));
    }
    return retv;
  }

  void deallocate(void* buffer) const override {
    if (LIKELY(buffer)) {
      std::lock_guard<std::mutex> lock(mtx_);
      allocations_.erase(buffer);
    }
  }

  void* scratchpad() const override {
    if (scratch_ == nullptr) {
      scratch_ = allocate(Eigen::kGpuScratchSize + sizeof(unsigned int));
    }
    return scratch_;
  }

  unsigned int* semaphore() const override {
    if (semaphore_ == nullptr) {
      char* scratch = static_cast<char*>(scratchpad()) + Eigen::kGpuScratchSize;
      semaphore_ = reinterpret_cast<unsigned int*>(scratch);
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(
          hipMemsetAsync(semaphore_, 0, sizeof(unsigned int), stream()));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaMemsetAsync(semaphore_, 0, sizeof(unsigned int), stream()));
#endif
    }
    return semaphore_;
  }

 private:
  GPUPlace place_;
  STREAM_TYPE stream_;                // not owned;
  Allocator* allocator_;              // not owned;
  const DEVICEPROP_TYPE* device_prop_;  // not owned;
  mutable void* scratch_;
  mutable unsigned int* semaphore_;
  mutable std::mutex mtx_;  // to protect allocations_
  mutable std::unordered_map<void*, Allocator::AllocationPtr> allocations_;
};

}  // namespace internal

struct CustomContext::Impl {
  explicit Impl(const CustomPlace& place) : place_(place) {}

  ~Impl() {}

  void Init() {
    phi::DeviceGuard guard(place_);
    stream_.reset(new phi::stream::Stream());
    stream_->Init(place_);
    // stream_ = new CUDAStream(place_);
  }

  const Place& GetPlace() const { return place_; }

  STREAM_TYPE stream() const {
    return reinterpret_cast<STREAM_TYPE>(stream_->raw_stream());
  }

  std::shared_ptr<phi::stream::Stream> GetStream() const { return stream_; }

  void SetStream(std::shared_ptr<phi::stream::Stream> stream) {
    stream_ = stream;
  }

  void InitEigenDevice() {
    PADDLE_ENFORCE_NOT_NULL(
        allocator_,
        common::errors::InvalidArgument(
            "The allocator for eigen device is nullptr. It must not be null."));
    eigen_stream_ = std::make_unique<internal::EigenGpuStreamDevice>();
    eigen_stream_->Reinitialize(stream(), allocator_, place_);
    eigen_device_ = new Eigen::GpuDevice(eigen_stream_.get());
  }

  void DestroyInternalEigenDevice() {
    if (owned_ && eigen_device_ != nullptr) {
      delete eigen_device_;
      eigen_device_ = nullptr;
    }
  }

  void SetEigenDevice(Eigen::GpuDevice* device) { eigen_device_ = device; }

  void SetEigenDevice(std::function<Eigen::GpuDevice*()>&& creator) {
    eigen_device_creator_ = std::move(creator);
  }


  Eigen::GpuDevice* eigen_device() {
    std::call_once(flag_eigen_device_, [&]() {
      if (!eigen_device_) {
        if (!eigen_device_creator_)
          InitEigenDevice();
        else
          eigen_device_ = eigen_device_creator_();
      }
    });
    PADDLE_ENFORCE_NOT_NULL(
        eigen_device_,
        common::errors::InvalidArgument(
            "The GPU eigen_device is nullptr. It must not be null."));
    return eigen_device_;
  }

  // Eigen::DefaultDevice* eigen_device_{nullptr};

  // Eigen::DefaultDevice* GetEigenDevice() const {
  //   PADDLE_ENFORCE_NE(
  //       eigen_device_,
  //       nullptr,
  //       common::errors::Unavailable("the custom eigen_device is nullptr."));
  //   return eigen_device_;
  // }  

  void Wait() const { stream_->Wait(); }

  phi::ccl::CCLComm xccl_comm() const { return comm_; }

  void set_xccl_comm(phi::ccl::CCLComm comm) { comm_ = comm; }

  Place place_;

  std::shared_ptr<phi::stream::Stream> stream_;
  // CUDAStream* stream_{nullptr};

  phi::ccl::CCLComm comm_;

  //////////////////////
  int compute_capability_ = 0;
  int runtime_version_ = 0;
  int driver_version_ = 0;
  int multi_process_ = 0;
  int max_threads_per_mp_ = 0;
  int max_threads_per_block_ = 0;
  std::array<unsigned int, 3> max_grid_dim_size_;

  Eigen::GpuDevice* eigen_device_{nullptr};
  std::function<Eigen::GpuDevice*()> eigen_device_creator_{nullptr};
  std::once_flag flag_eigen_device_;

  std::unique_ptr<internal::EigenGpuStreamDevice> eigen_stream_{nullptr};
};

void CustomContext::Init() { impl_->Init(); }

const Place& CustomContext::GetPlace() const { return impl_->GetPlace(); }

STREAM_TYPE CustomContext::stream() const { return impl_->stream(); }

std::shared_ptr<phi::stream::Stream> CustomContext::GetStream() const {
  return impl_->GetStream();
}

void CustomContext::SetStream(std::shared_ptr<phi::stream::Stream> stream) {
  impl_->SetStream(stream);
}

// void CustomContext::SetStream(gpuStream_t stream) { impl_->SetStream(stream); }
void CustomContext::Wait() const { return impl_->Wait(); }

CustomContext::CustomContext(const CustomPlace& place)
    : DeviceContext(), impl_(std::make_unique<Impl>(place)) {
  impl_->Init();
}

CustomContext::~CustomContext() { impl_.reset(); }

phi::ccl::CCLComm CustomContext::xccl_comm() const {
  return impl_->xccl_comm();
}

void CustomContext::set_xccl_comm(phi::ccl::CCLComm comm) {
  impl_->set_xccl_comm(comm);
}

////////////////////////for cuda///////////////////////////////
int CustomContext::GetComputeCapability() const {
  return impl_->compute_capability_;
}

int CustomContext::GetMaxThreadsPerBlock() const {
  return impl_->max_threads_per_block_;
}

int CustomContext::GetSMCount() const { return impl_->multi_process_; }

std::array<unsigned int, 3> CustomContext::GetCUDAMaxGridDimSize() const {
  return impl_->max_grid_dim_size_;
}

int CustomContext::GetMaxPhysicalThreadCount() const {
  return impl_->multi_process_ * impl_->max_threads_per_mp_;
}

void CustomContext::SetComputeCapability(int val) {
  impl_->compute_capability_ = val;
}

void CustomContext::SetMaxThreadsPerMultiProcessor(int val) {
  impl_->max_threads_per_mp_ = val;
}

void CustomContext::SetMultiProcessors(int val) { impl_->multi_process_ = val; }

void CustomContext::SetMaxThreadsPerBlock(int val) {
  impl_->max_threads_per_block_ = val;
}

void CustomContext::SetMaxGridDimSize(const std::array<unsigned int, 3>& val) {
  impl_->max_grid_dim_size_ = val;
}

void CustomContext::SetDriverVersion(int val) { impl_->driver_version_ = val; }

void CustomContext::SetRuntimeVersion(int val) { impl_->runtime_version_ = val; }

Eigen::DefaultDevice* CustomContext::eigen_device() const {
  return impl_->GetEigenDevice();
}

void CustomContext::SetEigenDevice(Eigen::DefaultDevice* device) {
  impl_->eigen_device_ = device;
}
////////////////////////for cuda///////////////////////////////
}  // namespace phi
