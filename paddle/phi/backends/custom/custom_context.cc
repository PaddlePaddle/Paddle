// /* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License. */

// #include "paddle/phi/backends/custom/custom_context.h"

// #include "paddle/phi/backends/device_guard.h"
// #include "paddle/phi/backends/stream.h"

// namespace phi {

// struct CustomContext::Impl {
//   explicit Impl(const CustomPlace& place) : place_(place) {}

//   ~Impl() {}

//   void Init() {
//     phi::DeviceGuard guard(place_);
//     // stream_.reset(new phi::stream::Stream());
//     // stream_->Init(place_);
//     stream_ = new CUDAStream(place_);
//   }

//   const Place& GetPlace() const { return place_; }

//   STREAM_TYPE stream() const {
//     return reinterpret_cast<STREAM_TYPE>(stream_->raw_stream());
//   }

//   std::shared_ptr<phi::stream::Stream> GetStream() const { return stream_; }

//   // void SetStream(std::shared_ptr<phi::stream::Stream> stream) {
//   //   stream_ = stream;
//   // }

//   // void InitEigenDevice() {
//   //   PADDLE_ENFORCE_NOT_NULL(
//   //       allocator_,
//   //       common::errors::InvalidArgument(
//   //           "The allocator for eigen device is nullptr. It must not be
//   //           null."));
//   //   eigen_stream_ = std::make_unique<internal::EigenGpuStreamDevice>();
//   //   eigen_stream_->Reinitialize(stream(), allocator_, place_);
//   //   eigen_device_ = new Eigen::GpuDevice(eigen_stream_.get());
//   // }

//   // Eigen::GpuDevice* eigen_device() {
//   //   std::call_once(flag_eigen_device_, [&]() {
//   //     if (!eigen_device_) {
//   //       if (!eigen_device_creator_)
//   //         InitEigenDevice();
//   //       else
//   //         eigen_device_ = eigen_device_creator_();
//   //     }
//   //   });
//   //   PADDLE_ENFORCE_NOT_NULL(
//   //       eigen_device_,
//   //       common::errors::InvalidArgument(
//   //           "The GPU eigen_device is nullptr. It must not be null."));
//   //   return eigen_device_;
//   // }

//   void Wait() const { stream_->Wait(); }

//   phi::ccl::CCLComm xccl_comm() const { return comm_; }

//   void set_xccl_comm(phi::ccl::CCLComm comm) { comm_ = comm; }

//   Place place_;

//   // std::shared_ptr<phi::stream::Stream> stream_;
//   CUDAStream* stream_{nullptr};

//   phi::ccl::CCLComm comm_;

//   //////////////////////
//   int compute_capability_ = 0;
//   int runtime_version_ = 0;
//   int driver_version_ = 0;
//   int multi_process_ = 0;
//   int max_threads_per_mp_ = 0;
//   int max_threads_per_block_ = 0;
//   std::array<unsigned int, 3> max_grid_dim_size_;

//   Eigen::GpuDevice* eigen_device_{nullptr};
//   std::function<Eigen::GpuDevice*()> eigen_device_creator_{nullptr};
//   std::once_flag flag_eigen_device_;

//   std::unique_ptr<internal::EigenGpuStreamDevice> eigen_stream_{nullptr};
// };

// void CustomContext::Init() { impl_->Init(); }

// const Place& CustomContext::GetPlace() const { return impl_->GetPlace(); }

// STREAM_TYPE CustomContext::stream() const { return impl_->stream(); }

// std::shared_ptr<phi::stream::Stream> CustomContext::GetStream() const {
//   return impl_->GetStream();
// }

// void CustomContext::SetStream(std::shared_ptr<phi::stream::Stream> stream) {
//   impl_->SetStream(stream);
// }

// void CustomContext::SetStream(gpuStream_t stream) { impl_->SetStream(stream); }
// void CustomContext::Wait() const { return impl_->Wait(); }

// CustomContext::CustomContext(const CustomPlace& place)
//     : DeviceContext(), impl_(std::make_unique<Impl>(place)) {
//   impl_->Init();
// }

// CustomContext::~CustomContext() { impl_.reset(); }

// phi::ccl::CCLComm CustomContext::xccl_comm() const {
//   return impl_->xccl_comm();
// }

// void CustomContext::set_xccl_comm(phi::ccl::CCLComm comm) {
//   impl_->set_xccl_comm(comm);
// }

// ////////////////////////for cuda///////////////////////////////
// int CustomContext::GetComputeCapability() const {
//   return impl_->compute_capability_;
// }

// int CustomContext::GetMaxThreadsPerBlock() const {
//   return impl_->max_threads_per_block_;
// }

// int CustomContext::GetSMCount() const { return impl_->multi_process_; }

// std::array<unsigned int, 3> CustomContext::GetCUDAMaxGridDimSize() const {
//   return impl_->max_grid_dim_size_;
// }

// int GPUContext::GetMaxPhysicalThreadCount() const {
//   return impl_->multi_process_ * impl_->max_threads_per_mp_;
// }
// ////////////////////////for cuda///////////////////////////////
// }  // namespace phi


#include "paddle/phi/backends/custom/custom_context.h"

#include "paddle/phi/backends/device_guard.h"
#include "paddle/phi/backends/stream.h"

namespace phi {

struct CustomContext::Impl {
  explicit Impl(const CustomPlace& place) : place_(place) {}

  ~Impl() {}

  void Init() {
    phi::DeviceGuard guard(place_);
    stream_.reset(new phi::stream::Stream());
    stream_->Init(place_);
  }

  const Place& GetPlace() const { return place_; }

  void* stream() const {
    return reinterpret_cast<void*>(stream_->raw_stream());
  }

  std::shared_ptr<phi::stream::Stream> GetStream() const { return stream_; }

  void SetStream(std::shared_ptr<phi::stream::Stream> stream) {
    stream_ = stream;
  }

  void Wait() const { stream_->Wait(); }

  phi::ccl::CCLComm xccl_comm() const { return comm_; }

  void set_xccl_comm(phi::ccl::CCLComm comm) { comm_ = comm; }

  Place place_;

  std::shared_ptr<phi::stream::Stream> stream_;

  phi::ccl::CCLComm comm_;
};

void CustomContext::Init() { impl_->Init(); }

const Place& CustomContext::GetPlace() const { return impl_->GetPlace(); }

void* CustomContext::stream() const { return impl_->stream(); }

std::shared_ptr<phi::stream::Stream> CustomContext::GetStream() const {
  return impl_->GetStream();
}

void CustomContext::SetStream(std::shared_ptr<phi::stream::Stream> stream) {
  impl_->SetStream(stream);
}

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
}  // namespace phi