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

#include "paddle/platform/device_context.h"
#include "paddle/memory/memory.h"

namespace paddle {
namespace platform {

#ifdef PADDLE_WITH_CUDA

CUDADeviceContext::EigenCudaStreamDevice::EigenCudaStreamDevice()
    : scratch_(nullptr), semaphore_(nullptr) {
  Eigen::initializeDeviceProp();
}
CUDADeviceContext::EigenCudaStreamDevice::~EigenCudaStreamDevice() override {}

void CUDADeviceContext::EigenCudaStreamDevice::SetValues(
    const cudaStream_t* cuda_stream, GPUPlace place) {
  stream_ = cuda_stream;
  place_ = place;
  device_prop_ = &Eigen::m_deviceProperties[place.device];
}

const cudaStream_t& void CUDADeviceContext::EigenCudaStreamDevice::stream()
    const override {
  return *stream_;
}

const cudaDeviceProp& void
CUDADeviceContext::EigenCudaStreamDevice::deviceProperties() const override {
  return *device_prop_;
}

void* void CUDADeviceContext::EigenCudaStreamDevice::allocate(
    size_t num_bytes) const override {
  return paddle::memory::Alloc(place_, num_bytes);
}

void void CUDADeviceContext::EigenCudaStreamDevice::deallocate(
    void* buffer) const override {
  paddle::memory::Free(place_, buffer);
}

void* void CUDADeviceContext::EigenCudaStreamDevice::scratchpad()
    const override {
  if (scratch_ == NULL) {
    scratch_ = allocate(Eigen::kCudaScratchSize + sizeof(unsigned int));
  }
  return scratch_;
}

unsigned int* void CUDADeviceContext::EigenCudaStreamDevice::semaphore()
    const override {
  if (semaphore_ == NULL) {
    char* scratch = static_cast<char*>(scratchpad()) + Eigen::kCudaScratchSize;
    semaphore_ = reinterpret_cast<unsigned int*>(scratch);
    PADDLE_ENFORCE(
        cudaMemsetAsync(semaphore_, 0, sizeof(unsigned int), *stream_));
  }
  return semaphore_;
}

CUDADeviceContext::CUDADeviceContext(GPUPlace place) : place_(place) {
  // Create CUDA stream on the given device.
  SetDeviceId(place_.device);
  PADDLE_ENFORCE(cudaStreamCreate(&stream_));

  // Set the CUDA stream into the EigenCudaStreamDevice instance.
  eigen_stream_.reset(new EigenCudaStreamDevice());
  eigen_stream_->SetValues(&stream_, place);

  // Initialize Eigen::CpuDevice using EigenCudaStreamDevice.
  eigen_device_.reset(new Eigen::GpuDevice(eigen_stream_.get()));

  // Create other handles in addition to the CUDA stream.
  PADDLE_ENFORCE(dynload::cublasCreate(&cublas_handle_));
  PADDLE_ENFORCE(dynload::cublasSetStream(cublas_handle_, stream_));
  PADDLE_ENFORCE(dynload::cudnnCreate(&cudnn_handle_));
  PADDLE_ENFORCE(dynload::cudnnSetStream(cudnn_handle_, stream_));
}

CUDADeviceContext::~CUDADeviceContext() {
  // Wait for the completion of all operations before destructing.
  SetDeviceId(place_.device);
  Wait();

  // Note: the destruction order must be the same with the
  // construction order.
  PADDLE_ENFORCE(dynload::cublasDestroy(cublas_handle_));
  PADDLE_ENFORCE(dynload::cudnnDestroy(cudnn_handle_));
  eigen_stream_.reset();
  eigen_device_.reset();
  PADDLE_ENFORCE(cudaStreamDestroy(stream_));
}

#endif  // PADDLE_WITH_CUDA

}  // namespace platform
}  // namespace paddle
