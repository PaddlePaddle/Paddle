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

#include "paddle/platform/enforce.h"
#include "paddle/platform/place.h"

#ifndef PADDLE_ONLY_CPU
#include "paddle/platform/dynload/cublas.h"
#include "paddle/platform/dynload/cudnn.h"
#include "paddle/platform/gpu_info.h"
#define EIGEN_USE_GPU
#endif
#include <memory>
#include "paddle/platform/place.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace paddle {
namespace platform {

class CPUDeviceContext : public DeviceContext {
 public:
  CPUDeviceContext() { eigen_device_.reset(new Eigen::DefaultDevice()); }
  explicit CPUDeviceContext(CPUPlace place) {
    eigen_device_.reset(new Eigen::DefaultDevice());
  }

  Eigen::DefaultDevice* GetEigenDevice() const { return eigen_device_.get(); }
  Place GetPlace() const { return CPUPlace(); }

 private:
  std::unique_ptr<Eigen::DefaultDevice> eigen_device_;
};

#ifdef PADDLE_WITH_CUDA

// The CUDADeviceContext is a parameter to framework::OperatorBase::Run:
/*
  virtual void Run(const Scope& scope,
                   const platform::DeviceContext& dev_ctx) const = 0;
*/
// To call Eigen functions in Run, we'd need to provide a parameter of
// type Eigen::CpuDevice, from CUDADeviceContext::GetEigenDevice().
//
//   SomeEigenFunction(dev_ctx.GetEigenDevice(), ...);
//
// If we are going to call CUDA, cuDNN, cuBLAS function, we need to
// pass them handles returned by stream, cudnn_handle, cublas_handle.
// For example:
//
//  SomeCUDNNFunction(dev_ctx.cudnn_handle(), ...);
//
class CUDADeviceContext : public DeviceContext {
 public:
  explicit CUDADeviceContext(GPUPlace place);
  virtual ~CUDADeviceContext();

  Eigen::GpuDevice* GetEigenDevice() const { return eigen_device_.get(); }
  Place GetPlace() const override { return place_; }

  /*! \brief  Wait for all operations completion in the stream. */
  void Wait() const override { PADDLE_ENFORCE(cudaStreamSynchronize(stream_)); }

  cublasHandle_t cublas_handle() const { return cublas_handle_; }
  cudnnHandle_t cudnn_handle() const { return cudnn_handle_; }
  cudaStream_t stream() const { return stream_; }

 private:
  // Eigen requires that a Eigen::GpuDevice instance being initialized
  // from a class derived from Eigen::StreamInterface.
  class EigenCudaStreamDevice : public Eigen::StreamInterface {
   public:
    EigenCudaStreamDevice();
    ~EigenCudaStreamDevice() override {}

    // https://github.com/PaddlePaddle/Paddle/pull/3497#issue-250238535
    // explained that initializing CUDA stream in the constructor
    // would cause SEGFAULT, so we add this method.
    void SetValues(const cudaStream_t* cuda_stream, GPUPlace place);

    const cudaStream_t& stream() const override;
    const cudaDeviceProp& deviceProperties() const override;
    void* allocate(size_t num_bytes) const override;
    void deallocate(void* buffer) const override;
    void* scratchpad() const override;
    unsigned int* semaphore() const override;

   private:
    GPUPlace place_;
    const cudaStream_t* stream_;         // not owned;
    const cudaDeviceProp* device_prop_;  // not owned;
    mutable void* scratch_;
    mutable unsigned int* semaphore_;
  };

  GPUPlace place_;
  std::unique_ptr<Eigen::GpuDevice> eigen_device_;
  std::unique_ptr<EigenCudaStreamDevice> eigen_stream_;

  cudaStream_t stream_;
  cudnnHandle_t cudnn_handle_;
  cublasHandle_t cublas_handle_;
};

#endif  // PADDLE_WITH_CUDA

#ifdef PADDLE_WITH_CUDA
typedef boost::variant<CPUDeviceContext, CUDADeviceContext> DeviceContext;
#else
typedef boost::variant<CPUDeviceContext> DeviceContext;
#endif  // PADDLE_WITH_CUDA

}  // namespace platform
}  // namespace paddle
