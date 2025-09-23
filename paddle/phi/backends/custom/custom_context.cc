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
#include "paddle/phi/backends/custom/custom_context.h"

#include "paddle/common/exception.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/device_guard.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/backends/stream.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/memory/allocation/allocator_facade.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace phi {

struct CustomContext::Impl {
  explicit Impl(const CustomPlace& place) : place_(place) {}

  ~Impl() {
    if (owned_ && eigen_device_) {
      DeviceManager::DestroyEigenDevice(place_, eigen_device_);
    }
    if (stream_owned_ && stream_) {
      stream_ = nullptr;
    }
    if (blas_handle_) {
      DeviceManager::DestroyBlasHandle(place_,
                                       reinterpret_cast<void*>(blas_handle_));
    }
    if (blas_tensor_core_handle_) {
      DeviceManager::DestroyBlasHandle(
          place_, reinterpret_cast<void*>(blas_tensor_core_handle_));
    }
    if (blas_tf32_tensor_core_handle_) {
      DeviceManager::DestroyBlasHandle(
          place_, reinterpret_cast<void*>(blas_tf32_tensor_core_handle_));
    }
    if (blaslt_handle_) {
      DeviceManager::DestroyBlasLtHandle(
          place_, reinterpret_cast<void*>(blaslt_handle_));
    }
  }

  void Init() {
    owned_ = true;
    phi::DeviceGuard guard(place_);
    compute_capability_ = DeviceManager::GetComputeCapability(place_);
    runtime_version_ = DeviceManager::GetRuntimeVersion(place_);
    driver_version_ = DeviceManager::GetDriverVersion(place_);
    multi_process_ = DeviceManager::GetMultiProcessors(place_);
    max_threads_per_mp_ = DeviceManager::GetMaxThreadsPerMultiProcessor(place_);
    max_threads_per_block_ = DeviceManager::GetMaxThreadsPerBlock(place_);
    max_grid_dim_size_ = DeviceManager::GetMaxGridDimSize(place_);
    eigen_device_ =
        reinterpret_cast<Eigen::GpuDevice*>(DeviceManager::InitEigenDevice(
            place_, stream_->raw_stream(), allocator_));

    stream_.reset(new phi::stream::Stream());
    stream_->Init(place_);
  }

  void PartialInitWithoutAllocator() {
    owned_ = true;
    stream_owned_ = true;
    phi::DeviceGuard guard(place_);
    compute_capability_ = DeviceManager::GetComputeCapability(place_);
    runtime_version_ = DeviceManager::GetRuntimeVersion(place_);
    driver_version_ = DeviceManager::GetDriverVersion(place_);
    multi_process_ = DeviceManager::GetMultiProcessors(place_);
    max_threads_per_mp_ = DeviceManager::GetMaxThreadsPerMultiProcessor(place_);
    max_threads_per_block_ = DeviceManager::GetMaxThreadsPerBlock(place_);
    max_grid_dim_size_ = DeviceManager::GetMaxGridDimSize(place_);

    stream_.reset(new phi::stream::Stream());
    stream_->Init(place_);
  }

  void PartialInitWithAllocator() {
    owned_ = true;
    stream_owned_ = true;
    phi::DeviceGuard guard(place_);
  }

  const Place& GetPlace() const { return place_; }

  phi::stream::stream_t stream() const {
    return reinterpret_cast<phi::stream::stream_t>(stream_->raw_stream());
  }

  std::shared_ptr<phi::stream::Stream> GetStream() const { return stream_; }

  void SetStream(std::shared_ptr<phi::stream::Stream> stream) {
    stream_ = stream;
  }

  void SetEigenDevice(Eigen::GpuDevice* device) { eigen_device_ = device; }

  void SetEigenDevice(std::function<Eigen::GpuDevice*()>&& creator) {
    eigen_device_creator_ = std::move(creator);
  }

  Eigen::GpuDevice* eigen_device() {
    std::call_once(flag_eigen_device_, [&]() {
      if (!eigen_device_) {
        if (!eigen_device_creator_) {
          // use default initial
          eigen_device_ = reinterpret_cast<Eigen::GpuDevice*>(
              DeviceManager::InitEigenDevice(
                  place_, stream_->raw_stream(), allocator_));
        } else {
          eigen_device_ = eigen_device_creator_();
        }
      }
    });
    PADDLE_ENFORCE_NOT_NULL(
        eigen_device_,
        common::errors::InvalidArgument(
            "The custom eigen_device is nullptr. It must not be null."));
    return eigen_device_;
  }

  void Wait() const { stream_->Wait(); }

  void WaitEvent(phi::event::event_t ev) const {
    event::Event event_(place_, ev);
    stream_->WaitEvent(&event_);
  }

  void RecordEvent(phi::event::event_t ev,
                   const std::function<void()>& callback) const {
    event::Event event_(place_, ev);
    stream_->RecordEvent(&event_, callback);
  }

  void RecordEvent(phi::event::event_t ev) const {
    event::Event event_(place_, ev);
    stream_->RecordEvent(&event_);
  }

  phi::ccl::CCLComm xccl_comm() const { return comm_; }

  void set_xccl_comm(phi::ccl::CCLComm comm) { comm_ = comm; }

  cublasHandle_t GetBlasHandle() {
    std::call_once(flag_blas_, [&]() {
      if (!blas_handle_) {
        if (!blas_handle_creator_) {
          phi::DeviceManager::InitBlasHandle(
              place_, reinterpret_cast<void**>(&blas_handle_), stream());
        } else {
          blas_handle_ = blas_handle_creator_();
        }
      }

      if (!blas_tensor_core_handle_) {
        if (!blas_tensor_core_handle_creator_) {
          phi::DeviceManager::InitBlasHandle(
              place_,
              reinterpret_cast<void**>(&blas_tensor_core_handle_),
              stream());
        } else {
          blas_tensor_core_handle_ = blas_tensor_core_handle_creator_();
        }
        phi::DeviceManager::BlasSetMathMode(
            place_, blas_tensor_core_handle_, BLAS_TENSOR_OP_MATH);
      }

      if (!blas_tf32_tensor_core_handle_) {
        if (!blas_tf32_tensor_core_handle_creator_) {
          phi::DeviceManager ::InitBlasHandle(
              place_,
              reinterpret_cast<void**>(&blas_tf32_tensor_core_handle_),
              stream());
        } else {
          blas_tf32_tensor_core_handle_ =
              blas_tf32_tensor_core_handle_creator_();
        }
        phi::DeviceManager::BlasSetMathMode(
            place_, blas_tf32_tensor_core_handle_, BLAS_TF32_TENSOR_OP_MATH);
      }
    });
    PADDLE_ENFORCE_NOT_NULL(
        blas_handle_,
        common::errors::InvalidArgument(
            "The Custom Device blas handle is nullptr. It must not be null."));
    return blas_handle_;
  }

  void SetBlasHandle(cublasHandle_t blas) { blas_handle_ = blas; }

  void SetBlasHandle(std::function<cublasHandle_t()>&& handle_creator) {
    blas_handle_creator_ = std::move(handle_creator);
  }

  void SetBlasTensorCoreHandle(cublasHandle_t handle) {
    blas_tensor_core_handle_ = handle;
  }

  void SetBlasTensorCoreHandle(
      std::function<cublasHandle_t()>&& handle_creator) {
    blas_tensor_core_handle_creator_ = std::move(handle_creator);
  }

  void SetBlasTF32Handle(cublasHandle_t handle) {
    blas_tf32_tensor_core_handle_ = handle;
  }

  void SetBlasTF32Handle(std::function<cublasHandle_t()>&& handle_creator) {
    blas_tf32_tensor_core_handle_creator_ = std::move(handle_creator);
  }

  void SetBlasLtHandle(cublasLtHandle_t blaslt) { blaslt_handle_ = blaslt; }

  void SetBlasLtHandle(std::function<cublasLtHandle_t()>&& handle_creator) {
    blaslt_handle_creator_ = std::move(handle_creator);
  }

  cublasLtHandle_t GetBlasLtHandle() {
    std::call_once(flag_blaslt_, [&]() {
      if (!blaslt_handle_) {
        if (!blaslt_handle_creator_)
          phi::DeviceManager::InitBlasLtHandle(
              place_, reinterpret_cast<void**>(&blaslt_handle_));
        else
          blaslt_handle_ = blaslt_handle_creator_();
      }
    });
    PADDLE_ENFORCE_NOT_NULL(
        blaslt_handle_,
        common::errors::InvalidArgument("The Custom Device blasLt handle is "
                                        "nullptr. It must not be null."));
    return blaslt_handle_;
  }

  bool IsTensorCoreAvailable() const {
    return blas_tensor_core_handle_ != nullptr;
  }

  inline void CublasCall(const std::function<void(cublasHandle_t)>& callback) {
    std::call_once(flag_cublas_, [&]() {
      if (!blas_handle_) {
        if (!blas_handle_creator_) {
          phi::DeviceManager::InitBlasHandle(
              place_, reinterpret_cast<void**>(&blas_handle_), stream());
        } else {
          blas_handle_ = blas_handle_creator_();
        }
      }
      if (!blas_tensor_core_handle_) {
        if (!blas_tensor_core_handle_creator_) {
          phi::DeviceManager::InitBlasHandle(
              place_,
              reinterpret_cast<void**>(&blas_tensor_core_handle_),
              stream());
        } else {
          blas_tensor_core_handle_ = blas_tensor_core_handle_creator_();
        }
        phi::DeviceManager::BlasSetMathMode(
            place_, blas_tensor_core_handle_, BLAS_TENSOR_OP_MATH);
      }
      if (!blas_tf32_tensor_core_handle_) {
        if (!blas_tf32_tensor_core_handle_creator_) {
          phi::DeviceManager::InitBlasHandle(
              place_,
              reinterpret_cast<void**>(&blas_tf32_tensor_core_handle_),
              stream());
        } else {
          blas_tf32_tensor_core_handle_ =
              blas_tf32_tensor_core_handle_creator_();
        }
        phi::DeviceManager::BlasSetMathMode(
            place_, blas_tf32_tensor_core_handle_, BLAS_TF32_TENSOR_OP_MATH);
      }
    });

    if (blas_tf32_tensor_core_handle_ && phi::AllowTF32Cublas()) {
      std::lock_guard<std::mutex> guard(blas_tf32_mtx_);
      callback(blas_tf32_tensor_core_handle_);
    } else {
      std::lock_guard<std::mutex> guard(blas_mtx_);
      callback(blas_handle_);
    }
  }

  inline void TensorCoreCublasCallIfAvailable(
      const std::function<void(cublasHandle_t)>& callback) {
    std::call_once(flag_tensorcore_cublas_, [&]() {
      if (!blas_handle_) {
        if (!blas_handle_creator_) {
          phi::DeviceManager::InitBlasHandle(
              place_, reinterpret_cast<void**>(&blas_handle_), stream());
        } else {
          blas_handle_ = blas_handle_creator_();
        }
      }
      if (!blas_tensor_core_handle_) {
        if (!blas_tensor_core_handle_creator_) {
          phi::DeviceManager::InitBlasHandle(
              place_,
              reinterpret_cast<void**>(&blas_tensor_core_handle_),
              stream());
        } else {
          blas_tensor_core_handle_ = blas_tensor_core_handle_creator_();
        }
        phi::DeviceManager::BlasSetMathMode(
            place_, blas_tensor_core_handle_, BLAS_TENSOR_OP_MATH);
      }
      if (!blas_tf32_tensor_core_handle_) {
        if (!blas_tf32_tensor_core_handle_creator_) {
          phi::DeviceManager::InitBlasHandle(
              place_,
              reinterpret_cast<void**>(&blas_tf32_tensor_core_handle_),
              stream());
        } else {
          blas_tf32_tensor_core_handle_ =
              blas_tf32_tensor_core_handle_creator_();
        }
        phi::DeviceManager::BlasSetMathMode(
            place_, blas_tf32_tensor_core_handle_, BLAS_TF32_TENSOR_OP_MATH);
      }
    });
    if (blas_tensor_core_handle_ != nullptr) {
      std::lock_guard<std::mutex> guard(blas_tensor_core_mtx_);
      callback(blas_tensor_core_handle_);
    } else {
      std::lock_guard<std::mutex> guard(blas_mtx_);
      callback(blas_handle_);
    }
  }

  bool HasDnnAttr(const std::string& attr_name) const {
    return dnn_attrs_.count(attr_name) != 0UL;
  }

  const Attribute& GetDnnAttr(const std::string& attr_name) const {
    auto iter = dnn_attrs_.find(attr_name);
    PADDLE_ENFORCE_NE(iter,
                      dnn_attrs_.end(),
                      common::errors::NotFound(
                          "Attribute `%s` is not found in CustomContext."));
    return iter->second;
  }

  void SetDnnAttr(const std::string& attr_name, Attribute attr) {
    dnn_attrs_[attr_name] = attr;
  }

  void ClearDnnAttr() { dnn_attrs_.clear(); }

  Place place_;

  std::shared_ptr<phi::stream::Stream> stream_;

  Allocator* allocator_{nullptr};

  phi::ccl::CCLComm comm_;

  bool owned_{false};
  bool stream_owned_{false};
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
  cublasHandle_t blas_handle_{nullptr};
  std::function<cublasHandle_t()> blas_handle_creator_{nullptr};
  cublasHandle_t blas_tensor_core_handle_{nullptr};
  std::function<cublasHandle_t()> blas_tensor_core_handle_creator_{nullptr};
  cublasHandle_t blas_tf32_tensor_core_handle_{nullptr};
  std::function<cublasHandle_t()> blas_tf32_tensor_core_handle_creator_{
      nullptr};
  cublasLtHandle_t blaslt_handle_{nullptr};
  std::function<cublasLtHandle_t()> blaslt_handle_creator_{nullptr};

  static thread_local AttributeMap dnn_attrs_;

  enum BLASMathMode {
    BLAS_DEFAULT_MATH = 0,
    BLAS_TENSOR_OP_MATH = 1,
    BLAS_TF32_TENSOR_OP_MATH = 2
  };

  std::once_flag flag_sparse_;
  std::once_flag flag_blas_;
  std::once_flag flag_blaslt_;
  std::once_flag flag_dnn_;
  std::once_flag flag_solver_;
  std::once_flag flag_cublas_;
  std::once_flag flag_tensorcore_cublas_;

  mutable std::mutex blas_mtx_;
  mutable std::mutex blas_tensor_core_mtx_;
  mutable std::mutex blas_tf32_mtx_;
  mutable std::mutex sparse_mtx_;
  mutable std::mutex stream_call_back_mtx_;
  mutable std::future<void> last_future_;
};

thread_local AttributeMap CustomContext::Impl::dnn_attrs_ = {};

CustomContext::CustomContext(const CustomPlace& place)
    : DeviceContext(), impl_(std::make_unique<Impl>(place)) {
  impl_->PartialInitWithoutAllocator();
}

CustomContext::~CustomContext() { impl_.reset(); }

void CustomContext::Init() {
  impl_->allocator_ = const_cast<Allocator*>(&this->GetAllocator());
  impl_->Init();
}

void CustomContext::PartialInitWithoutAllocator() {
  impl_->PartialInitWithoutAllocator();
}

void CustomContext::PartialInitWithAllocator() {
  impl_->allocator_ = const_cast<Allocator*>(&this->GetAllocator());  // NOLINT
  impl_->PartialInitWithAllocator();
}

const Place& CustomContext::GetPlace() const { return impl_->GetPlace(); }

phi::stream::stream_t CustomContext::stream() const { return impl_->stream(); }

std::shared_ptr<phi::stream::Stream> CustomContext::GetStream() const {
  return impl_->GetStream();
}

void CustomContext::SetStream(std::shared_ptr<phi::stream::Stream> stream) {
#if !defined(_WIN32)
  this->SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                         .GetAllocator(impl_->GetPlace(), stream->raw_stream())
                         .get());
#endif
  impl_->allocator_ = const_cast<Allocator*>(&this->GetAllocator());  // NOLINT
  impl_->SetStream(stream);
}

void CustomContext::Wait() const { return impl_->Wait(); }

void CustomContext::RecordEvent(phi::event::event_t ev,
                                const std::function<void()>& callback) const {
  impl_->RecordEvent(ev, callback);
}

void CustomContext::RecordEvent(phi::event::event_t ev) const {
  impl_->RecordEvent(ev);
}

Eigen::GpuDevice* CustomContext::eigen_device() const {
  return impl_->eigen_device();
}

void CustomContext::SetEigenDevice(Eigen::GpuDevice* device) {
  impl_->SetEigenDevice(device);
}

void CustomContext::SetEigenDevice(
    std::function<Eigen::GpuDevice*()>&& creator) {
  impl_->SetEigenDevice(std::move(creator));
}

phi::ccl::CCLComm CustomContext::xccl_comm() const {
  return impl_->xccl_comm();
}

void CustomContext::set_xccl_comm(phi::ccl::CCLComm comm) {
  impl_->set_xccl_comm(comm);
}

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

void CustomContext::SetRuntimeVersion(int val) {
  impl_->runtime_version_ = val;
}

cublasHandle_t CustomContext::cublas_handle() const {
  return impl_->GetBlasHandle();
}

cublasLtHandle_t CustomContext::cublaslt_handle() const {
  return impl_->GetBlasLtHandle();
}

void CustomContext::SetBlasHandle(cublasHandle_t blas) {
  impl_->SetBlasHandle(blas);
}

void CustomContext::SetBlasHandle(std::function<cublasHandle_t()>&& func) {
  impl_->SetBlasHandle(std::move(func));
}

void CustomContext::SetBlasTensorCoreHandle(cublasHandle_t handle) {
  impl_->SetBlasTensorCoreHandle(handle);
}

void CustomContext::SetBlasTensorCoreHandle(
    std::function<cublasHandle_t()>&& func) {
  impl_->SetBlasTensorCoreHandle(std::move(func));
}

void CustomContext::SetBlasTF32Handle(cublasHandle_t handle) {
  impl_->SetBlasTF32Handle(handle);
}

void CustomContext::SetBlasTF32Handle(std::function<cublasHandle_t()>&& func) {
  impl_->SetBlasTF32Handle(std::move(func));
}

void CustomContext::SetBlasLtHandle(cublasLtHandle_t blaslt) {
  impl_->SetBlasLtHandle(blaslt);
}

void CustomContext::SetBlasLtHandle(std::function<cublasLtHandle_t()>&& func) {
  impl_->SetBlasLtHandle(std::move(func));
}

bool CustomContext::tensor_core_available() const {
  return impl_->IsTensorCoreAvailable();
}

void CustomContext::CublasCall(
    const std::function<void(cublasHandle_t)>& callback) const {
  impl_->CublasCall(callback);
}

void CustomContext::TensorCoreCublasCallIfAvailable(
    const std::function<void(cublasHandle_t)>& callback) const {
  impl_->TensorCoreCublasCallIfAvailable(callback);
}

bool CustomContext::HasDnnAttr(const std::string& attr_name) const {
  return impl_->HasDnnAttr(attr_name);
}

const Attribute& CustomContext::GetDnnAttr(const std::string& attr_name) const {
  return impl_->GetDnnAttr(attr_name);
}

void CustomContext::SetDnnAttr(const std::string& attr_name, Attribute attr) {
  return impl_->SetDnnAttr(attr_name, std::move(attr));
}

void CustomContext::ClearDnnAttr() { return impl_->ClearDnnAttr(); }

}  // namespace phi
