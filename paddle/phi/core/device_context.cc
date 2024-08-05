//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/device_context.h"

#if defined(PADDLE_WITH_CUDA)
#include "paddle/phi/backends/gpu/cuda/cuda_graph.h"
#elif defined(PADDLE_WITH_HIP)
#include "paddle/phi/backends/gpu/rocm/hip_graph.h"
#endif

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/selected_rows.h"
#include "paddle/phi/core/string_tensor.h"

namespace phi {
using DataType = phi::DataType;

struct DeviceContext::Impl {
  Impl() = default;
  ~Impl() = default;

  void SetAllocator(const Allocator* allocator) {
    PADDLE_ENFORCE_NOT_NULL(
        allocator,
        common::errors::InvalidArgument(
            "Required allocator shall not be nullptr, but received nullptr."));
    device_allocator_ = allocator;
  }

  void SetHostAllocator(const Allocator* allocator) {
    PADDLE_ENFORCE_NOT_NULL(
        allocator,
        common::errors::InvalidArgument(
            "Required allocator shall not be nullptr, but received nullptr."));
    host_allocator_ = allocator;
  }

  void SetZeroAllocator(const Allocator* allocator) {
    PADDLE_ENFORCE_NOT_NULL(
        allocator,
        common::errors::InvalidArgument(
            "Required allocator shall not be nullptr, but received nullptr."));
    zero_allocator_ = allocator;
  }

  void SetHostZeroAllocator(const Allocator* allocator) {
    PADDLE_ENFORCE_NOT_NULL(
        allocator,
        common::errors::InvalidArgument(
            "Required allocator shall not be nullptr, but received nullptr."));
    host_zero_allocator_ = allocator;
  }

  void SetPinnedAllocator(const Allocator* allocator) {
    PADDLE_ENFORCE_NOT_NULL(
        allocator,
        common::errors::InvalidArgument(
            "Required allocator shall not be nullptr, but received nullptr."));
    pinned_allocator_ = allocator;
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  void SetCUDAGraphAllocator(const Allocator* allocator) {
    // NOTE (Yuang): cuda graph allocator can be set to nullptr, so don't check
    // validation of the allocator here
    cuda_graph_allocator_ = allocator;
  }

  const Allocator& GetCUDAGraphAllocator() const {
    PADDLE_ENFORCE_NOT_NULL(cuda_graph_allocator_,
                            common::errors::InvalidArgument(
                                "Required cuda_graph_allocator_ shall not be "
                                "nullptr, but received nullptr."));
    return *cuda_graph_allocator_;
  }

  bool IsCUDAGraphAllocatorValid() const {
    return cuda_graph_allocator_ != nullptr;
  }
#endif

  const Allocator& GetAllocator() const {
    PADDLE_ENFORCE_NOT_NULL(device_allocator_,
                            common::errors::InvalidArgument(
                                "Required device_allocator_ shall not be "
                                "nullptr, but received nullptr."));
    return *device_allocator_;
  }

  const Allocator& GetHostAllocator() const {
    PADDLE_ENFORCE_NOT_NULL(
        host_allocator_,
        common::errors::InvalidArgument("Required host_allocator_ shall not be "
                                        "nullptr, but received nullptr."));
    return *host_allocator_;
  }

  const Allocator& GetZeroAllocator() const {
    PADDLE_ENFORCE_NOT_NULL(
        zero_allocator_,
        common::errors::InvalidArgument("Required zero_allocator_ shall not be "
                                        "nullptr, but received nullptr."));
    return *zero_allocator_;
  }

  const Allocator& GetHostZeroAllocator() const {
    PADDLE_ENFORCE_NOT_NULL(
        host_zero_allocator_,
        common::errors::InvalidArgument("Required zero_allocator_ shall not be "
                                        "nullptr, but received nullptr."));
    return *host_zero_allocator_;
  }

  const Allocator& GetPinnedAllocator() const {
    PADDLE_ENFORCE_NOT_NULL(pinned_allocator_,
                            common::errors::InvalidArgument(
                                "Required pinned_allocator_ shall not be "
                                "nullptr, but received nullptr."));
    return *pinned_allocator_;
  }

  void* Alloc(TensorBase* tensor,
              const Place& place,
              DataType dtype = DataType::UNDEFINED,
              size_t requested_size = 0,
              bool pinned = false,
              bool fake_alloc = false) const {
    PADDLE_ENFORCE_NOT_NULL(
        tensor,
        common::errors::InvalidArgument(
            "Required tensor shall not be nullptr, but received nullptr."));
    if (dtype == DataType::UNDEFINED) {
      dtype = tensor->dtype();
    }
    // NOTE(paddle-dev): In case of tensor has already hold allocation and
    // is going to allocate allocation on new place, we will clear its holder
    // firstly and then re-alloc it.
    if (phi::DenseTensor::classof(tensor)) {
      // NOTE(Ruibiao): The tensor hold zero-size allocation is not regarded as
      // `initialized`. Fix other tensor class when needed.
      if (static_cast<phi::DenseTensor*>(tensor)->Holder() &&
          tensor->place() != place) {
        ClearHolder(tensor);
      }
    } else {
      if (tensor->initialized() && tensor->place() != place) {
        ClearHolder(tensor);
      }
    }

    auto* allocator =
        (fake_alloc || tensor->numel() == 0) && requested_size == 0
            ? zero_allocator_
            : (pinned ? pinned_allocator_ : device_allocator_);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    bool must_cuda_graph_allocator =
        (!fake_alloc && tensor->numel() != 0) && !pinned;
    if (must_cuda_graph_allocator &&
        place.GetType() == phi::AllocationType::GPU &&
        phi::backends::gpu::CUDAGraph::IsThisThreadCapturing()) {
      PADDLE_ENFORCE_NOT_NULL(cuda_graph_allocator_,
                              common::errors::InvalidArgument(
                                  "Required cuda_graph_allocator_ shall not be "
                                  "nullptr, but received nullptr."));
      allocator = cuda_graph_allocator_;
    }
#endif
    return tensor->AllocateFrom(const_cast<Allocator*>(allocator),
                                dtype,
                                requested_size,
                                fake_alloc);  // NOLINT
  }

  template <typename T>
  T* Alloc(TensorBase* tensor,
           const Place& place,
           size_t requested_size = 0,
           bool pinned = false) const {
    DataType dtype = phi::CppTypeToDataType<T>::Type();
    return static_cast<T*>(Alloc(tensor, place, dtype, requested_size, pinned));
  }

  void* HostAlloc(TensorBase* tensor,
                  DataType dtype = DataType::UNDEFINED,
                  size_t requested_size = 0,
                  bool fake_alloc = false) const {
    PADDLE_ENFORCE_NOT_NULL(
        tensor,
        common::errors::InvalidArgument(
            "Required tensor shall not be nullptr, but received nullptr."));
    if (dtype == DataType::UNDEFINED) {
      dtype = tensor->dtype();
    }

    if (phi::DenseTensor::classof(tensor)) {
      // NOTE(Ruibiao): The tensor holds zero-size allocation is not regarded as
      // `initialized`. Fix other tensor class when needed.
      if (static_cast<phi::DenseTensor*>(tensor)->Holder() &&
          tensor->place() != CPUPlace()) {
        ClearHolder(tensor);
      }
    } else {
      if (tensor->initialized() && tensor->place() != CPUPlace()) {
        ClearHolder(tensor);
      }
    }

    auto* allocator =
        (fake_alloc || tensor->numel() == 0) && requested_size == 0
            ? host_zero_allocator_
            : host_allocator_;
    return tensor->AllocateFrom(const_cast<Allocator*>(allocator),
                                dtype,
                                requested_size,
                                fake_alloc);  // NOLINT
  }

  template <typename T>
  T* HostAlloc(phi::TensorBase* tensor, size_t requested_size = 0) const {
    DataType dtype = phi::CppTypeToDataType<T>::Type();
    return static_cast<T*>(HostAlloc(tensor, dtype, requested_size));
  }

  void SetGenerator(Generator* gen) {
    PADDLE_ENFORCE_NOT_NULL(
        gen,
        common::errors::InvalidArgument(
            "Required generator shall not be nullptr, but received nullptr."));
    device_generator_ = gen;
  }

  Generator* GetGenerator() const {
    PADDLE_ENFORCE_NOT_NULL(
        device_generator_,
        common::errors::InvalidArgument("Required generator_ shall not be "
                                        "nullptr, but received nullptr."));
    return device_generator_;
  }

  void SetHostGenerator(Generator* gen) {
    PADDLE_ENFORCE_NOT_NULL(
        gen,
        common::errors::InvalidArgument(
            "Required generator shall not be nullptr, but received nullptr."));
    host_generator_ = gen;
  }

  Generator* GetHostGenerator() const {
    PADDLE_ENFORCE_NOT_NULL(
        host_generator_,
        common::errors::InvalidArgument("Required generator_ shall not be "
                                        "nullptr, but received nullptr."));
    return host_generator_;
  }

  distributed::CommContext* GetCommContext() const { return comm_context_; }

  void SetCommContext(distributed::CommContext* comm_context) {
    comm_context_ = comm_context;
  }

 private:
  void ClearHolder(TensorBase* tensor) const {
    if (DenseTensor::classof(tensor)) {
      static_cast<DenseTensor*>(tensor)->clear();
    } else if (SelectedRows::classof(tensor)) {
      static_cast<SelectedRows*>(tensor)->mutable_value()->clear();
    } else if (StringTensor::classof(tensor)) {
      static_cast<StringTensor*>(tensor)->clear();
    } else {
      PADDLE_THROW(errors::Unimplemented(
          "Only support DenseTensor and SelectedRows now."));
    }
  }

  const Allocator* device_allocator_{nullptr};
  const Allocator* host_allocator_{nullptr};
  const Allocator* zero_allocator_{nullptr};
  const Allocator* host_zero_allocator_{nullptr};
  const Allocator* pinned_allocator_{nullptr};
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  const Allocator* cuda_graph_allocator_{nullptr};
#endif
  Generator* device_generator_{nullptr};
  Generator* host_generator_{nullptr};

  distributed::CommContext* comm_context_{nullptr};
};

DeviceContext::DeviceContext() { impl_ = std::make_unique<Impl>(); }

DeviceContext::DeviceContext(const DeviceContext& other) {
  impl_ = std::make_unique<Impl>();
  impl_->SetHostAllocator(&other.GetHostAllocator());
  impl_->SetAllocator(&other.GetAllocator());
  impl_->SetZeroAllocator(&other.GetZeroAllocator());
  impl_->SetHostZeroAllocator(&other.GetHostZeroAllocator());
  impl_->SetPinnedAllocator(&other.GetPinnedAllocator());
  impl_->SetHostGenerator(other.GetHostGenerator());
  impl_->SetGenerator(other.GetGenerator());
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (other.IsCUDAGraphAllocatorValid()) {
    impl_->SetCUDAGraphAllocator(&other.GetCUDAGraphAllocator());
  }
#endif
}

DeviceContext::DeviceContext(DeviceContext&& other) noexcept {
  impl_ = std::move(other.impl_);
}

DeviceContext& DeviceContext::operator=(DeviceContext&& other) noexcept =
    default;
DeviceContext::~DeviceContext() = default;

void DeviceContext::SetAllocator(const Allocator* allocator) {
  impl_->SetAllocator(allocator);
}

const Allocator& DeviceContext::GetAllocator() const {
  return impl_->GetAllocator();
}

void DeviceContext::SetHostAllocator(const Allocator* allocator) {
  impl_->SetHostAllocator(allocator);
}

const Allocator& DeviceContext::GetHostAllocator() const {
  return impl_->GetHostAllocator();
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
void DeviceContext::SetCUDAGraphAllocator(const Allocator* allocator) {
  impl_->SetCUDAGraphAllocator(allocator);
}

const Allocator& DeviceContext::GetCUDAGraphAllocator() const {
  return impl_->GetCUDAGraphAllocator();
}

bool DeviceContext::IsCUDAGraphAllocatorValid() const {
  return impl_->IsCUDAGraphAllocatorValid();
}
#endif

void DeviceContext::SetZeroAllocator(const Allocator* allocator) {
  impl_->SetZeroAllocator(allocator);
}

void DeviceContext::SetHostZeroAllocator(const Allocator* allocator) {
  impl_->SetHostZeroAllocator(allocator);
}

const Allocator& DeviceContext::GetZeroAllocator() const {
  return impl_->GetZeroAllocator();
}

const Allocator& DeviceContext::GetHostZeroAllocator() const {
  return impl_->GetHostZeroAllocator();
}

void DeviceContext::SetPinnedAllocator(const Allocator* allocator) {
  impl_->SetPinnedAllocator(allocator);
}
const Allocator& DeviceContext::GetPinnedAllocator() const {
  return impl_->GetPinnedAllocator();
}

void* DeviceContext::Alloc(TensorBase* tensor,
                           DataType dtype,
                           size_t requested_size,
                           bool pinned,
                           bool fake_alloc) const {
  if (pinned) {
    return impl_->Alloc(tensor,
                        GetPinnedPlace(GetPlace()),
                        dtype,
                        requested_size,
                        pinned,
                        fake_alloc);
  }
  return impl_->Alloc(
      tensor, GetPlace(), dtype, requested_size, pinned, fake_alloc);
}

template <typename T>
T* DeviceContext::Alloc(TensorBase* tensor,
                        size_t requested_size,
                        bool pinned) const {
  DataType dtype = phi::CppTypeToDataType<T>::Type();
  return static_cast<T*>(this->Alloc(tensor, dtype, requested_size, pinned));
}

void* DeviceContext::HostAlloc(TensorBase* tensor,
                               DataType dtype,
                               size_t requested_size,
                               bool fake_alloc) const {
  return impl_->HostAlloc(tensor, dtype, requested_size, fake_alloc);
}

template <typename T>
T* DeviceContext::HostAlloc(TensorBase* tensor, size_t requested_size) const {
  return impl_->HostAlloc<T>(tensor, requested_size);
}

#define DEVICE_CONTEXT_MEMBER_FUNC_INSTANTIATION(dtype)              \
  template TEST_API dtype* DeviceContext::Alloc(                     \
      TensorBase* tensor, size_t requested_size, bool pinned) const; \
  template dtype* DeviceContext::HostAlloc(TensorBase* tensor,       \
                                           size_t requested_size) const;

DEVICE_CONTEXT_MEMBER_FUNC_INSTANTIATION(bool)
DEVICE_CONTEXT_MEMBER_FUNC_INSTANTIATION(int8_t)
DEVICE_CONTEXT_MEMBER_FUNC_INSTANTIATION(uint8_t)
DEVICE_CONTEXT_MEMBER_FUNC_INSTANTIATION(int16_t)
DEVICE_CONTEXT_MEMBER_FUNC_INSTANTIATION(int32_t)
DEVICE_CONTEXT_MEMBER_FUNC_INSTANTIATION(int64_t)
DEVICE_CONTEXT_MEMBER_FUNC_INSTANTIATION(float)
DEVICE_CONTEXT_MEMBER_FUNC_INSTANTIATION(double)
DEVICE_CONTEXT_MEMBER_FUNC_INSTANTIATION(::phi::float8_e4m3fn)
DEVICE_CONTEXT_MEMBER_FUNC_INSTANTIATION(::phi::float8_e5m2)
DEVICE_CONTEXT_MEMBER_FUNC_INSTANTIATION(::phi::bfloat16)
DEVICE_CONTEXT_MEMBER_FUNC_INSTANTIATION(::phi::float16)
DEVICE_CONTEXT_MEMBER_FUNC_INSTANTIATION(::phi::complex64)
DEVICE_CONTEXT_MEMBER_FUNC_INSTANTIATION(::phi::complex128)
DEVICE_CONTEXT_MEMBER_FUNC_INSTANTIATION(::phi::pstring)

#undef DEVICE_CONTEXT_MEMBER_FUNC_INSTANTIATION

void DeviceContext::SetGenerator(Generator* gen) { impl_->SetGenerator(gen); }

Generator* DeviceContext::GetGenerator() const { return impl_->GetGenerator(); }

void DeviceContext::SetHostGenerator(Generator* gen) {
  impl_->SetHostGenerator(gen);
}

Generator* DeviceContext::GetHostGenerator() const {
  return impl_->GetHostGenerator();
}

void DeviceContext::SetCommContext(distributed::CommContext* comm_context) {
  impl_->SetCommContext(comm_context);
}

distributed::CommContext* DeviceContext::GetCommContext() const {
  return impl_->GetCommContext();
}

}  // namespace phi
