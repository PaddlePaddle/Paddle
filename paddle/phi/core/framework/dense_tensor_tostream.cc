// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/framework/dense_tensor_tostream.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/framework/convert_utils.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/contiguous_kernel.h"

namespace phi {

namespace proto = paddle::framework::proto;

template <typename Context>
phi::DenseTensor InnerTensorContiguous(const Context& dev_ctx,
                                       const phi::DenseTensor& tensor) {
  phi::DenseTensor dense_out;
  phi::MetaTensor meta_input(tensor);
  phi::MetaTensor meta_out(&dense_out);
  UnchangedInferMeta(meta_input, &meta_out);

  PD_VISIT_ALL_TYPES(tensor.dtype(), "InnerTensorContiguous", ([&] {
                       phi::ContiguousKernel<data_t, Context>(
                           dev_ctx, tensor, &dense_out);
                     }));
  return dense_out;
}

phi::DenseTensor InnerTensorContiguous(const phi::DenseTensor& tensor) {
  auto& pool = phi::DeviceContextPool::Instance();

  if (tensor.place().GetType() == phi::AllocationType::CPU) {
    auto* dev_ctx = static_cast<phi::CPUContext*>(pool.Get(tensor.place()));
    return InnerTensorContiguous<phi::CPUContext>(*dev_ctx, tensor);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  } else if (tensor.place().GetType() == phi::AllocationType::GPU) {
    auto* dev_ctx = static_cast<phi::GPUContext*>(pool.Get(tensor.place()));
    return InnerTensorContiguous<phi::GPUContext>(*dev_ctx, tensor);
#endif
#ifdef PADDLE_WITH_XPU
  } else if (tensor.place().GetType() == phi::AllocationType::XPU) {
    auto* dev_ctx = static_cast<phi::XPUContext*>(pool.Get(tensor.place()));
    return InnerTensorContiguous<phi::XPUContext>(*dev_ctx, tensor);
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  } else if (tensor.place().GetType() == phi::AllocationType::CUSTOM) {
    auto* dev_ctx = static_cast<phi::CustomContext*>(pool.Get(tensor.place()));
    phi::DenseTensor dense_out;
    phi::MetaTensor meta_input(tensor);
    phi::MetaTensor meta_out(&dense_out);
    UnchangedInferMeta(meta_input, &meta_out);
    const phi::KernelKey& kernel_key = {phi::TransToPhiBackend(tensor.place()),
                                        phi::DataLayout::ALL_LAYOUT,
                                        tensor.dtype()};
    using kernel_signature = void (*)(
        const phi::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
    PD_VISIT_KERNEL("contiguous",
                    kernel_key,
                    kernel_signature,
                    false,
                    *dev_ctx,
                    tensor,
                    &dense_out);
    return dense_out;
#endif
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Place type is not supported when casting data type."));
  }

  return tensor;
}

void TensorToStream(std::ostream& os,
                    const phi::DenseTensor& tensor,
                    const phi::DeviceContext& dev_ctx) {
  const auto ensure_contiguous = [](const phi::DenseTensor& tensor) {
    if (tensor.meta().is_contiguous()) {
      return tensor;
    }
    return InnerTensorContiguous(tensor);
  };
  const phi::DenseTensor& contiguous_tensor = ensure_contiguous(tensor);
  {  // the 1st field, uint32_t version
    constexpr uint32_t version = 0;
    os.write(reinterpret_cast<const char*>(&version), sizeof(version));
  }
  {  // the 2nd field, tensor description
     // int32_t  size
     // void*    protobuf message
    proto::VarType::TensorDesc desc;
    desc.set_data_type(
        TransToProtoVarTypeReturnType(contiguous_tensor.dtype()));
    auto dims = common::vectorize(contiguous_tensor.dims());
    auto* pb_dims = desc.mutable_dims();
    pb_dims->Resize(static_cast<int>(dims.size()), 0);
    std::copy(dims.begin(), dims.end(), pb_dims->begin());
    int32_t size = desc.ByteSize();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    auto out = desc.SerializeAsString();
    os.write(out.data(), size);
  }
  {  // the 3rd field, tensor data
    uint64_t size =
        contiguous_tensor.numel() * phi::SizeOf(contiguous_tensor.dtype());

    auto* data_ptr = contiguous_tensor.data();
    PADDLE_ENFORCE_LT(size,
                      (std::numeric_limits<std::streamsize>::max)(),
                      common::errors::ResourceExhausted(
                          "tensor size %d overflow when writing tensor", size));
    if (phi::is_gpu_place(contiguous_tensor.place())) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      constexpr size_t kBufSize = 1024 * 1024 * 64;  // 64MB
      std::unique_ptr<char[]> buf(new char[kBufSize]);
      auto& gpu_dev_ctx = static_cast<const phi::GPUContext&>(dev_ctx);
      phi::CPUPlace cpu;
      uintptr_t data = reinterpret_cast<uintptr_t>(data_ptr);
      while (size != 0) {
        size_t size_to_write = std::min(kBufSize, static_cast<size_t>(size));
        phi::memory_utils::Copy(cpu,
                                buf.get(),
                                contiguous_tensor.place(),
                                reinterpret_cast<const void*>(data),  // NOLINT
                                size_to_write,
                                gpu_dev_ctx.stream());
        gpu_dev_ctx.Wait();
        os.write(buf.get(), size_to_write);
        data += size_to_write;
        size -= size_to_write;
      }
#else
      PADDLE_THROW(common::errors::Unimplemented(
          "CUDAPlace is not supported when not compiled with CUDA"));
#endif
    } else if (phi::is_xpu_place(contiguous_tensor.place())) {
#ifdef PADDLE_WITH_XPU
      constexpr size_t kBufSize = 1024 * 1024 * 64;  // 64MB
      std::unique_ptr<char[]> buf(new char[kBufSize]);
      auto& xpu_dev_ctx = static_cast<const phi::XPUContext&>(dev_ctx);
      phi::CPUPlace cpu;
      uintptr_t data = reinterpret_cast<uintptr_t>(data_ptr);
      while (size != 0) {
        size_t size_to_write = std::min(kBufSize, static_cast<size_t>(size));
        phi::memory_utils::Copy(cpu,
                                buf.get(),
                                contiguous_tensor.place(),
                                reinterpret_cast<const void*>(data),
                                size_to_write);
        xpu_dev_ctx.Wait();
        os.write(buf.get(), size_to_write);
        data += size_to_write;
        size -= size_to_write;
      }
#else
      PADDLE_THROW(common::errors::Unimplemented(
          "XPUPlace is not supported when not compiled with XPU"));
#endif
    } else if (phi::is_custom_place(contiguous_tensor.place())) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
      constexpr size_t kBufSize = 1024 * 1024 * 64;     // 64MB
      std::unique_ptr<char[]> buf(new char[kBufSize]);  // NOLINT
      auto& custom_device_context =
          static_cast<const phi::CustomContext&>(dev_ctx);
      phi::CPUPlace cpu;
      uintptr_t data = reinterpret_cast<uintptr_t>(data_ptr);
      while (size != 0) {
        size_t size_to_write = std::min(kBufSize, static_cast<size_t>(size));
        phi::memory_utils::Copy(cpu,
                                buf.get(),
                                contiguous_tensor.place(),
                                reinterpret_cast<const void*>(data),
                                size_to_write,
                                custom_device_context.stream());
        custom_device_context.Wait();
        os.write(buf.get(), size_to_write);
        data += size_to_write;
        size -= size_to_write;
      }
#else
      PADDLE_THROW(common::errors::Unimplemented(
          "CustomPlace is not supported when not compiled with "
          "CustomDevice"));
#endif
    } else {
      os.write(static_cast<const char*>(data_ptr),
               static_cast<std::streamsize>(size));
    }
  }
}

struct DeserializedDataFunctor {
  DeserializedDataFunctor(void** buf,
                          phi::DenseTensor* tensor,
                          const phi::Place& place)
      : buf_(buf), tensor_(tensor), place_(place) {}

  template <typename T>
  void apply() {
    auto& pool = phi::DeviceContextPool::Instance();
    auto* dev_ctx = pool.Get(place_);
    *buf_ = dev_ctx->Alloc<T>(tensor_);
  }

  void** buf_;
  phi::DenseTensor* tensor_;
  phi::Place place_;
};

void TensorFromStream(std::istream& is,
                      phi::DenseTensor* tensor,
                      const phi::DeviceContext& dev_ctx,
                      const size_t& seek,
                      const std::vector<int64_t>& shape) {
  uint32_t version = 0;
  is.read(reinterpret_cast<char*>(&version), sizeof(version));

  PADDLE_ENFORCE_EQ(
      version,
      0U,
      common::errors::InvalidArgument(
          "tensor version %u is not supported, Only version 0 is supported",
          version));

  proto::VarType::TensorDesc desc;
  {  // int32_t size
    // proto buffer
    int32_t size = 0;
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    std::unique_ptr<char[]> buf(new char[size]);  // NOLINT
    is.read(reinterpret_cast<char*>(buf.get()), size);
    PADDLE_ENFORCE_EQ(
        desc.ParseFromArray(buf.get(), size),
        true,
        common::errors::InvalidArgument("Cannot parse tensor desc"));
  }
  {  // read tensor
    tensor->Resize(common::make_ddim(shape));
    size_t seekg = seek * SizeOfType(desc.data_type());
    is.seekg(seekg, is.cur);  // NOLINT

    void* buf = nullptr;
    phi::CPUContext ctx;
    size_t size = tensor->numel() * SizeOfType(desc.data_type());
    if (phi::is_gpu_place(dev_ctx.GetPlace()) ||
        phi::is_xpu_place(dev_ctx.GetPlace()) ||
        phi::is_custom_place(dev_ctx.GetPlace())) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP) || \
    defined(PADDLE_WITH_XPU) || defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DenseTensor cpu_tensor;
      cpu_tensor.Resize(common::make_ddim(shape));
      VisitDataType(desc.data_type(),
                    DeserializedDataFunctor(&buf, &cpu_tensor, ctx.GetPlace()));
      is.read(static_cast<char*>(buf), size);  // NOLINT
      auto dst_place = dev_ctx.GetPlace();
      phi::Copy(dev_ctx, cpu_tensor, dst_place, false, tensor);
      if (phi::is_custom_place(dev_ctx.GetPlace())) {
        dev_ctx.Wait();
      }
#else
      if (phi::is_gpu_place(dev_ctx.GetPlace())) {
        PADDLE_THROW(common::errors::Unimplemented(
            "CUDAPlace is not supported when not compiled with CUDA"));
      } else if (phi::is_xpu_place(dev_ctx.GetPlace())) {
        PADDLE_THROW(common::errors::Unimplemented(
            "XPUPlace is not supported when not compiled with XPU"));
      }
#endif
    } else {
      VisitDataType(desc.data_type(),
                    DeserializedDataFunctor(&buf, tensor, ctx.GetPlace()));
      is.read(static_cast<char*>(buf), size);  // NOLINT
    }
  }
}

void TensorFromStream(std::istream& is,
                      phi::DenseTensor* tensor,
                      const phi::DeviceContext& dev_ctx) {
  uint32_t version = 0;
  is.read(reinterpret_cast<char*>(&version), sizeof(version));
  PADDLE_ENFORCE_EQ(
      version,
      0U,
      common::errors::InvalidArgument(
          "tensor version %u is not supported, Only version 0 is supported",
          version));
  proto::VarType::TensorDesc desc;
  {  // int32_t size
     // proto buffer
    int32_t size = -1;
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    PADDLE_ENFORCE_EQ(
        is.good(),
        true,
        common::errors::Unavailable("Cannot read tensor desc size"));
    PADDLE_ENFORCE_GE(size,
                      0,
                      common::errors::InvalidArgument(
                          "phi::DenseTensor desc size should >= 0"));
    std::unique_ptr<char[]> buf(new char[size]);  // NOLINT
    is.read(reinterpret_cast<char*>(buf.get()), size);
    PADDLE_ENFORCE_EQ(
        desc.ParseFromArray(buf.get(), size),
        true,
        common::errors::InvalidArgument("Cannot parse tensor desc"));
  }
  {  // read tensor
    std::vector<int64_t> dims;
    dims.reserve(static_cast<size_t>(desc.dims().size()));
    std::copy(desc.dims().begin(), desc.dims().end(), std::back_inserter(dims));
    tensor->Resize(common::make_ddim(dims));
    void* buf = nullptr;
    phi::CPUContext ctx;
    size_t size = tensor->numel() * SizeOfType(desc.data_type());
    if (phi::is_gpu_place(dev_ctx.GetPlace()) ||
        phi::is_xpu_place(dev_ctx.GetPlace()) ||
        phi::is_custom_place(dev_ctx.GetPlace())) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP) || \
    defined(PADDLE_WITH_XPU) || defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DenseTensor cpu_tensor;
      cpu_tensor.Resize(common::make_ddim(dims));
      VisitDataType(desc.data_type(),
                    DeserializedDataFunctor(&buf, &cpu_tensor, ctx.GetPlace()));
      is.read(static_cast<char*>(buf), size);  // NOLINT
      auto dst_place = dev_ctx.GetPlace();
      phi::Copy(dev_ctx, cpu_tensor, dst_place, false, tensor);
      if (phi::is_custom_place(dev_ctx.GetPlace())) {
        dev_ctx.Wait();
      }
#else
      if (phi::is_gpu_place(dev_ctx.GetPlace())) {
        PADDLE_THROW(common::errors::Unimplemented(
            "CUDAPlace is not supported when not compiled with CUDA"));
      } else if (phi::is_xpu_place(dev_ctx.GetPlace())) {
        PADDLE_THROW(common::errors::Unimplemented(
            "XPUPlace is not supported when not compiled with XPU"));
      } else {
        PADDLE_THROW(
            common::errors::Unimplemented("CustomPlace is not supported when "
                                          "not compiled with CustomDevice"));
      }
#endif
    } else {
      VisitDataType(desc.data_type(),
                    DeserializedDataFunctor(&buf, tensor, ctx.GetPlace()));
      is.read(static_cast<char*>(buf), size);  // NOLINT
    }
  }
}

}  // namespace phi
