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
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {

/**
 * @brief   Copy the content of external tensor to a new place.
 *
 * @param[in] src        The external tensor.
 * @param[in] dst_place  The dst place.
 * @param[in] ctx        The device context contains device resources.
 *
 * @note    Copy supports CPU <-> GPU, GPU <-> GPU.
 */
inline void Copy(const Tensor& src, const platform::Place& dst_place,
                 const platform::DeviceContext& ctx, Tensor* dst) {
  VLOG(3) << "Copy " << src.dims() << " from " << src.place() << " to "
          << dst_place;
  src.check_memory_size();

  dst->Resize(src.dims());
  dst->set_layout(src.layout());
  auto src_place = src.place();
  auto src_ptr = src.data<void>();

  auto dst_ptr = dst->mutable_data(dst_place, src.type());

  auto size = src.numel() * SizeOfType(src.type());

  if (platform::is_cpu_place(src_place) && platform::is_cpu_place(dst_place)) {
    memory::Copy(boost::get<platform::CPUPlace>(dst_place), dst_ptr,
                 boost::get<platform::CPUPlace>(src_place), src_ptr, size);
  }
#ifdef PADDLE_WITH_CUDA
  else if (platform::is_gpu_place(src_place) &&  // NOLINT
           platform::is_cpu_place(dst_place)) {
    auto src_gpu_place = boost::get<platform::CUDAPlace>(src_place);
    auto dst_cpu_place = boost::get<platform::CPUPlace>(dst_place);
    auto ctx_place = ctx.GetPlace();
    PADDLE_ENFORCE(platform::is_gpu_place(ctx_place));
    auto ctx_gpu_place = boost::get<platform::CUDAPlace>(ctx_place);
    PADDLE_ENFORCE_EQ(src_gpu_place, ctx_gpu_place);
    memory::Copy(
        dst_cpu_place, dst_ptr, src_gpu_place, src_ptr, size,
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream());
  } else if (platform::is_cpu_place(src_place) &&
             platform::is_gpu_place(dst_place)) {
    auto src_cpu_place = boost::get<platform::CPUPlace>(src_place);
    auto dst_gpu_place = boost::get<platform::CUDAPlace>(dst_place);
    auto ctx_place = ctx.GetPlace();
    PADDLE_ENFORCE(platform::is_gpu_place(ctx_place));
    auto ctx_gpu_place = boost::get<platform::CUDAPlace>(ctx_place);
    PADDLE_ENFORCE_EQ(dst_gpu_place, ctx_gpu_place);
    memory::Copy(
        dst_gpu_place, dst_ptr, src_cpu_place, src_ptr, size,
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream());
  } else if (platform::is_gpu_place(src_place) &&
             platform::is_gpu_place(dst_place)) {
    auto src_gpu_place = boost::get<platform::CUDAPlace>(src_place);
    auto dst_gpu_place = boost::get<platform::CUDAPlace>(dst_place);
    auto ctx_place = ctx.GetPlace();
    PADDLE_ENFORCE(platform::is_gpu_place(ctx_place));
    auto ctx_gpu_place = boost::get<platform::CUDAPlace>(ctx_place);
    PADDLE_ENFORCE_EQ(src_gpu_place, ctx_gpu_place);
    memory::Copy(
        dst_gpu_place, dst_ptr, src_gpu_place, src_ptr, size,
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream());
  }
#endif
}

/**
 * @brief Wrapper on
 *     Copy(const Tensor& src, const platform::Place& dst_place,
 *              const platform::DeviceContext& ctx, Tensor* dst);
 *
 * @param[in] src        The external tensor.
 * @param[in] dst_place  The dst place.
 *
 * @note    Copy supports CPU <-> GPU, GPU <-> GPU.
 */
inline void Copy(const Tensor& src, const platform::Place& dst_place,
                 Tensor* dst) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  const platform::DeviceContext* dev_ctx;
  if (platform::is_gpu_place(src.place())) {
    dev_ctx = pool.Get(src.place());
  } else {
    dev_ctx = pool.Get(dst_place);
  }
  Copy(src, dst_place, *dev_ctx, dst);
}

/**
 * @brief   Copy the content of an external vector to a tensor.
 *
 * @param[in] src        The external tensor.
 * @param[in] ctx        The device context contains device resources.
 *
 * * @note    CopyFromVector will resize dst to an 1D tensor with the same
 *            size as src.
 */
template <typename T>
inline void CopyFromVector(const std::vector<T>& src,
                           const platform::DeviceContext& ctx, Tensor* dst) {
  auto dst_place = ctx.GetPlace();
  auto src_ptr = static_cast<const void*>(src.data());
  platform::CPUPlace src_place;
  dst->Resize({static_cast<int64_t>(src.size())});
  auto dst_ptr = static_cast<void*>(dst->mutable_data<T>(dst_place));
  auto size = src.size() * sizeof(T);

  if (platform::is_cpu_place(dst_place)) {
    memory::Copy(boost::get<platform::CPUPlace>(dst_place), dst_ptr, src_place,
                 src_ptr, size);
  }
#ifdef PADDLE_WITH_CUDA
  else if (platform::is_gpu_place(dst_place)) {  // NOLINT
    memory::Copy(
        boost::get<platform::CUDAPlace>(dst_place), dst_ptr, src_place, src_ptr,
        size,
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream());
  }
#endif
}

/**
 * @brief CopyFromVector CPU vector -> CPU Tensor
 */
template <typename T>
inline void CopyFromVector(const std::vector<T>& src, Tensor* dst) {
  platform::CPUPlace dst_place = platform::CPUPlace();
  auto src_ptr = static_cast<const void*>(src.data());
  platform::CPUPlace src_place;
  dst->Resize({static_cast<int64_t>(src.size())});
  auto dst_ptr = static_cast<void*>(dst->mutable_data<T>(dst_place));
  auto size = src.size() * sizeof(T);

  memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
}

/**
 * @brief   Copy the content of a tensor to a vector
 *
 * @param[in] src        The external tensor.
 * @param[in] ctx        The device context contains device resources.
 *
 * * @note    CopyFromVector assumes that the tensor has been resized
 *            before invoking.
 */
template <typename T>
inline void CopyToVector(const Tensor& src, const platform::DeviceContext& ctx,
                         std::vector<T>* dst) {
  auto src_ptr = static_cast<const void*>(src.data<T>());
  auto size = src.numel() * sizeof(T);

  platform::CPUPlace dst_place;
  dst->resize(src.numel());
  auto dst_ptr = static_cast<void*>(dst->data());

  if (platform::is_cpu_place(src.place())) {
    memory::Copy(dst_place, dst_ptr,
                 boost::get<platform::CPUPlace>(src.place()), src_ptr, size);
  }
#ifdef PADDLE_WITH_CUDA
  else if (platform::is_gpu_place(src.place())) {  // NOLINT
    memory::Copy(
        dst_place, dst_ptr, boost::get<platform::CUDAPlace>(src.place()),
        src_ptr, size,
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream());
  }
#endif
}

/**
 * @brief CopyToVector CPUTensor <-> CPU Vector
 */
template <typename T>
inline void CopyToVector(const Tensor& src, std::vector<T>* dst) {
  auto src_ptr = static_cast<const void*>(src.data<T>());
  auto size = src.numel() * sizeof(T);

  platform::CPUPlace dst_place;
  dst->resize(src.numel());
  auto dst_ptr = static_cast<void*>(dst->data());

  PADDLE_ENFORCE(platform::is_cpu_place(src.place()));

  memory::Copy(dst_place, dst_ptr, boost::get<platform::CPUPlace>(src.place()),
               src_ptr, size);
}

// Returns true if a tensor contains NAN, i.e., Not A Number.
bool HasNAN(const framework::Tensor& tensor);

// Returns true if a tensor contains Inf, i.e., Infinity.
bool HasInf(const framework::Tensor& tensor);

inline void SerializeToStream(std::ostream& os, const Tensor& tensor,
                              const platform::DeviceContext& dev_ctx) {
  // TODO(typhoonzero): serialize to ostream
  {  // the 1st field, uint32_t version
    constexpr uint32_t version = 0;
    os.write(reinterpret_cast<const char*>(&version), sizeof(version));
  }
  {  // the 2nd field, tensor description
     // int32_t  size
     // void*    protobuf message
    proto::TensorDesc desc;
    desc.set_data_type(framework::ToDataType(tensor.type()));
    auto dims = framework::vectorize(tensor.dims());
    auto* pb_dims = desc.mutable_dims();
    pb_dims->Resize(static_cast<int>(dims.size()), 0);
    std::copy(dims.begin(), dims.end(), pb_dims->begin());
    int32_t size = desc.ByteSize();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    auto out = desc.SerializeAsString();
    os.write(out.data(), size);
  }
  {  // the 3rd field, tensor data
    uint64_t size = tensor.memory_size();
    auto* data_ptr = tensor.data<void>();
    PADDLE_ENFORCE(size < std::numeric_limits<std::streamsize>::max(),
                   "Index overflow when writing tensor");
    if (platform::is_gpu_place(tensor.place())) {
#ifdef PADDLE_WITH_CUDA
      constexpr size_t kBufSize = 1024 * 1024 * 64;  // 64MB
      std::unique_ptr<char[]> buf(new char[kBufSize]);
      auto& gpu_dev_ctx =
          static_cast<const platform::CUDADeviceContext&>(dev_ctx);
      platform::CPUPlace cpu;
      uintptr_t data = reinterpret_cast<uintptr_t>(data_ptr);
      while (size != 0) {
        size_t size_to_write = std::min(kBufSize, static_cast<size_t>(size));
        memory::Copy(cpu, buf.get(),
                     boost::get<platform::CUDAPlace>(tensor.place()),
                     reinterpret_cast<const void*>(data), size_to_write,
                     gpu_dev_ctx.stream());
        gpu_dev_ctx.Wait();
        os.write(buf.get(), size_to_write);
        data += size_to_write;
        size -= size_to_write;
      }
#else
      PADDLE_THROW("Unexpected branch");
#endif
    } else {
      os.write(static_cast<const char*>(data_ptr),
               static_cast<std::streamsize>(size));
    }
  }
}

struct DeserializedDataFunctor {
  DeserializedDataFunctor(void** buf, Tensor* tensor,
                          const platform::Place& place)
      : buf_(buf), tensor_(tensor), place_(place) {}

  template <typename T>
  void operator()() {
    *buf_ = tensor_->mutable_data<T>(place_);
  }

  void** buf_;
  Tensor* tensor_;
  platform::Place place_;
};

inline void DeserializeFromStream(std::istream& is, Tensor* tensor,
                                  const platform::DeviceContext& dev_ctx) {
  uint32_t version;
  is.read(reinterpret_cast<char*>(&version), sizeof(version));
  PADDLE_ENFORCE_EQ(version, 0U, "Only version 0 is supported");
  proto::TensorDesc desc;
  {  // int32_t size
     // proto buffer
    int32_t size;
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    std::unique_ptr<char[]> buf(new char[size]);
    is.read(reinterpret_cast<char*>(buf.get()), size);
    PADDLE_ENFORCE(desc.ParseFromArray(buf.get(), size),
                   "Cannot parse tensor desc");
  }
  {  // read tensor
    std::vector<int64_t> dims;
    dims.reserve(static_cast<size_t>(desc.dims().size()));
    std::copy(desc.dims().begin(), desc.dims().end(), std::back_inserter(dims));
    tensor->Resize(framework::make_ddim(dims));
    void* buf;
    auto ctx = platform::CPUDeviceContext();
    if (platform::is_gpu_place(dev_ctx.GetPlace())) {
#ifdef PADDLE_WITH_CUDA
      Tensor cpu_tensor;
      cpu_tensor.Resize(framework::make_ddim(dims));
      framework::VisitDataType(
          desc.data_type(),
          DeserializedDataFunctor(&buf, &cpu_tensor, ctx.GetPlace()));
      is.read(static_cast<char*>(buf), cpu_tensor.memory_size());
      auto dst_place = dev_ctx.GetPlace();
      framework::Copy(cpu_tensor, dst_place, dev_ctx, tensor);
#else
      PADDLE_THROW("Unexpected branch");
#endif
    } else {
      framework::VisitDataType(
          desc.data_type(),
          DeserializedDataFunctor(&buf, tensor, ctx.GetPlace()));
      is.read(static_cast<char*>(buf), tensor->memory_size());
    }
  }
}

}  // namespace framework
}  // namespace paddle
