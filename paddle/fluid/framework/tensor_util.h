/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include <algorithm>
#include <codecvt>
#include <locale>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/dlpack_tensor.h"
#include "paddle/fluid/framework/string_array.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/memory/allocation/allocator_facade.h"
#ifdef PADDLE_WITH_ASCEND_CL
#include "paddle/fluid/memory/allocation/npu_pinned_allocator.h"
#endif
#include "paddle/fluid/platform/device_context.h"
#ifdef PADDLE_WITH_MLU
#include "paddle/fluid/platform/device/mlu/device_context.h"
#endif

#include "paddle/fluid/memory/memory.h"
#include "paddle/phi/core/dense_tensor.h"

namespace paddle {
namespace framework {

class PrintOptions {
 public:
  static PrintOptions& Instance() {
    static PrintOptions instance;
    return instance;
  }
  ~PrintOptions() {}
  PrintOptions(const PrintOptions& o) = delete;
  const PrintOptions& operator=(const PrintOptions& o) = delete;

  int precision = 8;
  int threshold = 1000;
  int edgeitems = 3;
  int linewidth = 75;
  bool sci_mode = false;

 private:
  PrintOptions() {}
};

void TensorToStream(std::ostream& os,
                    const phi::DenseTensor& tensor,
                    const platform::DeviceContext& dev_ctx);
void TensorFromStream(std::istream& is,
                      phi::DenseTensor* tensor,
                      const platform::DeviceContext& dev_ctx);
void TensorFromStream(std::istream& is,
                      phi::DenseTensor* tensor,
                      const platform::DeviceContext& dev_ctx,
                      const size_t& seek,
                      const std::vector<int64_t>& shape);

// NOTE(zcd): Because TensorCopy is an async operation, when the src_place
// and dst_place are two different GPU, to ensure that the operation can
// be carried out correctly, there is a src_ctx wait operation in TensorCopy.
// If ctx_place and src_place are the same, src_ctx.Wait() is added
// after memory::Copy; if ctx_place and dst_place are the same,
// src_ctx.Wait() is added before memory::Copy.
void TensorCopy(const phi::DenseTensor& src,
                const platform::Place& dst_place,
                const platform::DeviceContext& ctx,
                phi::DenseTensor* dst);

// NOTE(zcd): If the src.place() and dst_place are two different GPU,
// the copy operation is carried out on the dst_place's stream. This is
// very important, because TensorCopy is an async operator, and in most
// case, once this copy operator returns, dst is to be used in dst_place's
// stream, if this copy operation is carried out on the src_place's stream,
// when dst is used in dst_place's stream the copy operation may be
// not completed.
void TensorCopy(const phi::DenseTensor& src,
                const platform::Place& dst_place,
                phi::DenseTensor* dst);

void TensorCopySync(const phi::DenseTensor& src,
                    const platform::Place& dst_place,
                    phi::DenseTensor* dst);

template <typename T>
void TensorFromVector(const std::vector<T>& src,
                      const platform::DeviceContext& ctx,
                      phi::DenseTensor* dst);
template <typename T>
void TensorFromVector(const std::vector<T>& src, phi::DenseTensor* dst);

template <typename T>
void TensorToVector(const phi::DenseTensor& src,
                    const platform::DeviceContext& ctx,
                    std::vector<T>* dst);
template <typename T>
void TesnorToVector(const phi::DenseTensor& src, std::vector<T>* dst);

// convert dlpack's DLTensor to tensor
void TensorFromDLPack(const ::DLTensor& dl_tensor, phi::DenseTensor* dst);

//
// The implementation of template functions.
//

template <typename T>
void TensorFromArray(const T* src,
                     const size_t& array_size,
                     const platform::DeviceContext& ctx,
                     phi::DenseTensor* dst) {
  auto dst_place = ctx.GetPlace();
  auto src_ptr = static_cast<const void*>(src);
  platform::CPUPlace src_place;
  dst->Resize({static_cast<int64_t>(array_size)});
  auto dst_ptr = static_cast<void*>(dst->mutable_data<T>(dst_place));
  auto size = array_size * sizeof(T);

  if (platform::is_cpu_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  else if (platform::is_gpu_place(dst_place)) {  // NOLINT
    memory::Copy(dst_place,
                 dst_ptr,
                 src_place,
                 src_ptr,
                 size,
                 reinterpret_cast<const phi::GPUContext&>(ctx).stream());
  }
#endif
#ifdef PADDLE_WITH_ASCEND_CL
  else if (platform::is_npu_place(dst_place)) {  // NOLINT
    //  1. vector -> npu pinned tensor
    platform::NPUPinnedPlace npu_pinned_place;
    phi::DenseTensor npu_pinned_tensor;
    npu_pinned_tensor.Resize(dst->dims());
    auto npu_pinned_ptr =
        npu_pinned_tensor.mutable_data(npu_pinned_place, dst->dtype());
    memory::Copy(npu_pinned_place, npu_pinned_ptr, src_place, src_ptr, size);

    //  2. async copy npu pinned tensor -> npu tensor
    memory::Copy(
        dst_place,
        dst_ptr,
        npu_pinned_place,
        npu_pinned_ptr,
        size,
        reinterpret_cast<const platform::NPUDeviceContext&>(ctx).stream());

    //  3. record event
    auto npu_pinned_allocator =
        static_cast<paddle::memory::allocation::NPUPinnedAllocator*>(
            paddle::memory::allocation::AllocatorFacade::Instance()
                .GetAllocator(npu_pinned_place)
                .get());
    phi::Allocation* allocation = npu_pinned_tensor.Holder().get();
    npu_pinned_allocator->RecordEvent(
        allocation,
        reinterpret_cast<const platform::NPUDeviceContext&>(ctx).stream());
  }
#endif
#ifdef PADDLE_WITH_MLU
  else if (platform::is_mlu_place(dst_place)) {  // NOLINT
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size, nullptr);
  }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  else if (platform::is_custom_place(dst_place)) {  // NOLINT
    memory::Copy(
        dst_place,
        dst_ptr,
        src_place,
        src_ptr,
        size,
        reinterpret_cast<const platform::CustomDeviceContext&>(ctx).stream());
  }
#endif
  else {  // NOLINT
    PADDLE_THROW(platform::errors::Unimplemented(
        "TensorFromArray on %s is not supported.", dst_place));
  }
}

template <typename T>
void TensorFromVector(const std::vector<T>& src,
                      const platform::DeviceContext& ctx,
                      phi::DenseTensor* dst) {
  auto dst_place = ctx.GetPlace();
  auto src_ptr = static_cast<const void*>(src.data());
  platform::CPUPlace src_place;
  dst->Resize({static_cast<int64_t>(src.size())});
  auto dst_ptr = static_cast<void*>(dst->mutable_data<T>(dst_place));
  auto size = src.size() * sizeof(T);

  if (platform::is_cpu_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  else if (platform::is_gpu_place(dst_place)) {  // NOLINT
    memory::Copy(dst_place,
                 dst_ptr,
                 src_place,
                 src_ptr,
                 size,
                 reinterpret_cast<const phi::GPUContext&>(ctx).stream());
  }
#endif
#ifdef PADDLE_WITH_ASCEND_CL
  // NOTE(zhiqiu): Becareful that aclrtMemcpyAsync is different from
  // cudaMemcpyAsync.
  // cudaMemcpyAsync is actually "sync" between cpu <-> gpu.
  // aclrtMemcpyAsync is really "async" between cpu <-> npu.
  // Since vector is on cpu, I think this function should be a "sync" operation,
  // so pass nullptr as stream to  memory::Copy().
  else if (platform::is_npu_place(dst_place)) {  // NOLINT
    //  1. vector -> npu pinned tensor
    phi::DenseTensor npu_pinned_tensor(dst->dtype());
    platform::NPUPinnedPlace npu_pinned_place;
    auto npu_pinned_ptr =
        npu_pinned_tensor.mutable_data<T>(dst->dims(), npu_pinned_place);
    memory::Copy(npu_pinned_place, npu_pinned_ptr, src_place, src_ptr, size);

    //  2. async copy npu pinned tensor -> npu tensor
    memory::Copy(
        dst_place,
        dst_ptr,
        npu_pinned_place,
        npu_pinned_ptr,
        size,
        reinterpret_cast<const platform::NPUDeviceContext&>(ctx).stream());

    //  3. record event
    auto npu_pinned_allocator =
        static_cast<paddle::memory::allocation::NPUPinnedAllocator*>(
            paddle::memory::allocation::AllocatorFacade::Instance()
                .GetAllocator(npu_pinned_place)
                .get());
    phi::Allocation* allocation = npu_pinned_tensor.Holder().get();
    npu_pinned_allocator->RecordEvent(
        allocation,
        reinterpret_cast<const platform::NPUDeviceContext&>(ctx).stream());
  }
#endif
#ifdef PADDLE_WITH_MLU
  else if (platform::is_mlu_place(dst_place)) {  // NOLINT
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size, nullptr);
  }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  else if (platform::is_custom_place(dst_place)) {  // NOLINT
    memory::Copy(
        dst_place,
        dst_ptr,
        src_place,
        src_ptr,
        size,
        reinterpret_cast<const platform::CustomDeviceContext&>(ctx).stream());
  }
#endif
#ifdef PADDLE_WITH_XPU
  else if (platform::is_xpu_place(dst_place)) {  // NOLINT
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
#endif
  else {  // NOLINT
    PADDLE_THROW(platform::errors::Unimplemented(
        "TensorFromVector on %s is not supported.", dst_place));
  }
}

// The fully specialized function should be inline to avoid
// multi-definition.
template <>
inline void TensorFromVector(const std::vector<bool>& src,
                             const platform::DeviceContext& ctx,
                             phi::DenseTensor* dst) {
  // vector<bool> has no data() member, use array instead.
  // See details:
  // https://stackoverflow.com/questions/46115669/why-does-stdvectorbool-have-no-data/46115714
  bool* array = new bool[src.size()];
  for (unsigned int i = 0; i < src.size(); i++) {
    array[i] = static_cast<bool>(src[i]);
  }

  auto dst_place = ctx.GetPlace();
  auto src_ptr = static_cast<const void*>(array);
  platform::CPUPlace src_place;
  dst->Resize({static_cast<int64_t>(src.size())});
  auto dst_ptr = static_cast<void*>(dst->mutable_data<bool>(dst_place));
  auto size = src.size() * sizeof(bool);

  if (platform::is_cpu_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
#ifdef PADDLE_WITH_CUDA
  else if (platform::is_gpu_place(dst_place)) {  // NOLINT
    memory::Copy(dst_place,
                 dst_ptr,
                 src_place,
                 src_ptr,
                 size,
                 reinterpret_cast<const phi::GPUContext&>(ctx).stream());
  }
#endif
#ifdef PADDLE_WITH_ASCEND_CL
  else if (platform::is_npu_place(dst_place)) {  // NOLINT
    //  1. vector -> npu pinned tensor
    platform::NPUPinnedPlace npu_pinned_place;
    phi::DenseTensor npu_pinned_tensor;
    npu_pinned_tensor.Resize(dst->dims());
    auto npu_pinned_ptr =
        npu_pinned_tensor.mutable_data(npu_pinned_place, dst->dtype());
    memory::Copy(npu_pinned_place, npu_pinned_ptr, src_place, src_ptr, size);

    //  2. async copy npu pinned tensor -> npu tensor
    memory::Copy(
        dst_place,
        dst_ptr,
        npu_pinned_place,
        npu_pinned_ptr,
        size,
        reinterpret_cast<const platform::NPUDeviceContext&>(ctx).stream());

    //  3. record event
    auto npu_pinned_allocator =
        static_cast<paddle::memory::allocation::NPUPinnedAllocator*>(
            paddle::memory::allocation::AllocatorFacade::Instance()
                .GetAllocator(npu_pinned_place)
                .get());
    phi::Allocation* allocation = npu_pinned_tensor.Holder().get();
    npu_pinned_allocator->RecordEvent(
        allocation,
        reinterpret_cast<const platform::NPUDeviceContext&>(ctx).stream());
  }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEICE
  else if (platform::is_custom_place(dst_place)) {  // NOLINT
    auto stream =
        reinterpret_cast<const platform::CustomDeviceContext&>(ctx).stream();
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size, stream);
  }
#endif
#ifdef PADDLE_WITH_XPU
  else if (platform::is_xpu_place(dst_place)) {  // NOLINT
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
#endif
  else {  // NOLINT
    PADDLE_THROW(platform::errors::Unimplemented(
        "TensorFromVector on %s is not supported.", dst_place));
  }
  delete[] array;
}

template <typename T>
void TensorFromVector(const std::vector<T>& src, phi::DenseTensor* dst) {
  platform::CPUPlace dst_place = platform::CPUPlace();
  auto src_ptr = static_cast<const void*>(src.data());
  platform::CPUPlace src_place;
  dst->Resize({static_cast<int64_t>(src.size())});
  auto dst_ptr = static_cast<void*>(dst->mutable_data<T>(dst_place));
  auto size = src.size() * sizeof(T);

  memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
}

template <>
inline void TensorFromVector(const std::vector<bool>& src,
                             phi::DenseTensor* dst) {
  bool* array = new bool[src.size()];
  for (unsigned int i = 0; i < src.size(); i++) {
    array[i] = static_cast<bool>(src[i]);
  }
  platform::CPUPlace dst_place = platform::CPUPlace();
  auto src_ptr = static_cast<const void*>(array);
  platform::CPUPlace src_place;
  dst->Resize({static_cast<int64_t>(src.size())});
  auto dst_ptr = static_cast<void*>(dst->mutable_data<bool>(dst_place));
  auto size = src.size() * sizeof(bool);

  memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  delete[] array;
}

template <typename T>
void TensorToVector(const phi::DenseTensor& src,
                    const platform::DeviceContext& ctx,
                    std::vector<T>* dst) {
  auto src_ptr = static_cast<const void*>(src.data<T>());
  auto size = src.numel() * sizeof(T);

  platform::CPUPlace dst_place;
  dst->resize(src.numel());
  auto dst_ptr = static_cast<void*>(dst->data());

  if (platform::is_cpu_place(src.place())) {
    memory::Copy(dst_place, dst_ptr, src.place(), src_ptr, size);
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  else if (platform::is_gpu_place(src.place())) {  // NOLINT
    memory::Copy(dst_place,
                 dst_ptr,
                 src.place(),
                 src_ptr,
                 size,
                 reinterpret_cast<const phi::GPUContext&>(ctx).stream());
  }
#endif
#if defined(PADDLE_WITH_XPU)
  else if (platform::is_xpu_place(src.place())) {  // NOLINT
    memory::Copy(dst_place, dst_ptr, src.place(), src_ptr, size);
  }
#endif
#ifdef PADDLE_WITH_ASCEND_CL
  else if (platform::is_npu_place(src.place())) {  // NOLINT
    memory::Copy(dst_place, dst_ptr, src.place(), src_ptr, size, nullptr);
  }
#endif
#ifdef PADDLE_WITH_MLU
  else if (platform::is_mlu_place(src.place())) {  // NOLINT
    memory::Copy(
        dst_place,
        dst_ptr,
        src.place(),
        src_ptr,
        size,
        reinterpret_cast<const platform::MLUDeviceContext&>(ctx).stream());
  }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  else if (platform::is_custom_place(src.place())) {  // NOLINT
    memory::Copy(dst_place, dst_ptr, src.place(), src_ptr, size, nullptr);
  }
#endif
  else {  // NOLINT
    PADDLE_THROW(platform::errors::Unimplemented(
        "TensorToVector on %s is not supported.", src.place()));
  }
}

template <>
inline void TensorToVector(const phi::DenseTensor& src,
                           const platform::DeviceContext& ctx,
                           std::vector<bool>* dst) {
  auto src_ptr = static_cast<const void*>(src.data<bool>());
  auto size = src.numel() * sizeof(bool);

  bool* array = new bool[src.numel()];

  platform::CPUPlace dst_place;
  dst->resize(src.numel());
  auto dst_ptr = static_cast<void*>(array);

  if (platform::is_cpu_place(src.place())) {
    memory::Copy(dst_place, dst_ptr, src.place(), src_ptr, size);
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  else if (platform::is_gpu_place(src.place())) {  // NOLINT
    memory::Copy(dst_place,
                 dst_ptr,
                 src.place(),
                 src_ptr,
                 size,
                 reinterpret_cast<const phi::GPUContext&>(ctx).stream());
  }
#endif
#if defined(PADDLE_WITH_XPU)
  else if (platform::is_xpu_place(src.place())) {  // NOLINT
    memory::Copy(dst_place, dst_ptr, src.place(), src_ptr, size);
  }
#endif
#ifdef PADDLE_WITH_ASCEND_CL
  else if (platform::is_npu_place(src.place())) {  // NOLINT
    memory::Copy(dst_place, dst_ptr, src.place(), src_ptr, size, nullptr);
  }
#endif
#ifdef PADDLE_WITH_MLU
  else if (platform::is_mlu_place(src.place())) {  // NOLINT
    memory::Copy(dst_place, dst_ptr, src.place(), src_ptr, size, nullptr);
  }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  else if (platform::is_custom_place(src.place())) {  // NOLINT
    memory::Copy(dst_place, dst_ptr, src.place(), src_ptr, size, nullptr);
  }
#endif
  for (unsigned int i = 0; i < src.numel(); i++) {
    (*dst)[i] = static_cast<bool>(array[i]);
  }
  delete[] array;
}

template <typename T>
void TensorToVector(const phi::DenseTensor& src, std::vector<T>* dst) {
  auto src_ptr = static_cast<const void*>(src.data<T>());
  auto size = src.numel() * sizeof(T);

  platform::CPUPlace dst_place;
  dst->resize(src.numel());
  auto dst_ptr = static_cast<void*>(dst->data());

  PADDLE_ENFORCE_EQ(
      platform::is_cpu_place(src.place()),
      true,
      platform::errors::InvalidArgument(
          "The input tensor should be CPU device, but actually it is in %s.",
          src.place()));

  memory::Copy(dst_place, dst_ptr, src.place(), src_ptr, size);
}

template <>
inline void TensorToVector(const phi::DenseTensor& src,
                           std::vector<bool>* dst) {
  auto src_ptr = static_cast<const void*>(src.data<bool>());
  auto size = src.numel() * sizeof(bool);

  bool* array = new bool[src.numel()];

  platform::CPUPlace dst_place;
  dst->resize(src.numel());
  auto dst_ptr = static_cast<void*>(array);

  PADDLE_ENFORCE_EQ(
      platform::is_cpu_place(src.place()),
      true,
      platform::errors::InvalidArgument(
          "The input tensor should be CPU device, but actually it is in %s.",
          src.place()));

  memory::Copy(dst_place, dst_ptr, src.place(), src_ptr, size);

  for (unsigned int i = 0; i < src.numel(); i++) {
    (*dst)[i] = static_cast<bool>(array[i]);
  }
  delete[] array;
}

std::ostream& operator<<(std::ostream& os, const LoD& lod);

inline phi::DenseTensor ReshapeToMatrix(const phi::DenseTensor& src,
                                        int num_col_dims) {
  int rank = src.dims().size();
  PADDLE_ENFORCE_GE(
      rank,
      2,
      platform::errors::InvalidArgument(
          "'ReshapeToMatrix()' is only used for flatten high rank "
          "tensors to matrixs. The dimensions of phi::DenseTensor must be "
          "greater or equal than 2. "
          "But received dimensions of phi::DenseTensor is %d",
          rank));
  if (rank == 2) {
    return src;
  }
  phi::DenseTensor res;
  res.ShareDataWith(src);
  res.Resize(phi::flatten_to_2d(src.dims(), num_col_dims));
  return res;
}

template <typename T>
inline T GetValue(const phi::DenseTensor* x) {
  T value = static_cast<T>(0);
  if (!platform::is_cpu_place(x->place())) {
    phi::DenseTensor cpu_x;
    framework::TensorCopy(*x, platform::CPUPlace(), &cpu_x);
#if defined(PADDLE_WITH_ASCEND_CL) || defined(PADDLE_WITH_MLU)
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    const platform::DeviceContext* dev_ctx = pool.Get(x->place());
    dev_ctx->Wait();
#endif
    value = cpu_x.data<T>()[0];
  } else {
    value = x->data<T>()[0];
  }
  return value;
}

}  // namespace framework
}  // namespace paddle

namespace phi {
std::ostream& operator<<(std::ostream& os, const DenseTensor& t);
}
