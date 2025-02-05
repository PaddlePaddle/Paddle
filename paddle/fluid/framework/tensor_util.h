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
#include "paddle/fluid/framework/tensor.h"
#include "paddle/phi/core/memory/allocation/allocator_facade.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/phi/core/vocab/string_array.h"

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/memory/memory.h"

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

phi::DataType ConvertToPDDataType(const std::string& typestr);

TEST_API void TensorToStream(std::ostream& os,
                             const phi::DenseTensor& tensor,
                             const phi::DeviceContext& dev_ctx);
TEST_API void TensorFromStream(std::istream& is,
                               phi::DenseTensor* tensor,
                               const phi::DeviceContext& dev_ctx);
void TensorFromStream(std::istream& is,
                      phi::DenseTensor* tensor,
                      const phi::DeviceContext& dev_ctx,
                      const size_t& seek,
                      const std::vector<int64_t>& shape);

// NOTE(zcd): Because TensorCopy is an async operation, when the src_place
// and dst_place are two different GPU, to ensure that the operation can
// be carried out correctly, there is a src_ctx wait operation in TensorCopy.
// If ctx_place and src_place are the same, src_ctx.Wait() is added
// after memory::Copy; if ctx_place and dst_place are the same,
// src_ctx.Wait() is added before memory::Copy.
TEST_API void TensorCopy(const phi::DenseTensor& src,
                         const phi::Place& dst_place,
                         const phi::DeviceContext& ctx,
                         phi::DenseTensor* dst);

// NOTE(zcd): If the src.place() and dst_place are two different GPU,
// the copy operation is carried out on the dst_place's stream. This is
// very important, because TensorCopy is an async operator, and in most
// case, once this copy operator returns, dst is to be used in dst_place's
// stream, if this copy operation is carried out on the src_place's stream,
// when dst is used in dst_place's stream the copy operation may be
// not completed.
TEST_API void TensorCopy(const phi::DenseTensor& src,
                         const phi::Place& dst_place,
                         phi::DenseTensor* dst);

TEST_API void TensorCopySync(const phi::DenseTensor& src,
                             const phi::Place& dst_place,
                             phi::DenseTensor* dst);

template <typename T>
void TensorFromVector(const std::vector<T>& src,
                      const phi::DeviceContext& ctx,
                      phi::DenseTensor* dst);
template <typename T>
void TensorFromVector(const std::vector<T>& src, phi::DenseTensor* dst);

template <typename T>
void TensorToVector(const phi::DenseTensor& src,
                    const phi::DeviceContext& ctx,
                    std::vector<T>* dst);
template <typename T>
void TensorToVector(const phi::DenseTensor& src, std::vector<T>* dst);

// convert dlpack's DLTensor to tensor
TEST_API void TensorFromDLPack(const ::DLTensor& dl_tensor,
                               phi::DenseTensor* dst);

TEST_API phi::DenseTensor TensorFromDLPack(DLManagedTensor* src);
inline phi::DenseTensor TensorFromDLPack(const DLManagedTensor* src) {
  return TensorFromDLPack(const_cast<DLManagedTensor*>(src));
}

phi::DenseTensor TensorFromDLPack(DLManagedTensor* src,
                                  std::function<void(void*)> deleter);
//
// The implementation of template functions.
//

template <typename T>
void TensorFromArray(const T* src,
                     const size_t& array_size,
                     const phi::DeviceContext& ctx,
                     phi::DenseTensor* dst) {
  auto dst_place = ctx.GetPlace();
  auto src_ptr = static_cast<const void*>(src);
  phi::CPUPlace src_place;
  dst->Resize({static_cast<int64_t>(array_size)});
  auto dst_ptr = static_cast<void*>(dst->mutable_data<T>(dst_place));
  auto size = array_size * sizeof(T);

  if (phi::is_cpu_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  else if (phi::is_gpu_place(dst_place)) {  // NOLINT
    memory::Copy(dst_place,
                 dst_ptr,
                 src_place,
                 src_ptr,
                 size,
                 reinterpret_cast<const phi::GPUContext&>(ctx).stream());
  }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  else if (phi::is_custom_place(dst_place)) {  // NOLINT
    memory::Copy(dst_place,
                 dst_ptr,
                 src_place,
                 src_ptr,
                 size,
                 reinterpret_cast<const phi::CustomContext&>(ctx).stream());
  }
#endif
#ifdef PADDLE_WITH_XPU
  else if (phi::is_xpu_place(dst_place)) {  // NOLINT
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
#endif
  else {  // NOLINT
    PADDLE_THROW(common::errors::Unimplemented(
        "TensorFromArray on %s is not supported.", dst_place));
  }
}

template <typename T>
void TensorFromVector(const std::vector<T>& src,
                      const phi::DeviceContext& ctx,
                      phi::DenseTensor* dst) {
  auto dst_place = ctx.GetPlace();
  auto src_ptr = static_cast<const void*>(src.data());
  phi::CPUPlace src_place;
  dst->Resize({static_cast<int64_t>(src.size())});
  auto dst_ptr = static_cast<void*>(dst->mutable_data<T>(dst_place));
  auto size = src.size() * sizeof(T);

  if (phi::is_cpu_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  else if (phi::is_gpu_place(dst_place)) {  // NOLINT
    memory::Copy(dst_place,
                 dst_ptr,
                 src_place,
                 src_ptr,
                 size,
                 reinterpret_cast<const phi::GPUContext&>(ctx).stream());
  }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  else if (phi::is_custom_place(dst_place)) {  // NOLINT
    memory::Copy(dst_place,
                 dst_ptr,
                 src_place,
                 src_ptr,
                 size,
                 reinterpret_cast<const phi::CustomContext&>(ctx).stream());
  }
#endif
#ifdef PADDLE_WITH_XPU
  else if (phi::is_xpu_place(dst_place)) {  // NOLINT
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
#endif
  else {  // NOLINT
    PADDLE_THROW(common::errors::Unimplemented(
        "TensorFromVector on %s is not supported.", dst_place));
  }
}

// The fully specialized function should be inline to avoid
// multi-definition.
template <>
inline void TensorFromVector(const std::vector<bool>& src,
                             const phi::DeviceContext& ctx,
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
  phi::CPUPlace src_place;
  dst->Resize({static_cast<int64_t>(src.size())});
  auto dst_ptr = static_cast<void*>(dst->mutable_data<bool>(dst_place));
  auto size = src.size() * sizeof(bool);

  if (phi::is_cpu_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
#ifdef PADDLE_WITH_CUDA
  else if (phi::is_gpu_place(dst_place)) {  // NOLINT
    memory::Copy(dst_place,
                 dst_ptr,
                 src_place,
                 src_ptr,
                 size,
                 reinterpret_cast<const phi::GPUContext&>(ctx).stream());
  }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  else if (phi::is_custom_place(dst_place)) {  // NOLINT
    auto stream = reinterpret_cast<const phi::CustomContext&>(ctx).stream();
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size, stream);
  }
#endif
#ifdef PADDLE_WITH_XPU
  else if (phi::is_xpu_place(dst_place)) {  // NOLINT
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
#endif
  else {  // NOLINT
    PADDLE_THROW(common::errors::Unimplemented(
        "TensorFromVector on %s is not supported.", dst_place));
  }
  delete[] array;
}

template <typename T>
void TensorFromVector(const std::vector<T>& src, phi::DenseTensor* dst) {
  phi::CPUPlace dst_place = phi::CPUPlace();
  auto src_ptr = static_cast<const void*>(src.data());
  phi::CPUPlace src_place;
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
  phi::CPUPlace dst_place = phi::CPUPlace();
  auto src_ptr = static_cast<const void*>(array);
  phi::CPUPlace src_place;
  dst->Resize({static_cast<int64_t>(src.size())});
  auto dst_ptr = static_cast<void*>(dst->mutable_data<bool>(dst_place));
  auto size = src.size() * sizeof(bool);

  memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  delete[] array;
}

template <typename T>
void TensorToVector(const phi::DenseTensor& src,
                    const phi::DeviceContext& ctx,
                    std::vector<T>* dst) {
  auto src_ptr = static_cast<const void*>(src.data<T>());
  auto size = src.numel() * sizeof(T);

  phi::CPUPlace dst_place;
  dst->resize(src.numel());
  auto dst_ptr = static_cast<void*>(dst->data());

  if (phi::is_cpu_place(src.place())) {
    memory::Copy(dst_place, dst_ptr, src.place(), src_ptr, size);
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  else if (phi::is_gpu_place(src.place())) {  // NOLINT
    memory::Copy(dst_place,
                 dst_ptr,
                 src.place(),
                 src_ptr,
                 size,
                 reinterpret_cast<const phi::GPUContext&>(ctx).stream());
  }
#endif
#if defined(PADDLE_WITH_XPU)
  else if (phi::is_xpu_place(src.place())) {  // NOLINT
    memory::Copy(dst_place, dst_ptr, src.place(), src_ptr, size);
  }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  else if (phi::is_custom_place(src.place())) {  // NOLINT
    memory::Copy(dst_place, dst_ptr, src.place(), src_ptr, size, nullptr);
  }
#endif
  else {  // NOLINT
    PADDLE_THROW(common::errors::Unimplemented(
        "TensorToVector on %s is not supported.", src.place()));
  }
}

template <>
inline void TensorToVector(const phi::DenseTensor& src,
                           const phi::DeviceContext& ctx,
                           std::vector<bool>* dst) {
  auto src_ptr = static_cast<const void*>(src.data<bool>());
  auto size = src.numel() * sizeof(bool);

  bool* array = new bool[src.numel()];

  phi::CPUPlace dst_place;
  dst->resize(src.numel());
  auto dst_ptr = static_cast<void*>(array);

  if (phi::is_cpu_place(src.place())) {
    memory::Copy(dst_place, dst_ptr, src.place(), src_ptr, size);
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  else if (phi::is_gpu_place(src.place())) {  // NOLINT
    memory::Copy(dst_place,
                 dst_ptr,
                 src.place(),
                 src_ptr,
                 size,
                 reinterpret_cast<const phi::GPUContext&>(ctx).stream());
  }
#endif
#if defined(PADDLE_WITH_XPU)
  else if (phi::is_xpu_place(src.place())) {  // NOLINT
    memory::Copy(dst_place, dst_ptr, src.place(), src_ptr, size);
  }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  else if (phi::is_custom_place(src.place())) {  // NOLINT
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

  phi::CPUPlace dst_place;
  dst->resize(src.numel());
  auto dst_ptr = static_cast<void*>(dst->data());

  PADDLE_ENFORCE_EQ(
      phi::is_cpu_place(src.place()),
      true,
      common::errors::InvalidArgument(
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

  phi::CPUPlace dst_place;
  dst->resize(src.numel());
  auto dst_ptr = static_cast<void*>(array);

  PADDLE_ENFORCE_EQ(
      phi::is_cpu_place(src.place()),
      true,
      common::errors::InvalidArgument(
          "The input tensor should be CPU device, but actually it is in %s.",
          src.place()));

  memory::Copy(dst_place, dst_ptr, src.place(), src_ptr, size);

  for (unsigned int i = 0; i < src.numel(); i++) {
    (*dst)[i] = static_cast<bool>(array[i]);
  }
  delete[] array;
}

std::ostream& operator<<(std::ostream& os, const LegacyLoD& lod);

template <typename T>
inline T GetValue(const phi::DenseTensor* x) {
  T value = static_cast<T>(0);
  if (!phi::is_cpu_place(x->place())) {
    phi::DenseTensor cpu_x;
    framework::TensorCopy(*x, phi::CPUPlace(), &cpu_x);
    value = cpu_x.data<T>()[0];
  } else {
    value = x->data<T>()[0];
  }
  return value;
}

}  // namespace framework
}  // namespace paddle

namespace phi {
TEST_API std::ostream& operator<<(std::ostream& os, const DenseTensor& t);
}
