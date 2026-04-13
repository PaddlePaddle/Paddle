// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <ATen/TensorIndexing.h>
#include <ATen/core/TensorBase.h>
#include <c10/core/Backend.h>
#include <c10/core/List.h>
#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>
#include <c10/core/Stream.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/util/OptionalArrayRef.h>
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/memory/malloc.h"

#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#elif defined(PADDLE_WITH_CUDA)
#include <cuda_runtime_api.h>
#endif

// Forward declaration to allow record_stream(at::cuda::CUDAStream) overload
// without pulling in the full CUDAStream header here.
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
namespace c10::cuda {
class CUDAStream;
}  // namespace c10::cuda
namespace at::cuda {
using c10::cuda::CUDAStream;
}  // namespace at::cuda
#endif

#include <limits>
#include <optional>
#include <utility>
#include <vector>
#include "paddle/common/ddim.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/enforce.h"

namespace at {
class Tensor;

// Type aliases for ATen compatibility
using Scalar = c10::Scalar;
using TensorOptions = c10::TensorOptions;
using MemoryFormat = c10::MemoryFormat;
using IntArrayRef = c10::IntArrayRef;
using OptionalIntArrayRef = c10::OptionalIntArrayRef;
using ScalarType = c10::ScalarType;
}  // namespace at

namespace at {  // NOLINT(build/namespaces)
using PaddleTensor = paddle::Tensor;
using PaddlePlace = phi::Place;

// Stub for DimnameList (not supported in Paddle)
using DimnameList = c10::ArrayRef<std::string>;

using Stream = c10::Stream;

class Tensor : public TensorBase {
 public:
  Tensor() = default;
  Tensor(const PaddleTensor& tensor) : TensorBase(tensor){};  // NOLINT
  Tensor(const Tensor& tensor) = default;
  Tensor(Tensor&& tensor) = default;

  // Implicitly move-constructible from TensorBase, but must be explicit to
  // increase refcount
  explicit Tensor(const TensorBase& base) : TensorBase(base) {}  // NOLINT
  /*implicit*/ Tensor(TensorBase&& base)                         // NOLINT
      : TensorBase(std::move(base)) {}

  Tensor& operator=(const PaddleTensor& x) & noexcept {
    tensor_ = x;
    return *this;
  }
  Tensor& operator=(const TensorBase& x) & noexcept {
    const PaddleTensor& inner = x._PD_GetInner();
    tensor_ = inner;
    return *this;
  }
  Tensor& operator=(PaddleTensor&& x) & noexcept {
    tensor_ = std::move(x);
    return *this;
  }
  Tensor& operator=(TensorBase&& x) & noexcept {
    tensor_ = std::move(x)._PD_GetInner();
    return *this;
  }

  Tensor& operator=(const Tensor& x) & noexcept {
    return operator=(static_cast<const TensorBase&>(x));
  }
  Tensor& operator=(Tensor&& x) & noexcept {
    return operator=(static_cast<TensorBase&&>(x));
  }
  Tensor& operator=(const Scalar& v) && {
    fill_(v);
    return *this;
  }
  Tensor& operator=(const Tensor& rhs) && {
    copy_(rhs);
    return *this;
  }
  Tensor& operator=(Tensor&& rhs) && {
    copy_(rhs);
    return *this;
  }

  void* data_ptr() const { return const_cast<void*>(tensor_.data()); }
  template <typename T>
  T* data_ptr() const {
    return const_cast<T*>(tensor_.data<T>());
  }

  template <typename T>
  void* data() const {
    return data_ptr<T>();
  }

  Tensor toBackend(c10::Backend b) const {
    if (b == c10::Backend::CPU) {
      PaddlePlace place(phi::AllocationType::CPU);
      return tensor_.copy_to(place, true);
    } else if (b == c10::Backend::CUDA) {
      auto place = paddle::DefaultGPUPlace();
      return tensor_.copy_to(place, true);
    } else if (b == c10::Backend::XPU) {
      PaddlePlace place(phi::AllocationType::XPU);
      return tensor_.copy_to(place, true);
    } else if (b == c10::Backend::IPU) {
      PaddlePlace place(phi::AllocationType::IPU);
      return tensor_.copy_to(place, true);
    } else {
      PD_CHECK(false, "Unsupported backend");
    }
    return tensor_;
  }

  Tensor cpu() const {
    PaddlePlace place(phi::AllocationType::CPU);
    return tensor_.copy_to(place, true);
  }

  Tensor cuda() const {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    auto place = paddle::DefaultGPUPlace();
    return tensor_.copy_to(place, true);
#elif defined(PADDLE_WITH_XPU)
    return tensor_.copy_to(paddle::DefaultXPUPlace(), true);
#elif defined(PADDLE_WITH_CUSTOM_DEVICE)
    return tensor_.copy_to(paddle::DefaultCustomPlace(), true);
#else
    PD_THROW(
        "cuda() is not supported: no GPU/XPU/Custom device enabled "
        "in this build.");
#endif
  }

  const void* const_data_ptr() const {
    return const_cast<void*>(tensor_.data());
  }

  template <typename T, std::enable_if_t<!std::is_const_v<T>, int> = 0>
  const T* const_data_ptr() const {
    return TensorBase::const_data_ptr<T>();
  }

  template <typename T, std::enable_if_t<std::is_const_v<T>, int> = 0>
  const std::remove_const_t<T>* const_data_ptr() const {
    return TensorBase::const_data_ptr<T>();
  }

  void* mutable_data_ptr() const { return const_cast<void*>(tensor_.data()); }

  template <typename T>
  T* mutable_data_ptr() const {
    return TensorBase::mutable_data_ptr<T>();
  }

  using TensorBase::stride;

  c10::IntArrayRef strides() const {
    return compat::_PD_PhiDDimToIntArrayRef(tensor_.strides());
  }

  using TensorBase::size;

  c10::IntArrayRef sizes() const {
    return compat::_PD_PhiDDimToIntArrayRef(tensor_.dims());
  }

  at::Tensor to(
      at::TensorOptions options = {},
      bool non_blocking = false,
      bool copy = false,
      ::std::optional<at::MemoryFormat> memory_format = ::std::nullopt) const;
  at::Tensor to(::std::optional<at::ScalarType> dtype,
                ::std::optional<at::Layout> layout,
                ::std::optional<at::Device> device,
                ::std::optional<bool> pin_memory,
                bool non_blocking,
                bool copy,
                ::std::optional<at::MemoryFormat> memory_format) const;
  at::Tensor to(
      at::Device device,
      at::ScalarType dtype,
      bool non_blocking = false,
      bool copy = false,
      ::std::optional<at::MemoryFormat> memory_format = ::std::nullopt) const;
  at::Tensor to(
      at::ScalarType dtype,
      bool non_blocking = false,
      bool copy = false,
      ::std::optional<at::MemoryFormat> memory_format = ::std::nullopt) const;
  at::Tensor to(
      const at::Tensor& other,
      bool non_blocking = false,
      bool copy = false,
      ::std::optional<at::MemoryFormat> memory_format = ::std::nullopt) const;

  Tensor meta() const {
    PD_THROW("`meta()` is not supported in this Paddle build.");
  }

  at::Scalar item() const;

  template <typename T>
  T item() const;

  bool equal(const at::Tensor& other) const;

  // Clamp functions
  at::Tensor clamp(
      const ::std::optional<at::Scalar>& min,
      const ::std::optional<at::Scalar>& max = ::std::nullopt) const;

  at::Tensor clamp(const ::std::optional<at::Tensor>& min = {},
                   const ::std::optional<at::Tensor>& max = {}) const;

  at::Tensor& clamp_(
      const ::std::optional<at::Scalar>& min,
      const ::std::optional<at::Scalar>& max = ::std::nullopt) const;

  at::Tensor& clamp_(const ::std::optional<at::Tensor>& min = {},
                     const ::std::optional<at::Tensor>& max = {}) const;

  at::Tensor clamp_max(const at::Scalar& max) const;
  at::Tensor clamp_max(const at::Tensor& max) const;
  at::Tensor& clamp_max_(const at::Scalar& max) const;
  at::Tensor& clamp_max_(const at::Tensor& max) const;

  at::Tensor clamp_min(const at::Scalar& min) const;
  at::Tensor clamp_min(const at::Tensor& min) const;
  at::Tensor& clamp_min_(const at::Scalar& min) const;
  at::Tensor& clamp_min_(const at::Tensor& min) const;

  // as_strided: Create a tensor view with custom size, stride, and
  // storage_offset
  at::Tensor as_strided(
      at::IntArrayRef size,
      at::IntArrayRef stride,
      ::std::optional<int64_t> storage_offset = ::std::nullopt) const;

  // as_strided_: Inplace version
  const at::Tensor& as_strided_(
      at::IntArrayRef size,
      at::IntArrayRef stride,
      ::std::optional<int64_t> storage_offset = ::std::nullopt) const;

  // as_strided_scatter: Scatter src into a strided view
  at::Tensor as_strided_scatter(
      const at::Tensor& src,
      at::IntArrayRef size,
      at::IntArrayRef stride,
      ::std::optional<int64_t> storage_offset = ::std::nullopt) const;

  // Standard deviation functions
  Tensor std(int dim) const;
  Tensor std(bool unbiased = true) const;
  Tensor std(at::OptionalIntArrayRef dim,
             bool unbiased = true,
             bool keepdim = false) const;
  Tensor std(at::OptionalIntArrayRef dim,
             const ::std::optional<at::Scalar>& correction,
             bool keepdim = false) const;

  Tensor tensor_data() const {
    PaddleTensor result;
    if (tensor_.initialized()) {
      auto src_impl = tensor_.impl();
      auto* src_tensor =
          std::dynamic_pointer_cast<phi::DenseTensor>(src_impl).get();
      if (src_tensor && src_tensor->meta().is_contiguous()) {
        result.set_impl(std::make_shared<phi::DenseTensor>());
        auto* dst_tensor =
            std::dynamic_pointer_cast<phi::DenseTensor>(result.impl()).get();
        dst_tensor->ShareDataWith(*src_tensor);
      } else {
        result = paddle::experimental::assign(tensor_);
      }
    }
    // For uninitialized tensor, return an uninitialized tensor (no assign
    // needed)
    return Tensor(result);
  }

  Tensor variable_data() const {
    PaddleTensor result;
    if (tensor_.initialized()) {
      auto src_impl = tensor_.impl();
      auto* src_tensor =
          std::dynamic_pointer_cast<phi::DenseTensor>(src_impl).get();
      if (src_tensor && src_tensor->meta().is_contiguous()) {
        result.set_impl(std::make_shared<phi::DenseTensor>());
        auto* dst_tensor =
            std::dynamic_pointer_cast<phi::DenseTensor>(result.impl()).get();
        dst_tensor->ShareDataWith(*src_tensor);
      } else {
        result = paddle::experimental::assign(tensor_);
      }
    }
    // For uninitialized tensor, return an uninitialized tensor (no assign
    // needed)
    return Tensor(result);
  }

  // index: Get values at specified tensor indices
  at::Tensor index(const c10::List<::std::optional<at::Tensor>>& indices) const;

  // index_put_: Set values at specified indices in-place
  at::Tensor& index_put_(const c10::List<::std::optional<at::Tensor>>& indices,
                         const at::Tensor& values,
                         bool accumulate = false) const;

  // index_put_: Set scalar value at specified indices in-place
  at::Tensor& index_put_(const c10::List<::std::optional<at::Tensor>>& indices,
                         const at::Scalar& v,
                         bool accumulate = false) const;

  // index_put: Non-inplace version of index_put_
  at::Tensor index_put(const c10::List<::std::optional<at::Tensor>>& indices,
                       const at::Tensor& values,
                       bool accumulate = false) const;

  Tensor toType(ScalarType t) const {
    return Tensor(paddle::experimental::cast(
        tensor_, compat::_PD_AtenScalarTypeToPhiDataType(t)));
  }

  int64_t numel() const { return tensor_.numel(); }

  caffe2::TypeMeta dtype() const {
    return caffe2::TypeMeta::fromScalarType(
        compat::_PD_PhiDataTypeToAtenScalarType(tensor_.dtype()));
  }

  c10::Device device() const { return c10::Device(tensor_.place()); }
  c10::DeviceIndex get_device() const {
    return c10::Device(tensor_.place()).index();
  }

  int64_t dim() const { return tensor_.dims().size(); }
  int64_t ndimension() const { return dim(); }

  at::Tensor contiguous(
      c10::MemoryFormat memory_format = c10::MemoryFormat::Contiguous) const {
    PD_CHECK(memory_format == c10::MemoryFormat::Contiguous,
             "`MemoryFormat` other than Contiguous");

    return tensor_.contiguous();
  }

  bool is_contiguous(
      at::MemoryFormat memory_format = at::MemoryFormat::Contiguous) const {
    PD_CHECK(memory_format == c10::MemoryFormat::Contiguous,
             "`MemoryFormat` other than Contiguous");

    return tensor_.is_contiguous();
  }

  c10::ScalarType scalar_type() const {
    return compat::_PD_PhiDataTypeToAtenScalarType(tensor_.dtype());
  }

  at::Tensor flatten(int64_t start_dim, int64_t end_dim) const;
  at::Tensor unflatten(int64_t dim, at::IntArrayRef sizes) const;
  at::Tensor unflatten_symint(int64_t dim, c10::SymIntArrayRef sizes) const;

  Tensor& fill_(const at::Scalar& value) const {
    paddle::experimental::fill_(const_cast<PaddleTensor&>(tensor_), value);
    return const_cast<at::Tensor&>(*this);
  }

  Tensor& zero_() const {
    paddle::experimental::fill_(const_cast<PaddleTensor&>(tensor_), 0.0);
    return const_cast<at::Tensor&>(*this);
  }

  bool is_cpu() const { return phi::is_cpu_place(tensor_.place()); }
  bool is_cuda() const { return phi::is_gpu_place(tensor_.place()); }

  bool is_pinned(::std::optional<c10::Device> device = ::std::nullopt) const {
    if (device.has_value()) {
      phi::enforce::ThrowWarnInternal(
          "The argument 'device' of Tensor.is_pinned() is deprecated. "
          "Please do not pass this argument.");
    }

    const PaddlePlace place = tensor_.place();
    const bool is_gpu_pinned = phi::is_cuda_pinned_place(place);
    const bool is_xpu_pinned = phi::is_xpu_pinned_place(place);

    // Keep parity with PyTorch behavior: only host tensors are pinnable.
    if (!(phi::is_cpu_place(place) || is_gpu_pinned || is_xpu_pinned)) {
      return false;
    }

    if (!device.has_value()) {
      return is_gpu_pinned || is_xpu_pinned;
    }

    const auto device_type = device.value().type();
    if (device_type == c10::DeviceType::CUDA) {
      return is_gpu_pinned;
    }
    if (device_type == c10::DeviceType::XPU) {
      return is_xpu_pinned;
    }
    // CPU and non-accelerator devices are not valid pinned backends.
    return false;
  }

  Tensor pin_memory(
      ::std::optional<c10::Device> device = ::std::nullopt) const {
    if (device.has_value()) {
      phi::enforce::ThrowWarnInternal(
          "The argument 'device' of Tensor.pin_memory() is deprecated. "
          "Please do not pass this argument.");
    }

    if (is_pinned(device)) {
      return *this;
    }

    const PaddlePlace current_place = tensor_.place();
    if (!phi::is_cpu_place(current_place)) {
      PD_THROW("cannot pin '" + this->toString() +
               "', only dense CPU tensors can be pinned");
    }

    PaddlePlace pinned_place;

    if (device.has_value()) {
      const auto device_type = device.value().type();
      if (device_type == c10::DeviceType::CUDA) {
        pinned_place = phi::Place(phi::GPUPinnedPlace());
      } else if (device_type == c10::DeviceType::XPU) {
        pinned_place = phi::Place(phi::XPUPinnedPlace());
      } else {
        PD_THROW("pin_memory device type must be an accelerator (GPU/XPU)");
      }
    } else {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      pinned_place = phi::Place(phi::GPUPinnedPlace());
#elif defined(PADDLE_WITH_XPU)
      pinned_place = phi::Place(phi::XPUPinnedPlace());
#else
      PD_THROW("pin_memory is not supported: no GPU/XPU backend enabled");
#endif
    }

    return tensor_.copy_to(pinned_place, true);
  }

  at::Tensor narrow_copy(int64_t dim, int64_t start, int64_t length) const;
  at::Tensor narrow_copy_symint(int64_t dim,
                                c10::SymInt start,
                                c10::SymInt length) const;

  at::Tensor narrow(int64_t dim, int64_t start, int64_t length) const;
  at::Tensor narrow_symint(int64_t dim,
                           c10::SymInt start,
                           c10::SymInt length) const;
  at::Tensor narrow(int64_t dim, const at::Tensor& start, int64_t length) const;
  at::Tensor narrow_symint(int64_t dim,
                           const at::Tensor& start,
                           c10::SymInt length) const;

  at::Tensor reshape(at::IntArrayRef shape) const;

  at::Tensor transpose(int64_t dim0, int64_t dim1) const;
  at::Tensor& transpose_(int64_t dim0, int64_t dim1) const;

  at::Tensor permute(at::IntArrayRef dims) const;

  at::Tensor reciprocal() const;
  at::Tensor& reciprocal_() const;

  at::Tensor detach() const;
  at::Tensor& detach_() const;

  at::Tensor select(int64_t dim, int64_t index) const;
  at::Tensor select_symint(int64_t dim, c10::SymInt index) const;

  at::Tensor& copy_(const at::Tensor& src, bool non_blocking = false) const {
    const_cast<PaddleTensor&>(tensor_).copy_(
        src._PD_GetInner(), tensor_.place(), /*blocking=*/!non_blocking);
    return const_cast<at::Tensor&>(*this);
  }

  at::Tensor view(at::IntArrayRef size) const;
  at::Tensor view(at::ScalarType dtype) const;

  at::Tensor squeeze() const;
  at::Tensor squeeze(int64_t dim) const;
  at::Tensor squeeze(at::IntArrayRef dim) const;
  at::Tensor& squeeze_() const;
  at::Tensor& squeeze_(int64_t dim) const;
  at::Tensor& squeeze_(at::IntArrayRef dim) const;

  at::Tensor unsqueeze() const;
  at::Tensor unsqueeze(int64_t dim) const;
  at::Tensor unsqueeze(at::IntArrayRef dim) const;
  at::Tensor& unsqueeze_() const;
  at::Tensor& unsqueeze_(int64_t dim) const;
  at::Tensor& unsqueeze_(at::IntArrayRef dim) const;

  at::Tensor sum(::std::optional<at::ScalarType> dtype = ::std::nullopt) const;
  at::Tensor sum(at::OptionalIntArrayRef dim,
                 bool keepdim = false,
                 ::std::optional<at::ScalarType> dtype = ::std::nullopt) const;

  at::Tensor t() const;
  at::Tensor& t_() const;

  at::Tensor view_as(const at::Tensor& other) const;

  at::Tensor coalesce() const;
  bool is_coalesced() const;

  int64_t _nnz() const;
  at::Tensor _values() const;

  bool is_variable() const noexcept { return true; }

  at::Tensor index_select(int64_t dim, const at::Tensor& index) const {
    return Tensor(
        paddle::experimental::index_select(tensor_, index._PD_GetInner(), dim));
  }

  at::Tensor masked_select(const at::Tensor& mask) const;

  std::vector<at::Tensor> tensor_split(int64_t sections, int64_t dim) const;
  std::vector<at::Tensor> tensor_split_symint(c10::SymInt sections,
                                              int64_t dim) const;
  std::vector<at::Tensor> tensor_split(at::IntArrayRef indices,
                                       int64_t dim) const;
  std::vector<at::Tensor> tensor_split_symint(c10::SymIntArrayRef indices,
                                              int64_t dim) const;
  std::vector<at::Tensor> tensor_split(
      const at::Tensor& tensor_indices_or_sections, int64_t dim) const;

  std::vector<at::Tensor> split(int64_t split_size, int64_t dim) const;
  std::vector<at::Tensor> split_symint(c10::SymInt split_size,
                                       int64_t dim) const;
  std::vector<at::Tensor> split(at::IntArrayRef split_sizes, int64_t dim) const;
  std::vector<at::Tensor> split_symint(c10::SymIntArrayRef split_sizes,
                                       int64_t dim) const;

  std::vector<at::Tensor> unsafe_split(int64_t split_size, int64_t dim) const;
  std::vector<at::Tensor> unsafe_split_symint(c10::SymInt split_size,
                                              int64_t dim) const;

  std::vector<at::Tensor> split_with_sizes(at::IntArrayRef split_sizes,
                                           int64_t dim) const;
  std::vector<at::Tensor> split_with_sizes_symint(
      c10::SymIntArrayRef split_sizes, int64_t dim) const;

  std::vector<at::Tensor> unsafe_split_with_sizes(at::IntArrayRef split_sizes,
                                                  int64_t dim) const;
  std::vector<at::Tensor> unsafe_split_with_sizes_symint(
      c10::SymIntArrayRef split_sizes, int64_t dim) const;

  std::vector<at::Tensor> hsplit(int64_t sections) const;
  std::vector<at::Tensor> hsplit(at::IntArrayRef indices) const;

  std::vector<at::Tensor> vsplit(int64_t sections) const;
  std::vector<at::Tensor> vsplit(at::IntArrayRef indices) const;

  std::vector<at::Tensor> dsplit(int64_t sections) const;
  std::vector<at::Tensor> dsplit(at::IntArrayRef indices) const;

  at::Tensor bitwise_right_shift(const Scalar& other) const {
    return Tensor(paddle::experimental::bitwise_right_shift(
        tensor_, paddle::experimental::full({}, other, other.dtype())));
  }

  at::Tensor slice(int64_t dim = 0,
                   ::std::optional<int64_t> start = ::std::nullopt,
                   ::std::optional<int64_t> end = ::std::nullopt,
                   int64_t step = 1);

  at::Tensor index(ArrayRef<at::indexing::TensorIndex> indices) const;
  inline at::Tensor index(
      std::initializer_list<at::indexing::TensorIndex> indices) const {
    return index(ArrayRef<at::indexing::TensorIndex>(indices));
  }

  at::Tensor& floor_divide_(const at::Scalar& other) const {
    paddle::experimental::floor_divide_(
        const_cast<PaddleTensor&>(tensor_),
        paddle::experimental::full({}, other, other.dtype()));
    return const_cast<at::Tensor&>(*this);
  }

  // Paddle Tensor has no storage_offset, so we add it here, and it is always
  // 0.
  //   int64_t storage_offset() const { return storage_offset_; }

  inline size_t nbytes() const {
    PD_CHECK(
        ((tensor_.layout() != common::DataLayout::SPARSE_COO) &&
         (tensor_.layout() != common::DataLayout::SPARSE_CSR)),
        "nbytes is not defined for sparse tensors.  If you want the size of "
        "the constituent "
        "tensors, add the nbytes of the indices and values.  If you want the "
        "size of the  "
        "equivalent dense tensor, multiply numel() by element_size()");
    return tensor_.numel() * SizeOf(tensor_.dtype());
  }

  size_t itemsize() const { return SizeOf(tensor_.dtype()); }

  int64_t element_size() const {
    return static_cast<int64_t>(SizeOf(tensor_.dtype()));
  }

  inline Tensor clone() const {
    PaddleTensor cloned_tensor = paddle::experimental::assign(tensor_);
    return Tensor(cloned_tensor);
  }

  // all: Check if all elements are true (non-zero)
  at::Tensor all() const;
  at::Tensor all(int64_t dim, bool keepdim = false) const;
  at::Tensor all(at::OptionalIntArrayRef dim, bool keepdim = false) const;

  // allclose: Check if two tensors are close to each other
  bool allclose(const at::Tensor& other,
                double rtol = 1e-05,
                double atol = 1e-08,
                bool equal_nan = false) const;

  at::Tensor abs() const;

  at::Tensor& abs_() const;

  at::Tensor absolute() const { return abs(); }

  at::Tensor& absolute_() const { return abs_(); }

  Tensor operator[](int64_t index) const {
    // Use as_strided to create a view (shares storage with original tensor)
    // This allows fill_ to modify the original tensor
    int64_t numel = tensor_.numel();
    if (numel == 0) {
      PD_THROW("operator[]: cannot index empty tensor");
    }

    // Handle negative index
    if (index < 0) {
      index += tensor_.dims()[0];
    }

    // Check bounds
    if (index < 0 || index >= tensor_.dims()[0]) {
      PD_THROW("operator[]: index ",
               index,
               " out of range for tensor of size ",
               tensor_.dims(),
               " at dimension 0");
    }

    // For 1D tensor: create a scalar view (0-dim tensor) with proper offset
    // For multi-D tensor: create a view of the row at index
    std::vector<int64_t> new_sizes;
    std::vector<int64_t> new_strides;

    auto dims = tensor_.dims();
    auto stride = tensor_.strides();

    // Skip the first dimension (dim 0)
    for (int i = 1; i < dims.size(); ++i) {
      new_sizes.push_back(dims[i]);
      new_strides.push_back(stride[i]);
    }

    // Calculate storage offset
    int64_t storage_offset = index * stride[0];

    return as_strided(c10::IntArrayRef(new_sizes),
                      c10::IntArrayRef(new_strides),
                      storage_offset);
  }

  void record_stream(at::Stream s) const;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  void record_stream(at::cuda::CUDAStream s) const;
// TODO(youge325): Remove after DeepEP paddle branch is updated to use
// at::Stream
#ifdef PADDLE_WITH_HIP
  void record_stream(hipStream_t s) const;
#else
  void record_stream(cudaStream_t s) const;
#endif
#endif

  Tensor var(int dim) const { return var(at::IntArrayRef{dim}, true, false); }

  Tensor var(bool unbiased = true) const {
    std::vector<int64_t> empty_dims;
    double correction = unbiased ? 1.0 : 0.0;
    return var_impl(empty_dims, correction, false);
  }

  Tensor var(at::OptionalIntArrayRef dim,
             bool unbiased = true,
             bool keepdim = false) const {
    // Convert unbiased to correction: unbiased=True means correction=1
    double correction = unbiased ? 1.0 : 0.0;
    std::vector<int64_t> dims_vec;
    if (dim.has_value() && dim.value().size() > 0) {
      dims_vec.assign(dim.value().begin(), dim.value().end());
    }
    return var_impl(dims_vec, correction, keepdim);
  }

  Tensor var(at::OptionalIntArrayRef dim,
             const ::std::optional<at::Scalar>& correction,
             bool keepdim = false) const {
    double correction_value = 1.0;
    if (correction.has_value()) {
      const at::Scalar& scalar = correction.value();
      correction_value = scalar.to<double>();
    }
    std::vector<int64_t> dims_vec;
    if (dim.has_value() && dim.value().size() > 0) {
      dims_vec.assign(dim.value().begin(), dim.value().end());
    }
    return var_impl(dims_vec, correction_value, keepdim);
  }

 private:
  Tensor var_impl(const std::vector<int64_t>& dims_vec,
                  double correction_value,
                  bool keepdim) const {
    phi::IntArray dims_int_array(dims_vec);

    PaddleTensor mean_tensor;
    if (dims_vec.empty()) {
      mean_tensor = paddle::experimental::mean(
          tensor_, phi::IntArray(std::vector<int64_t>{}), true);
    } else {
      mean_tensor = paddle::experimental::mean(tensor_, dims_int_array, true);
    }

    PaddleTensor diff = paddle::experimental::subtract(tensor_, mean_tensor);
    PaddleTensor diff_squared = paddle::experimental::multiply(diff, diff);

    PaddleTensor sum_squared_diff;
    if (dims_vec.empty()) {
      sum_squared_diff =
          paddle::experimental::sum(diff_squared,
                                    phi::IntArray(std::vector<int64_t>{}),
                                    diff_squared.dtype(),
                                    keepdim);
    } else {
      sum_squared_diff = paddle::experimental::sum(
          diff_squared, dims_int_array, diff_squared.dtype(), keepdim);
    }

    int64_t n = tensor_.numel();
    if (!dims_vec.empty()) {
      n = 1;
      for (int64_t d : dims_vec) {
        int64_t dim_idx = d < 0 ? d + tensor_.dims().size() : d;
        if (dim_idx >= 0 &&
            dim_idx < static_cast<int64_t>(tensor_.dims().size())) {
          n *= tensor_.dims()[dim_idx];
        }
      }
    }

    double corrected_n = static_cast<double>(n) - correction_value;
    if (corrected_n <= 0.0) {
      corrected_n = static_cast<double>(n);
    }

    std::vector<int64_t> result_shape_vec;
    for (int64_t i = 0; i < sum_squared_diff.dims().size(); ++i) {
      result_shape_vec.push_back(sum_squared_diff.dims()[i]);
    }
    PaddleTensor correction_scalar =
        paddle::experimental::full(phi::IntArray(result_shape_vec),
                                   phi::Scalar(corrected_n),
                                   sum_squared_diff.dtype(),
                                   sum_squared_diff.place());
    PaddleTensor result =
        paddle::experimental::divide(sum_squared_diff, correction_scalar);

    return Tensor(result);
  }

 public:
  // Deprecated packed_accessor for compatibility with PyTorch
  // Use packed_accessor32 or packed_accessor64 instead
  template <typename T,
            size_t N,
            template <typename U> class PtrTraits = DefaultPtrTraits,
            typename index_t = int64_t>
  [[deprecated(
      "packed_accessor is deprecated, use packed_accessor32 or "
      "packed_accessor64 instead")]] GenericPackedTensorAccessor<T,
                                                                 N,
                                                                 PtrTraits,
                                                                 index_t>
  packed_accessor() const& {
    return this->template generic_packed_accessor<T, N, PtrTraits, index_t>();
  }

  template <typename T,
            size_t N,
            template <typename U> class PtrTraits = DefaultPtrTraits,
            typename index_t = int64_t>
  [[deprecated(
      "packed_accessor is deprecated, use packed_accessor32 or "
      "packed_accessor64 instead")]] GenericPackedTensorAccessor<T,
                                                                 N,
                                                                 PtrTraits,
                                                                 index_t>
  packed_accessor() && = delete;

  // register_hook - throws exception for Paddle compatibility
  // Paddle does not support gradient hooks
  template <typename T>
  unsigned register_hook(T&&) const {
    throw std::runtime_error(
        "register_hook is not supported in Paddle, this is an ATen "
        "compatibility API that is not available");
  }

  // any - returns true if any element is non-zero
  Tensor any(int64_t dim, bool keepdim = false) const;
  Tensor any(at::OptionalIntArrayRef dim, bool keepdim = false) const;
  Tensor any() const;

  // chunk - splits tensor into chunks
  std::vector<Tensor> chunk(int64_t chunks, int64_t dim = 0) const;

  // rename - stub for Paddle (Dimname not supported)
  Tensor rename(::std::optional<at::DimnameList> names) const;

  // new_empty - creates uninitialized tensor with same dtype/device
  Tensor new_empty(at::IntArrayRef size, at::TensorOptions options = {}) const;
  Tensor new_empty(at::IntArrayRef size,
                   ::std::optional<at::ScalarType> dtype,
                   ::std::optional<at::Layout> layout,
                   ::std::optional<at::Device> device,
                   ::std::optional<bool> pin_memory) const;

  // new_full - creates tensor filled with fill_value
  Tensor new_full(at::IntArrayRef size,
                  const at::Scalar& fill_value,
                  at::TensorOptions options = {}) const;
  Tensor new_full(at::IntArrayRef size,
                  const at::Scalar& fill_value,
                  ::std::optional<at::ScalarType> dtype,
                  ::std::optional<at::Layout> layout,
                  ::std::optional<at::Device> device,
                  ::std::optional<bool> pin_memory) const;

  // new_zeros - creates zero tensor
  Tensor new_zeros(at::IntArrayRef size, at::TensorOptions options = {}) const;
  Tensor new_zeros(at::IntArrayRef size,
                   ::std::optional<at::ScalarType> dtype,
                   ::std::optional<at::Layout> layout,
                   ::std::optional<at::Device> device,
                   ::std::optional<bool> pin_memory) const;

  // new_ones - creates tensor filled with ones
  Tensor new_ones(at::IntArrayRef size, at::TensorOptions options = {}) const;
  Tensor new_ones(at::IntArrayRef size,
                  ::std::optional<at::ScalarType> dtype,
                  ::std::optional<at::Layout> layout,
                  ::std::optional<at::Device> device,
                  ::std::optional<bool> pin_memory) const;

  // resize_ - in-place resize
  const Tensor& resize_(
      at::IntArrayRef size,
      ::std::optional<at::MemoryFormat> memory_format = ::std::nullopt) const;

  // expand - expands tensor to new size
  Tensor expand(at::IntArrayRef size, bool implicit = false) const;

  // expand_as - expands to same size as another tensor
  Tensor expand_as(const Tensor& other) const;

  PaddleTensor _PD_GetInner() const { return tensor_; }
  PaddleTensor& _PD_GetInner() { return tensor_; }
};  // NOLINT(readability/braces)
}  // namespace at
