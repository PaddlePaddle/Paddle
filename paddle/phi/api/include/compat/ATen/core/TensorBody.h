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

#include <ATen/core/TensorBase.h>
#include <ATen/indexing.h>
#include <c10/core/Backend.h>
#include <c10/core/Device.h>
#include <utility>
#include <vector>
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/memory/malloc.h"

namespace at {  // NOLINT(build/namespaces)
using PaddleTensor = paddle::Tensor;
using PaddlePlace = phi::Place;
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
      PaddlePlace place(phi::AllocationType::GPU);
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
    PaddlePlace place(phi::AllocationType::GPU);
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
  const T* const_data_ptr() const;

  template <typename T, std::enable_if_t<std::is_const_v<T>, int> = 0>
  const std::remove_const_t<T>* const_data_ptr() const;

  void* mutable_data_ptr() const { return const_cast<void*>(tensor_.data()); }

  template <typename T>
  T* mutable_data_ptr() const;

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
      ::std::optional<at::MemoryFormat> memory_format = ::std::nullopt) const {
    return TensorBase::to(options, non_blocking, copy, memory_format);
  }

  Tensor meta() const {
    PD_THROW("`meta()` is not supported in this Paddle build.");
  }

  at::Scalar item() const {
    if (tensor_.numel() != 1) {
      PD_THROW("only one element tensors can be converted to Python scalars");
    }

    // Move to CPU if necessary (for compatibility with PyTorch behavior)
    PaddleTensor cpu_tensor = tensor_;
    if (!phi::is_cpu_place(tensor_.place())) {
      PaddlePlace place(phi::AllocationType::CPU);
      cpu_tensor = tensor_.copy_to(place, true);
    }

    auto dtype = cpu_tensor.dtype();
    if (dtype == phi::DataType::FLOAT32) {
      return at::Scalar(*(cpu_tensor.data<float>()));
    } else if (dtype == phi::DataType::FLOAT64) {
      return at::Scalar(*(cpu_tensor.data<double>()));
    } else if (dtype == phi::DataType::FLOAT16) {
      return at::Scalar(
          static_cast<float>(*(cpu_tensor.data<phi::dtype::float16>())));
    } else if (dtype == phi::DataType::BFLOAT16) {
      return at::Scalar(
          static_cast<float>(*(cpu_tensor.data<phi::dtype::bfloat16>())));
    } else if (dtype == phi::DataType::INT8) {
      return at::Scalar(*(cpu_tensor.data<int8_t>()));
    } else if (dtype == phi::DataType::INT16) {
      return at::Scalar(*(cpu_tensor.data<int16_t>()));
    } else if (dtype == phi::DataType::INT32) {
      return at::Scalar(*(cpu_tensor.data<int32_t>()));
    } else if (dtype == phi::DataType::INT64) {
      return at::Scalar(*(cpu_tensor.data<int64_t>()));
    } else if (dtype == phi::DataType::UINT8) {
      return at::Scalar(*(cpu_tensor.data<uint8_t>()));
    } else if (dtype == phi::DataType::BOOL) {
      return at::Scalar(*(cpu_tensor.data<bool>()));
    } else if (dtype == phi::DataType::COMPLEX64) {
      return at::Scalar(*(cpu_tensor.data<phi::dtype::complex<float>>()));
    } else if (dtype == phi::DataType::COMPLEX128) {
      return at::Scalar(*(cpu_tensor.data<phi::dtype::complex<double>>()));
    }
    PD_THROW("item(): Unsupported data type");
  }

  template <typename T>
  T item() const {
    if (tensor_.numel() != 1) {
      PD_THROW("only one element tensors can be converted to Python scalars");
    }

    // Move to CPU if necessary (for compatibility with PyTorch behavior)
    PaddleTensor cpu_tensor = tensor_;
    if (!phi::is_cpu_place(tensor_.place())) {
      PaddlePlace place(phi::AllocationType::CPU);
      cpu_tensor = tensor_.copy_to(place, true);
    }

    return *(cpu_tensor.data<T>());
  }

  at::Tensor to(
      at::ScalarType dtype,
      bool non_blocking = false,
      bool copy = false,
      ::std::optional<at::MemoryFormat> memory_format = ::std::nullopt) const {
    return to(
        at::TensorOptions().dtype(dtype), non_blocking, copy, memory_format);
  }

  Tensor toType(ScalarType t) const {
    return Tensor(paddle::experimental::cast(
        tensor_, compat::_PD_AtenScalarTypeToPhiDataType(t)));
  }

  int64_t numel() const { return tensor_.numel(); }

  c10::ScalarType dtype() const {  // Should we use `TypeMeta` here?
    return compat::_PD_PhiDataTypeToAtenScalarType(tensor_.dtype());
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

  // aten::flatten.using_ints(Tensor(a) self, int start_dim=0, int end_dim=-1)
  // -> Tensor(a)
  inline at::Tensor flatten(int64_t start_dim, int64_t end_dim) const {
    return Tensor(paddle::experimental::flatten(
        tensor_, static_cast<int>(start_dim), static_cast<int>(end_dim)));
  }

  // aten::unflatten.int(Tensor(a) self, int dim, SymInt[] sizes) -> Tensor(a)
  inline at::Tensor unflatten(int64_t dim, at::IntArrayRef sizes) const {
    // Compute the new shape by replacing the dimension at 'dim' with 'sizes'
    int64_t ndim = tensor_.dims().size();
    int64_t actual_dim = dim < 0 ? dim + ndim : dim;
    std::vector<int64_t> new_shape;
    for (int64_t i = 0; i < ndim; ++i) {
      if (i == actual_dim) {
        for (auto s : sizes) {
          new_shape.push_back(s);
        }
      } else {
        new_shape.push_back(tensor_.dims()[i]);
      }
    }
    return Tensor(paddle::experimental::reshape(tensor_, new_shape));
  }

  // aten::unflatten.int(Tensor(a) self, int dim, SymInt[] sizes) -> Tensor(a)
  inline at::Tensor unflatten_symint(int64_t dim,
                                     c10::SymIntArrayRef sizes) const {
    // SymIntArrayRef is the same as IntArrayRef in this implementation
    return unflatten(dim, sizes);
  }

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
    return phi::is_cuda_pinned_place(tensor_.place()) ||
           phi::is_xpu_pinned_place(tensor_.place());
  }

  Tensor pin_memory(
      ::std::optional<c10::Device> device = ::std::nullopt) const {
    if (is_pinned(device)) {
      return *this;
    }

    PaddlePlace current_place = tensor_.place();
    PaddlePlace pinned_place;
    if (phi::is_cpu_place(current_place)) {
      // CPU place cannot be directly converted to pinned place
      PD_THROW(
          "pin_memory: Pinning memory is not supported for CPUPlace. "
          "Please use CUDAPlace or XPUPlace tensor, or specify "
          "CUDAPinnedPlace/XPUPinnedPlace as device.");
    } else {
      // For GPU/XPU tensors, use GetPinnedPlace to get the appropriate pinned
      // place
      pinned_place = phi::GetPinnedPlace(current_place);
    }
    return tensor_.copy_to(pinned_place, true);
  }

  // aten::narrow_copy(Tensor self, int dim, SymInt start, SymInt length) ->
  // Tensor
  inline at::Tensor narrow_copy(int64_t dim,
                                int64_t start,
                                int64_t length) const {
    // narrow_copy returns a copy of the narrowed tensor
    return narrow(dim, start, length).clone();
  }

  // aten::narrow_copy(Tensor self, int dim, SymInt start, SymInt length) ->
  // Tensor
  inline at::Tensor narrow_copy_symint(int64_t dim,
                                       c10::SymInt start,
                                       c10::SymInt length) const {
    return narrow_copy(dim, start, length);
  }

  // aten::narrow(Tensor(a) self, int dim, SymInt start, SymInt length) ->
  // Tensor(a)
  inline at::Tensor narrow(int64_t dim, int64_t start, int64_t length) const {
    // Use slice to implement narrow: narrow(dim, start, length) is equivalent
    // to slice(dim, start, start + length)
    return Tensor(paddle::experimental::slice(
        tensor_, {dim}, {start}, {start + length}, {1}, {}));
  }

  // aten::narrow(Tensor(a) self, int dim, SymInt start, SymInt length) ->
  // Tensor(a)
  inline at::Tensor narrow_symint(int64_t dim,
                                  c10::SymInt start,
                                  c10::SymInt length) const {
    return narrow(dim, start, length);
  }

  // aten::narrow.Tensor(Tensor(a) self, int dim, Tensor start, SymInt length)
  // -> Tensor(a)
  inline at::Tensor narrow(int64_t dim,
                           const at::Tensor& start,
                           int64_t length) const {
    // Extract scalar value from start tensor
    PD_CHECK(start.numel() == 1,
             "start must be a 0-dim tensor or 1-element tensor");
    int64_t start_val =
        static_cast<int64_t>(start._PD_GetInner().template data<int64_t>()[0]);
    return narrow(dim, start_val, length);
  }

  // aten::narrow.Tensor(Tensor(a) self, int dim, Tensor start, SymInt length)
  // -> Tensor(a)
  inline at::Tensor narrow_symint(int64_t dim,
                                  const at::Tensor& start,
                                  c10::SymInt length) const {
    return narrow(dim, start, length);
  }

  at::Tensor reshape(at::IntArrayRef shape) const {
    return Tensor(
        paddle::experimental::reshape(tensor_, shape._PD_ToPaddleIntArray()));
  }

  at::Tensor transpose(int64_t dim0, int64_t dim1) const {
    std::vector<int> perm(tensor_.dims().size());
    for (size_t i = 0; i < perm.size(); i++) {
      perm[i] = static_cast<int>(i);
    }
    std::swap(perm[dim0], perm[dim1]);
    return Tensor(paddle::experimental::transpose(tensor_, perm));
  }

  at::Tensor permute(at::IntArrayRef dims) const {
    std::vector<int> perm(dims.size());
    for (size_t i = 0; i < dims.size(); i++) {
      perm[i] = static_cast<int>(dims[i]);
    }
    return Tensor(paddle::experimental::transpose(tensor_, perm));
  }

  at::Tensor& copy_(const at::Tensor& src, bool non_blocking = false) const {
    const_cast<PaddleTensor&>(tensor_).copy_(
        src._PD_GetInner(), tensor_.place(), /*blocking=*/!non_blocking);
    return const_cast<at::Tensor&>(*this);
  }

  at::Tensor view(at::IntArrayRef size) const {
    return Tensor(paddle::experimental::view_shape(tensor_, size.vec()));
  }

  at::Tensor view(at::ScalarType dtype) const {
    return Tensor(paddle::experimental::view_dtype(
        tensor_, compat::_PD_AtenScalarTypeToPhiDataType(dtype)));
  }

  at::Tensor squeeze() const {
    return Tensor(paddle::experimental::squeeze(tensor_, {}));
  }

  at::Tensor squeeze(int64_t dim) const {
    return Tensor(paddle::experimental::squeeze(tensor_, {dim}));
  }

  at::Tensor squeeze(at::IntArrayRef dim) const {
    return Tensor(
        paddle::experimental::squeeze(tensor_, dim._PD_ToPaddleIntArray()));
  }

  at::Tensor& squeeze_() const {
    PaddleTensor& self = const_cast<PaddleTensor&>(tensor_);
    paddle::experimental::squeeze_(self, {});
    return const_cast<at::Tensor&>(*this);
  }

  at::Tensor& squeeze_(int64_t dim) const {
    PaddleTensor& self = const_cast<PaddleTensor&>(tensor_);
    paddle::experimental::squeeze_(self, {dim});
    return const_cast<at::Tensor&>(*this);
  }

  at::Tensor& squeeze_(at::IntArrayRef dim) const {
    PaddleTensor& self = const_cast<PaddleTensor&>(tensor_);
    paddle::experimental::squeeze_(self, dim._PD_ToPaddleIntArray());
    return const_cast<at::Tensor&>(*this);
  }

  at::Tensor unsqueeze() const {
    return Tensor(paddle::experimental::unsqueeze(tensor_, {}));
  }

  at::Tensor unsqueeze(int64_t dim) const {
    return Tensor(paddle::experimental::unsqueeze(tensor_, {dim}));
  }

  at::Tensor unsqueeze(at::IntArrayRef dim) const {
    return Tensor(
        paddle::experimental::unsqueeze(tensor_, dim._PD_ToPaddleIntArray()));
  }

  at::Tensor& unsqueeze_() const {
    PaddleTensor& self = const_cast<PaddleTensor&>(tensor_);
    paddle::experimental::unsqueeze_(self, {});
    return const_cast<at::Tensor&>(*this);
  }

  at::Tensor& unsqueeze_(int64_t dim) const {
    PaddleTensor& self = const_cast<PaddleTensor&>(tensor_);
    paddle::experimental::unsqueeze_(self, {dim});
    return const_cast<at::Tensor&>(*this);
  }

  at::Tensor& unsqueeze_(at::IntArrayRef dim) const {
    PaddleTensor& self = const_cast<PaddleTensor&>(tensor_);
    paddle::experimental::unsqueeze_(self, dim._PD_ToPaddleIntArray());
    return const_cast<at::Tensor&>(*this);
  }

  at::Tensor index_select(int64_t dim, const at::Tensor& index) const {
    return Tensor(
        paddle::experimental::index_select(tensor_, index._PD_GetInner(), dim));
  }

  at::Tensor bitwise_right_shift(const Scalar& other) const {
    return Tensor(paddle::experimental::bitwise_right_shift(
        tensor_, paddle::experimental::full({}, other, other.dtype())));
  }

  at::Tensor slice(int64_t dim = 0,
                   ::std::optional<int64_t> start = ::std::nullopt,
                   ::std::optional<int64_t> end = ::std::nullopt,
                   int64_t step = 1) {
    return Tensor(paddle::experimental::slice(
        tensor_,
        {dim},
        start.has_value() ? IntArrayRef(start.value())._PD_ToPaddleIntArray()
                          : IntArrayRef()._PD_ToPaddleIntArray(),
        end.has_value() ? IntArrayRef(end.value())._PD_ToPaddleIntArray()
                        : IntArrayRef()._PD_ToPaddleIntArray(),
        {1},
        {}));
  }

  // TODO(wangyanpeng04): modify the api to
  // Tensor index(ArrayRef<at::indexing::TensorIndex> indices) const;
  at::Tensor index(const std::vector<at::indexing::Slice>& indices) const {
    std::vector<int64_t> starts(indices.size());
    std::vector<int64_t> ends(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
      starts[i] = indices[i].start();
      ends[i] = indices[i].stop();
    }
    return Tensor(
        paddle::experimental::slice(tensor_, {0, 1}, starts, ends, {1}, {})
            .contiguous());
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

  at::Tensor abs() const;

  at::Tensor& abs_() const;

  at::Tensor absolute() const { return abs(); }

  at::Tensor& absolute_() const { return abs_(); }

  Tensor operator[](int64_t index) const {
    return paddle::experimental::slice(tensor_,
                                       /*axes=*/{0},
                                       /*starts=*/{index},
                                       /*ends=*/{index + 1},
                                       /*infer_flags=*/{1},
                                       /*decrease_axis=*/{0});
  }

#ifdef PADDLE_WITH_CUDA
  void record_stream(const cudaStream_t& stream) const {
    paddle::memory::RecordStream(
        std::dynamic_pointer_cast<phi::DenseTensor>(tensor_.impl())->Holder(),
        stream);
  }
#endif

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

  PaddleTensor _PD_GetInner() const { return tensor_; }
  PaddleTensor& _PD_GetInner() { return tensor_; }
};  // NOLINT(readability/braces)
}  // namespace at
namespace torch {
using at::Tensor;
}  // namespace torch
