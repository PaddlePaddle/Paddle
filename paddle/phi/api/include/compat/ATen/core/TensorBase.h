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

#include <ATen/core/TensorAccessor.h>
#include <c10/core/Device.h>
#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>
#include <c10/core/Storage.h>
#include <c10/core/SymInt.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/core/TensorOptions.h>
#include <utils/int_array_ref_conversion.h>
#include <utils/scalar_type_conversion.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include "paddle/common/layout.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"

namespace at {
using PaddleTensor = paddle::Tensor;

class PADDLE_API TensorBase {
 public:
  TensorBase() = default;
  TensorBase(const PaddleTensor& tensor) : tensor_(tensor){};  // NOLINT
  TensorBase(const TensorBase&) = default;
  TensorBase(TensorBase&&) noexcept = default;
  ~TensorBase() noexcept = default;

#if defined(_MSC_VER)
  TensorBase& operator=(const TensorBase& x) & {
    tensor_ = x.tensor_;
    return *this;
  }
  TensorBase& operator=(TensorBase&& x) & noexcept {
    tensor_ = std::move(x.tensor_);
    return *this;
  }
#else
  TensorBase& operator=(const TensorBase& x) & = default;
  TensorBase& operator=(TensorBase&& x) & noexcept = default;
#endif

  TensorBase& operator=(const TensorBase&) && = delete;
  TensorBase& operator=(TensorBase&&) && noexcept = delete;

  bool is_same(const TensorBase& other) const {
    return tensor_.impl().get() == other.tensor_.impl().get();
  }
  size_t use_count() const { return tensor_.impl().use_count(); }
  size_t weak_use_count() const {
    // TODO(youge325) : In PyTorch, weak pointer is defined and
    // implemented in c10/util/intrusive_ptr.h, namely c10::intrusive_ptr;
    // but in Paddle, we use std::shared_ptr, so here we just return 0
    // temporarily.
    return 0;
  }

  void print() const {
    if (defined()) {
      std::cerr << '[' << toString() << ' ' << sizes() << ']' << '\n';
    } else {
      std::cerr << "[UndefinedTensor]" << '\n';
    }
  }

  std::string toString() const {
    if (!tensor_.defined()) {
      return "UndefinedType";
    }

    std::string backend_str;
    const auto& place = tensor_.place();

    // Convert place to backend string
    if (phi::is_cpu_place(place)) {
      backend_str = "CPU";
    } else if (phi::is_gpu_place(place)) {
      backend_str = "CUDA";
    } else {
      backend_str = "Undefined";
    }

    // Get scalar type string
    std::string scalar_type_str = at::toString(scalar_type());

    return backend_str + scalar_type_str + "Type";
  }

  void* data_ptr() const { return const_cast<void*>(tensor_.data()); }
  template <typename T>
  T* data_ptr() const {
    return const_cast<T*>(tensor_.data<T>());
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

  int64_t stride(int64_t dim) const {
    if (dim < 0) {
      dim += tensor_.strides().size();
    }
    return tensor_.strides()[static_cast<int>(dim)];
  }

  c10::SymInt sym_stride(int64_t dim) const {
    return static_cast<c10::SymInt>(stride(dim));
  }

  c10::IntArrayRef strides() const {
    return compat::_PD_PhiDDimToIntArrayRef(tensor_.strides());
  }

  c10::SymIntArrayRef sym_strides() const {
    return c10::SymIntArrayRef(strides());
  }

  int64_t size(int64_t dim) const {
    if (dim < 0) {
      dim += tensor_.dims().size();
    }
    return tensor_.dims()[static_cast<int>(dim)];
  }

  c10::SymInt sym_size(int64_t dim) const {
    return static_cast<c10::SymInt>(size(dim));
  }

  c10::IntArrayRef sizes() const {
    return compat::_PD_PhiDDimToIntArrayRef(tensor_.dims());
  }

  c10::SymIntArrayRef sym_sizes() const { return c10::SymIntArrayRef(sizes()); }

  int64_t numel() const { return tensor_.numel(); }

  c10::SymInt sym_numel() const { return static_cast<c10::SymInt>(numel()); }

  c10::ScalarType dtype() const {  // Should we use `caffe2::TypeMeta` here?
    return compat::_PD_PhiDataTypeToAtenScalarType(tensor_.dtype());
  }

  c10::Device device() const { return c10::Device(tensor_.place()); }
  c10::DeviceIndex get_device() const {
    return c10::Device(tensor_.place()).index();
  }

  int64_t dim() const { return tensor_.dims().size(); }
  int64_t ndimension() const { return dim(); }

  at::TensorBase contiguous(
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

  bool is_non_overlapping_and_dense() const {
    // Empty or scalar tensors are always non-overlapping and dense
    if (numel() <= 1) {
      return true;
    }

    // If the tensor is contiguous, it is non-overlapping and dense
    if (tensor_.is_contiguous()) {
      return true;
    }

    // For non-contiguous tensors, check if sorted strides form a valid dense
    // layout
    auto sizes_vec = sizes();
    auto strides_vec = strides();
    int64_t ndim = dim();

    // Create a permutation sorted by strides (ascending order)
    std::vector<int64_t> perm(ndim);
    for (int64_t i = 0; i < ndim; ++i) {
      perm[i] = i;
    }
    std::sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
      return strides_vec[a] < strides_vec[b];
    });

    // Check if sorted strides form a valid dense layout without gaps/overlaps
    int64_t expected_stride = 1;
    for (int64_t i = 0; i < ndim; ++i) {
      int64_t dim_idx = perm[i];
      if (sizes_vec[dim_idx] == 0) {
        return true;  // Empty tensor
      }
      if (sizes_vec[dim_idx] == 1) {
        continue;  // Size-1 dimensions don't affect density
      }
      if (strides_vec[dim_idx] != expected_stride) {
        return false;
      }
      expected_stride *= sizes_vec[dim_idx];
    }
    return true;
  }

  bool is_contiguous_or_false(
      at::MemoryFormat memory_format = at::MemoryFormat::Contiguous) const {
    PD_CHECK(memory_format == c10::MemoryFormat::Contiguous,
             "`MemoryFormat` other than Contiguous");

    return tensor_.is_contiguous();
  }

  c10::ScalarType scalar_type() const {
    return compat::_PD_PhiDataTypeToAtenScalarType(tensor_.dtype());
  }

  bool has_names() const {
    // In PyTorch, has_names() is used to check if any dimension has names.
    // In Paddle, we don't support named dimension yet, so always return false.
    return false;
  }

  TensorOptions options() const {
    return TensorOptions().dtype(dtype()).device(device()).layout(layout());
  }

  const TensorBase& fill_(const at::Scalar& scalar) const {
    paddle::experimental::fill_(const_cast<PaddleTensor&>(tensor_), scalar);
    return *this;
  }

  const TensorBase& zero_() const {
    paddle::experimental::fill_(const_cast<PaddleTensor&>(tensor_), 0.0);
    return *this;
  }

  at::TensorBase to(
      at::TensorOptions options = {},
      bool non_blocking = false,
      bool copy = false,
      std::optional<at::MemoryFormat> memory_format = std::nullopt) const {
    if (options.device_opt().has_value()) {
      PADDLE_THROW(common::errors::Unimplemented(
          "The `to` method with device option is not supported yet."));
    }
    if (memory_format.has_value()) {
      PADDLE_THROW(common::errors::Unimplemented(
          "The `to` method with memory_format option is not supported yet."));
    }
    return paddle::experimental::cast(
        tensor_, compat::_PD_AtenScalarTypeToPhiDataType(options.dtype()));
  }

  bool is_complex() const { return at::isComplexType(this->scalar_type()); }

  bool is_floating_point() const {
    return at::isFloatingType(this->scalar_type());
  }

  bool is_signed() const { return at::isSignedType(this->scalar_type()); }

  bool is_cpu() const { return phi::is_cpu_place(tensor_.place()); }
  bool is_cuda() const { return phi::is_gpu_place(tensor_.place()); }

  bool is_sparse() const {
    return tensor_.is_sparse_coo_tensor() || tensor_.is_sparse_csr_tensor();
  }

  bool is_sparse_csr() const { return tensor_.is_sparse_csr_tensor(); }

  at::TensorBase reshape(at::IntArrayRef shape) const {
    return TensorBase(
        paddle::experimental::reshape(tensor_, shape._PD_ToPaddleIntArray()));
  }

  at::TensorBase& copy_(const at::TensorBase& src,
                        bool non_blocking = false) const {
    const_cast<PaddleTensor&>(tensor_).copy_(
        src._PD_GetInner(), tensor_.place(), /*blocking=*/!non_blocking);
    return const_cast<at::TensorBase&>(*this);
  }

  at::TensorBase view(at::IntArrayRef size) const {
    return TensorBase(paddle::experimental::view_shape(tensor_, size.vec()));
  }

  at::TensorBase view(at::ScalarType dtype) const {
    return TensorBase(paddle::experimental::view_dtype(
        tensor_, compat::_PD_AtenScalarTypeToPhiDataType(dtype)));
  }

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

  bool defined() const { return tensor_.defined(); }

  Layout layout() const {
    switch (tensor_.layout()) {
      case common::DataLayout::STRIDED:
      case common::DataLayout::NCHW:
      case common::DataLayout::NHWC:
      case common::DataLayout::NCDHW:
      case common::DataLayout::NDHWC:
        return c10::kStrided;
      case common::DataLayout::SPARSE_COO:
        return c10::kSparse;
      case common::DataLayout::SPARSE_CSR:
        return c10::kSparseCsr;
      case common::DataLayout::ONEDNN:
        return c10::kMkldnn;
      default:
        return c10::kStrided;
    }
  }

  void reset() { tensor_.reset(); }

  int64_t storage_offset() const {
    // Paddle DenseTensor stores offset in meta_.offset (in bytes)
    // We need to convert to element offset
    auto dense_tensor =
        std::dynamic_pointer_cast<phi::DenseTensor>(tensor_.impl());
    if (dense_tensor) {
      size_t byte_offset = dense_tensor->meta().offset;
      size_t element_size = SizeOf(tensor_.dtype());
      return element_size > 0 ? static_cast<int64_t>(byte_offset / element_size)
                              : 0;
    }
    return 0;
  }

  c10::SymInt sym_storage_offset() const {
    return c10::SymInt(storage_offset());
  }

  bool has_storage() const { return tensor_.defined(); }

  const Storage storage() const {
    return Storage(
        std::dynamic_pointer_cast<phi::DenseTensor>(tensor_.impl())->Holder());
  }

  bool is_alias_of(const at::TensorBase& other) const {
    return this->storage().allocation() == other.storage().allocation();
  }

  // Return a `TensorAccessor` for CPU `Tensor`s. You have to specify scalar
  // type and
  // dimension.
  template <typename T, size_t N>
  TensorAccessor<T, N> accessor() const& {
    static_assert(
        N > 0,
        "accessor is used for indexing tensor, for scalars use *data_ptr<T>()");
    TORCH_CHECK(dim() == N,
                "TensorAccessor expected ",
                N,
                " dims but tensor has ",
                dim());
    T* ptr = nullptr;
    if constexpr (std::is_const_v<T>) {
      ptr = const_data_ptr<T>();
    } else {
      ptr = mutable_data_ptr<T>();
    }
    return TensorAccessor<T, N>(ptr, sizes().data(), strides().data());
  }
  template <typename T, size_t N>
  TensorAccessor<T, N> accessor() && = delete;

  // Return a `GenericPackedTensorAccessor` for CUDA `Tensor`s. You have to
  // specify scalar type and dimension. You can optionally specify
  // RestrictPtrTraits as a template parameter to cast the data pointer to a
  // __restrict__ pointer. In order to use this, your CUDA kernel has to take a
  // corresponding GenericPackedTensorAccessor as an argument.
  template <typename T,
            size_t N,
            template <typename U> class PtrTraits = DefaultPtrTraits,
            typename index_t = int64_t>
  GenericPackedTensorAccessor<T, N, PtrTraits, index_t>
  generic_packed_accessor() const& {
    static_assert(
        N > 0,
        "accessor is used for indexing tensor, for scalars use *data_ptr<T>()");
    TORCH_CHECK(dim() == N,
                "TensorAccessor expected ",
                N,
                " dims but tensor has ",
                dim());
    T* ptr = nullptr;
    if constexpr (std::is_const_v<T>) {
      ptr = const_data_ptr<T>();
    } else {
      ptr = mutable_data_ptr<T>();
    }
    return GenericPackedTensorAccessor<T, N, PtrTraits, index_t>(
        static_cast<typename PtrTraits<T>::PtrType>(ptr),
        sizes().data(),
        strides().data());
  }
  template <typename T,
            size_t N,
            template <typename U> class PtrTraits = DefaultPtrTraits,
            typename index_t = int64_t>
  GenericPackedTensorAccessor<T, N> generic_packed_accessor() && = delete;

  template <typename T,
            size_t N,
            template <typename U> class PtrTraits = DefaultPtrTraits>
  PackedTensorAccessor32<T, N, PtrTraits> packed_accessor32() const& {
    TORCH_CHECK(
        numel() <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
        "numel needs to be smaller than int32_t max; otherwise, please use "
        "packed_accessor64");
    return generic_packed_accessor<T, N, PtrTraits, int32_t>();
  }
  template <typename T,
            size_t N,
            template <typename U> class PtrTraits = DefaultPtrTraits>
  PackedTensorAccessor32<T, N, PtrTraits> packed_accessor32() && = delete;

  template <typename T,
            size_t N,
            template <typename U> class PtrTraits = DefaultPtrTraits>
  PackedTensorAccessor64<T, N, PtrTraits> packed_accessor64() const& {
    return generic_packed_accessor<T, N, PtrTraits, int64_t>();
  }
  template <typename T,
            size_t N,
            template <typename U> class PtrTraits = DefaultPtrTraits>
  PackedTensorAccessor64<T, N, PtrTraits> packed_accessor64() && = delete;

  const PaddleTensor& _PD_GetInner() const& { return tensor_; }
  PaddleTensor& _PD_GetInner() & { return tensor_; }
  PaddleTensor&& _PD_GetInner() && { return std::move(tensor_); }

 protected:
  PaddleTensor tensor_;
};

}  // namespace at
