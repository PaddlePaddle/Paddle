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

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <utils/int_array_ref_conversion.h>
#include <utils/scalar_type_conversion.h>
#include "paddle/common/layout.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/place.h"

namespace at {
using PaddleTensor = paddle::Tensor;

class PADDLE_API TensorBase {
 public:
  TensorBase() = default;
  TensorBase(const PaddleTensor& tensor) : tensor_(tensor){};  // NOLINT

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
  c10::IntArrayRef strides() const {
    return compat::_PD_PhiDDimToIntArrayRef(tensor_.strides());
  }

  int64_t size(int64_t dim) const {
    return tensor_.dims()[static_cast<int>(dim)];
  }

  c10::IntArrayRef sizes() const {
    return compat::_PD_PhiDDimToIntArrayRef(tensor_.dims());
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

  c10::ScalarType scalar_type() const {
    return compat::_PD_PhiDataTypeToAtenScalarType(tensor_.dtype());
  }

  c10::TensorOptions options() const {
    // TODO(SigureMo): Implement layout
    return c10::TensorOptions().dtype(dtype()).device(device());
  }

  const TensorBase& fill_(const at::Scalar& scalar) const {
    paddle::experimental::fill_(const_cast<PaddleTensor&>(tensor_), scalar);
    return *this;
  }

  const TensorBase& zero_() const {
    paddle::experimental::fill_(const_cast<PaddleTensor&>(tensor_), 0.0);
    return *this;
  }

  bool is_cpu() const { return phi::is_cpu_place(tensor_.place()); }
  bool is_cuda() const { return phi::is_gpu_place(tensor_.place()); }

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

  PaddleTensor _PD_GetInner() const { return tensor_; }
  PaddleTensor& _PD_GetInner() { return tensor_; }

 protected:
  PaddleTensor tensor_;
};

}  // namespace at
