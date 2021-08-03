/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include <functional>
#include <memory>
#include <utility>

#include "paddle/pten/core/autograd_meta_interface.h"
#include "paddle/pten/core/tensor_impl_interface.h"

// fluid headers [may be replaced by new impl]
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/platform/place.h"

namespace pt {

/**
 * Tensor is the API description of the basic data structure in the
 * [ PaddlePaddle Tensor Operation Library ].
 *
 * It is not limited to a simple n-dimensional array.
 * It contains a smart pointer to `TensorImpl`. The data description contained
 * in Tensor is defined by TensorImpl. Tensor only defines the interface for
 * operation.
 *
 * This is a new Tensor design, which is independent of the original
 * framework::Tensor in fluid. The original Tensor will be gradually discarded
 * in the future.
 *
 * Note: Tensor can be NULL state, Tensor is meaningful only when the
 * TensorImpl to which it is pointed is not empty.
 *
 * Note: For the consistency of C++ API self, and the consistency between C++
 * API and Python API, all member methods of Tensor are named with lowercase
 * letters and underscores.
 *
 * Note: Tensor cannot be inherited. The heterogeneous Tensor implementation
 * can be achieved by inheriting the underlying TensorImplInterface.
 */

class Tensor final {
 public:
  /* Part 1: Construction and destruction methods */
  Tensor() {}
  Tensor(const Tensor&) = default;
  Tensor(Tensor&&) = default;

  /**
   * @description: Use a TensorImpl pointer to construct a Tensor
   * @param {shared_ptr<TensorImplInterface>} tensor_impl
   * @return {Tensor}
   */
  explicit Tensor(std::shared_ptr<TensorImplInterface> tensor_impl)
      : impl_(std::move(tensor_impl)) {
    if (impl_.get() == nullptr) {
      throw std::runtime_error("TensorImpl with nullptr is not supported");
    }
  }

  /* Part 2: Dimension, DataType and Layout methods */
  /**
   * @description: Return the number of elements of current Tensor.
   * @param None
   * @return {int64_t}
   */
  int64_t numel() const { return impl_->numel(); }

  /**
   * @description: Return the shape (dimensions) of current Tensor.
   * @param None
   * @return {DDim}
   */
  DDim shape() const { return impl_->dims(); }

  /**
   * @description: Resize the shape (dimensions) of current Tensor.
   * @param {const} DDim
   * @return {*}
   */
  void resize(const DDim& dims) { impl_->resize(dims); }

  /**
   * @description: Return the data type of current Tensor.
   * @param None
   * @return {DataType}
   */
  DataType type() const { return impl_->type(); }

  /**
   * @description: Return the layout of current Tensor.
   * @param None
   * @return {Layout}
   */
  Layout layout() const { return impl_->layout(); }

  /* Part 3: Device and Backend methods */
  /**
   * @description: Return the place (device) of current Tensor.
   * @param None
   * @return {Place}
   */
  Place place() const { return impl_->place(); }

  /**
   * @description: Convert the current Tensor to a Tensor of
   *               a specific data type for a specific device
   * @param {const} Backend
   * @param {const} DataType
   * @return {*}
   */
  // Tensor to(const Backend& backend, const DataType& dtype) {
  //   // TODO(chenweihang): use kernels to impl later
  // }

  /**
   * Backend judgment APIs, shield the concept of Backend.
   */
  // TODO(chenweihang): impl later
  bool is_cpu() const { return impl_->backend() == Backend::kCPU; }
  bool is_cuda() const;
  bool is_hip() const;
  bool is_xpu() const;
  bool is_npu() const;
  bool is_mkldnn() const;
  bool is_cudnn() const;

  /**
   * Backend convert APIs.
   */
  Tensor cpu() const;
  Tensor cuda() const;
  Tensor hip() const;
  Tensor xpu() const;
  Tensor npu() const;
  Tensor mkldnn() const;
  Tensor cudnn() const;

  /* Part 4: Data Access methods */
  /**
   * @description: Return the implemention of current Tensor.
   * @param None
   * @return {std::shared_ptr<TensorImplInterface>}
   */
  std::shared_ptr<TensorImplInterface> impl() const { return impl_; }

  /**
   * @description: Get the const memory pointer of current Tensor.
   * @param None
   * @return {const T*}
   */
  template <typename T>
  const T* data() const {
    return impl_->data<T>();
  }

  /**
   * @description: Get the mutable memory pointer of current Tensor.
   * @param None
   * @return {T*}
   */
  template <typename T>
  T* mutable_data() {
    return impl_->mutable_data<T>();
  }

  // TODO(chenweihang): slice and split methods use kernels?

  /* Part 5: Status utils methods */
  /**
   * @description: Determine whether it is a meaningful Tensor
   * @param None
   * @return {bool}
   */
  bool defined() const { return impl_ != nullptr; }

  /**
   * @description: Determine whether Tensor is initialized
   * @param None
   * @return {bool}
   */
  bool initialized() const { return impl_->initialized(); }

  /**
   * @description: Reset the Tensor implementation
   * @param None
   * @return {void}
   */
  void reset() { impl_.reset(); }

  /* Part 6: Operator overloading */
  Tensor& operator=(const Tensor& x) & {
    impl_ = x.impl_;
    return *this;
  }
  Tensor& operator=(Tensor&& x) & {
    impl_ = std::move(x.impl_);
    return *this;
  }
  // TODO(chenweihang): impl later
  Tensor& operator=(const Tensor&) &&;
  Tensor& operator=(Tensor&&) &&;

  /* Part 7: Autograd methods */
  // TODO(yangjiabin): Design autograd methods

  /* Part 8: Auto generated Tensor methods */
  // ...

 private:
  /**
   * [ Why use abstract TensorImpl interface here? ]
   *
   * We hope that the data structure at the API level of the framework can be
   * unified to Tensor, but Tensor itself is heterogeneous.
   *
   * Tensor can generally be represented by void* and size_t, place.
   * This is suitable for most scenarios including CPU, CUDA, HIP, CPU, etc.,
   * but there are a few cases where this definition cannot be described,
   * such as the Tensor representation in third-party lib such as Metal,
   * OpenCL, etc., as well as some special Tensor implementations, including
   * Tensor containing only one Scalar value, or Tensor representing String,
   * etc.
   *
   * Therefore, we hope to use a unified interface to shield the underlying
   * heterogeneous Tensor implementation, so that the API level can be unified
   * to one `Tensor`.
   */
  std::shared_ptr<TensorImplInterface> impl_;

  /**
   * [ Why need abstract AutogradMetaInterface here? ]
   *
   * Dynamic graphs need to hold backward information
   *
   * [ Why AutogradMeta not in TensorImpl? ]
   *
   * 1. AutogradMeta is only used in dynamic graph, It is execution-related
   *    information, not Tensor data description-related information.
   * 2. Kernel calculation does not require AutogradMeta.
   */
  std::unique_ptr<AutogradMetaInterface> autograd_meta_ = nullptr;
};

}  // namespace pt
