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

#include "paddle/tcmpt/core/tensor_interface.h"

/**
 * [ Why still include the fluid headers? ]
 *
 * We hope to organize the basic implementation of Tensor and the logic related
 * to Tensor computation into an independent library, which we call
 * [Tensor Compute Library, tcmpt], so we extract or rewrite the original
 * Kernels.
 *
 * In the future, the training library, inference library and custom operators
 * will link to this Tensor Compute library.
 *
 * However, if we directly split the link relation, we need to make too many
 * changes, which will affect the stability of the framework, so here we still
 * rely on the implementation of the framework, which is a intermediate state.
 *
 * In the future, the necessary components will be moved to the this library,
 * or the corresponding components will be re-implemented.
 */
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/platform/place.h"

namespace pt {

class Tensor;

class AbstractAutogradMeta {
 public:
  // No AbstractAutogradMeta should be created
  virtual ~AbstractAutogradMeta() {}
};

/**
 * Tensor is the API description of the basic data structure in the
 * [ Paddle "Tensor CoMPuTe (tcmpt)" Library ].
 *
 * It is not limited to a simple n-dimensional array.
 * It contains a smart pointer to `TensorImpl`. The data description contained
 * in Tensor is defined by TensorImpl. Tensor only defines the interface for
 * computation.
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
 * can be achieved by inheriting the underlying TensorInterface.
 *
 * Note: This Tensor API is suitable for training and custom operators,
 * another simple Tensor design may be required for inference.
 */

class Tensor final {
 public:
  /* Part 1: Construction and destruction methods */
  Tensor() {}
  Tensor(const Tensor&) = default;
  Tensor(Tensor&&) = default;

  /**
   * @description: Use a TensorImpl pointer to construct a Tensor
   * @param {shared_ptr<TensorInterface>} tensor_impl
   * @return {Tensor}
   */
  explicit Tensor(std::shared_ptr<TensorInterface> tensor_impl)
      : impl_(std::move(tensor_impl)) {
    if (impl_.get() == nullptr) {
      throw std::runtime_error("TensorImpl with nullptr is not supported");
    }
  }

  /* Part 2: Dimension, DataType and DataLayout methods */
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
   * @description: Return the data type of current Tensor.
   * @param None
   * @return {DataType}
   */
  DataType type() const { return impl_->type(); }

  /**
   * @description: Return the layout of current Tensor.
   * @param None
   * @return {DataLayout}
   */
  DataLayout layout() const { return impl_->layout(); }

  /* Part 3: Device and Backend methods */
  /**
   * @description: Return the place (device) of current Tensor.
   * @param None
   * @return {Place}
   */
  Place place() const { return impl_->place(); }

  /**
   * Backend judgment APIs, shield the concept of Backend.
   */
  bool is_cpu() const { return impl_->backend() == Backend::kCPU; }
  bool is_cuda() const { return impl_->backend() == Backend::kCUDA; }
  bool is_hip() const;
  bool is_xpu() const;
  bool is_npu() const;
  bool is_mkldnn() const;
  bool is_cudnn() const;

  bool is_selected_rows() const;

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
   * @return {std::shared_ptr<TensorInterface>}
   */
  std::shared_ptr<TensorInterface> impl() const { return impl_; }

  // Whether API Tensor need `data` and `mutable_data`?

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
  std::shared_ptr<TensorInterface> impl_;

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
