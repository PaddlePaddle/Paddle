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
#include <vector>

#ifdef PADDLE_WITH_CUDA
#include <cuda_runtime.h>
using gpuStream_t = cudaStream_t;
#endif

#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
using gpuStream_t = hipStream_t;
#endif

#include "paddle/phi/api/include/dll_decl.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/common/place.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace phi {
class TensorBase;
class DDim;
}  // namespace phi

namespace paddle {

namespace experimental {

class AbstractAutogradMeta {
 public:
  // No AbstractAutogradMeta should be created
  virtual ~AbstractAutogradMeta() {}
};

/**
 * Tensor is the API description of the basic data structure in the
 * [ "Paddle Tensor Operation (phi)" Library ].
 *
 * It is not limited to a simple n-dimensional array.
 * It contains a smart pointer to `TensorImpl`. The data description contained
 * in Tensor is defined by TensorImpl. Tensor only defines the interface for
 * computation.
 *
 * This is a new Tensor design, which is independent of the original
 * phi::DenseTensor in fluid. The original Tensor will be gradually discarded
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
 * can be achieved by inheriting the underlying TensorBase.
 *
 * Note: This Tensor API is suitable for training and custom operators,
 * another simple Tensor design may be required for inference.
 */

class PADDLE_API Tensor final {
 public:
  /* Part 1: Construction and destruction methods */

  /**
   * @brief Construct a new Tensor object
   */
  Tensor() = default;

  /**
   * @brief Construct a new Tensor object by copy
   */
  Tensor(const Tensor&) = default;

  /**
   * @brief Construct a new Tensor object by move
   */
  Tensor(Tensor&&) = default;

  /**
   * @brief Construct a new Tensor object by a TensorBase pointer
   *
   * @param tensor_impl
   */
  explicit Tensor(std::shared_ptr<phi::TensorBase> tensor_impl);

  /**
   * @brief Construct a new Tensor object on the target place.
   *
   * This is a deprecated method and may be removed in the future!!!
   *
   * @param place
   */
  explicit Tensor(const Place& place);

  /**
   * @brief Construct a new Tensor object on the target place
   * with specified shape.
   *
   * This is a deprecated method and may be removed in the future!!!
   *
   * @param place
   * @param shape
   */
  Tensor(const Place& place, const std::vector<int64_t>& shape);

  /**
   * @brief Construct a new Tensor object by a TensorBase pointer and name
   *
   * @param tensor_impl
   */
  Tensor(std::shared_ptr<phi::TensorBase> tensor_impl, const std::string& name);

  /**
   * @brief Construct a new Tensor object with name
   *
   * @note Internal method, used to adapt original execution mechanism and
   * debug analysis in the development of new dygraph. It may be removed in
   * the future.
   * */
  explicit Tensor(const std::string& name) : name_(name) {}

  /* Part 2: Dimension, DataType and DataLayout methods */

  /**
   * @brief Return the number of elements of Tensor.
   *
   * @return int64_t
   */
  int64_t numel() const;

  /**
   * @brief Get the size of current tensor.
   *
   * The compatible method of `Tensor::numel()`.
   * This is a deprecated method and may be removed in the future!
   *
   * @return int64_t
   */
  int64_t size() const;

  /**
   * @brief Return the dimensions of Tensor.
   *
   * @return phi::DDim
   */
  const phi::DDim& dims() const;

  /**
   * @brief Return the shape (dimensions) of Tensor.
   *
   * The compatible method of `Tensor::dims()`.
   * This is a deprecated method and may be removed in the future!
   *
   * @return std::vector<int64_t>
   */
  std::vector<int64_t> shape() const;

  /**
   * @brief Reset the shape of the tensor.
   * @note: This method means Reset the shape of the tensor,
   * and must be called before calling mutable_data() or
   * copy_to(const Place& place), this is not a standard definition of
   * reshape behavior, so we will deprecated this feature in the future.
   *
   * @param shape
   */
  void reshape(const std::vector<int64_t>& shape);

  /**
   * @brief Return the data type of Tensor.
   *
   * @return DataType
   */
  DataType dtype() const;

  /**
   * @brief Return the data type of Tensor.
   *
   * The compatible method of `Tensor::dtype()`.
   * This is a deprecated method and may be removed in the future!
   *
   * @return DataType
   */
  DataType type() const;

  /**
   * @brief Return the layout of Tensor.
   *
   * @return DataLayout
   */
  DataLayout layout() const;

  /**
   * @brief Determine whether tensor is DenseTensor
   *
   * @return true
   * @return false
   */
  bool is_dense_tensor() const;

  /**
   * @brief Determine whether tensor is SelectedRows
   *
   * @return true
   * @return false
   */
  bool is_selected_rows() const;

  /**
   * @brief Determine whether tensor is SparseCooTensor
   *
   * @return true
   * @return false
   */
  bool is_sparse_coo_tensor() const;

  /**
   * @brief Determine whether tensor is SparseCsrTensor
   *
   * @return true
   * @return false
   */
  bool is_sparse_csr_tensor() const;

  /**
   * @brief Determine whether tensor is StringTensor
   *
   * @return true
   * @return false
   */
  bool is_string_tensor() const;

  /* Part 3: Device and Backend methods */

  /**
   * @brief Return the place (device) of Tensor.
   *
   * @return Place
   */
  const Place& place() const;

  /**
   * @brief Determine whether the tensor device is CPU
   *
   * @return true
   * @return false
   */
  bool is_cpu() const;

  /**
   * @brief Determine whether the tensor device is GPU
   *
   * @return true
   * @return false
   */
  bool is_gpu() const;

  /**
   * @brief Determine whether the tensor device is GPU_PINNED
   *
   * @return true
   * @return false
   */
  bool is_gpu_pinned() const;

  /**
   * @brief Determine whether the tensor device is XPU
   *
   * @return true
   * @return false
   */
  bool is_xpu() const;

  /**
   * @brief Determine whether the tensor device is CustomDevice
   *
   * @return true
   * @return false
   */
  bool is_custom_device() const;

  /* Part 4: Data Access methods */

  /**
   * @brief Get the memory pointer in CPU or GPU with specific data type.
   * It's usually used to get the output data pointer, same as the T* data().
   *
   * @tparam T
   * @return T*
   */
  template <typename T>
  T* mutable_data();

  /**
   * @brief Get the memory pointer in CPU or GPU with specific data type.
   *
   * It's usually used to get the output data pointer.
   * This is a deprecated method and may be removed in the future!
   *
   * @tparam T
   * @param place
   * @return T*
   */
  template <typename T>
  T* mutable_data(const Place& place);

  /**
   * @brief Get the const memory pointer directly.
   * It's usually used to get the output data pointer.
   *
   * @tparam T
   * @return T*
   */
  template <typename T>
  const T* data() const;

  /**
   * @brief Get the memory pointer directly.
   * It's usually used to get the mutable output data pointer.
   *
   * @tparam T
   * @return T*
   */
  template <typename T>
  T* data();

  /**
   * @brief Return a sub-tensor of the given tensor.
   * It is usually used to extract a sub-tensor (which supports
   * modifying the data of the original tensor) to perform further
   * operations.
   *
   * @param begin_idx The index of the start row (inclusive) to slice.
   *                  The index number begins from 0.
   * @param end_idx The index of the end row (exclusive) to slice.
   *                 The index number begins from begin_idx + 1.
   * @return Tensor
   */
  Tensor slice(int64_t begin_idx, int64_t end_idx) const;

  /**
   * @brief Return the implemention of current Tensor.
   *
   * @return std::shared_ptr<phi::TensorBase>
   */
  const std::shared_ptr<phi::TensorBase>& impl() const;

  /**
   * @brief Set the implemention of current Tensor.
   *
   * @param impl
   */
  void set_impl(const std::shared_ptr<phi::TensorBase>& impl);

  /**
   * @brief Set the implemention of current Tensor.
   *
   * @param impl
   */
  void set_impl(std::shared_ptr<phi::TensorBase>&& impl);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  /**
   * @brief Get the stream where the tensor is currently located
   * This is a deprecated method and may be removed in the future!
   *
   * @return gpuStream_t
   */
  gpuStream_t stream() const;
#endif

  /**
   * @brief Return the name of Tensor.
   * @note Used to adapt original execution mechanism and debug analysis
   * in the development of new dygraph. It may be removed in the future.
   *
   * @return const std::string&
   */
  const std::string& name() const { return name_; }

  /**
   * @brief Set name of Tensor.
   * @note Used to adapt original execution mechanism and debug analysis
   * in the development of new dygraph. It may be removed in the future.
   *
   * @param const std::string& name
   */
  void set_name(const std::string& name) { name_ = name; }

  /* Part 5: Data Transform methods */
  /* Alert!!!!: All copy method can only deep copy impl, autograd info only be
   * copied */
  /* out of phi */
  /**
   * @brief Copy the current Tensor data to the specified device
   * and return the new Tensor. It's usually used to set the input tensor data.
   * @note The Tensor's `copy_to` method is deprecated since version 2.3, and
   * will be removed in version 2.4, please use `copy_to` method without
   * template argument instead.
   * reason: copying a Tensor to another device does not need to specify the
   * data type template argument
   *
   * @tparam T
   * @param target_place, the target place of which the tensor will copy to.
   * @return Tensor
   */
  template <typename T>
  Tensor copy_to(const Place& target_place) const;

  /**
   * @brief Transfer the current Tensor to the specified device and return.
   *
   * @param place, The target place of which the tensor will copy to.
   * @param blocking, Should we copy this in sync way.
   * @return Tensor
   */
  Tensor copy_to(const Place& place, bool blocking) const;

  /**
   * @brief Transfer the source Tensor to current Tensor.
   *
   * @param src, the source Tensor to be copied.
   * @param blocking, Should we copy this in sync way.
   * @return void
   */
  void copy_(const Tensor& src, const Place& target_place, bool blocking);

  /**
   * @brief Cast datatype from one to another
   *
   * @param target_type
   * @return Tensor
   */
  Tensor cast(DataType target_type) const;

  /* Part 6: Status utils methods */

  /**
   * @brief Determine whether it is a meaningful Tensor
   *
   * @return true
   * @return false
   */
  bool defined() const;

  /**
   * @brief Determine whether Tensor is initialized.
   *
   * @return true
   * @return false
   */
  bool initialized() const;

  /**
   * @brief Determine whether Tensor is initialized.
   * This is a deprecated method and may be removed in the future!
   *
   * @return true
   * @return false
   */
  bool is_initialized() const;

  /**
   * @brief Reset the Tensor implementation
   */
  void reset();

  /* Part 7: Operator overloading */

  /**
   * @brief Assignment operator
   *
   * @param x
   * @return Tensor&
   */
  Tensor& operator=(const Tensor& x) &;

  /**
   * @brief Move assignment operator
   *
   * @param x
   * @return Tensor&
   */
  Tensor& operator=(Tensor&& x) &;

  /* Part 8: Autograd methods */

  /**
   * @brief Get the autograd meta object pointer
   *
   * @return AbstractAutogradMeta*
   */
  AbstractAutogradMeta* get_autograd_meta() const;

  /**
   * @brief Get the shared pointer of autograd meta object
   *
   * @return std::shared_ptr<AbstractAutogradMeta>&
   */
  const std::shared_ptr<AbstractAutogradMeta>& mutable_autograd_meta() const;

  /**
   * @brief Set the autograd meta object
   *
   * @param autograd_meta
   */
  void set_autograd_meta(std::shared_ptr<AbstractAutogradMeta> autograd_meta);

  /* Part 9: Inplace methods */

  /**
   * @brief Increase inplace version
   */
  void bump_inplace_version();

  /**
   * @brief Get current inplace version
   *
   * @return uint32_t
   */
  uint32_t current_inplace_version();

  /**
   * @brief Reset inplace version
   */
  void reset_inplace_version(bool set_to_zero = false);

  /* Part 10: Auto generated Tensor methods */

  /* Part 11: Methods of converting underlying TensorType to each other
   */
  /**
   * @brief Convert DenseTensor or SparseCsrTensor to SparseCooTensor
   *
   * @param sparse_dim, The number of sparse dimensions
   * @return Tensor
   */
  Tensor to_sparse_coo(const int64_t sparse_dim) const;

  /**
   * @brief Convert DenseTensor or SparseCooTensor to SparseCsrTensor
   *
   * @return Tensor
   */
  Tensor to_sparse_csr() const;

  /**
   * @brief Convert SparseCooTensor or SparseCsrTensor to DenseTensor
   *
   * @return Tensor
   */
  Tensor to_dense() const;

 private:
  /**
   * [ Why use abstract TensorImpl interface here? ]
   *
   * We hope that the data structure at the API level of the framework can be
   * unified to Tensor, but Tensor itself is heterogeneous.
   *
   * Tensor can generally be represented by void* and size_t, place.
   * This is suitable for most scenarios including CPU, GPU, HIP, NPU, etc.,
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
  std::shared_ptr<phi::TensorBase> impl_{nullptr};

  /**
   * [ Why need abstract AbstractAutogradMeta here? ]
   *
   * Dynamic graphs need to hold backward information
   *
   * [ Why AutogradMeta not in TensorImpl? ]
   *
   * 1. AutogradMeta is only used in dynamic graph, It is execution-related
   *    information, not Tensor data description-related information.
   * 2. Kernel calculation does not require AutogradMeta.
   */
  std::shared_ptr<AbstractAutogradMeta> autograd_meta_{nullptr};

  /**
   * Tensor name: used to adapt original execution mechanism and debug analysis
   * in the development of new dygraph. It may be removed in the future.
   */
  std::string name_{""};
};

}  // namespace experimental
}  // namespace paddle
