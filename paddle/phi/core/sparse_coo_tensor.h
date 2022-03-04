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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/tensor_base.h"
#include "paddle/phi/core/tensor_meta.h"

namespace phi {

/// \brief The SparseCooTensor uses two DenseTensors to represent
/// the non zero elements and the indices of non zero elements of
/// original DenseTensor.
/// where non_zero_elements_ represents the non zero elements of original
/// DenseTensor.
/// non_zero_indices_ represents the indices of non zero elements in original
/// DenseTensor.
class SparseCooTensor : public TensorBase,
                        public TypeInfoTraits<TensorBase, SparseCooTensor> {
 public:
  /// \brief Create the sparse coo tensor
  /// \param non_zero_indices The indices of non zero elements in original dense
  /// tensor.
  /// \param non_zero_elements The non zero elements of original dense tensor.
  /// \param dims The dims of original dense tensor.
  SparseCooTensor(const DenseTensor& non_zero_indices,
                  const DenseTensor& non_zero_elements,
                  const DDim& dims);

  /// \brief Create the sparse coo tensor
  /// \param non_zero_indices The indices of non zero elements in original dense
  /// tensor.
  /// \param non_zero_elements The non zero elements of original dense tensor.
  /// \param dims The dims of original dense tensor.
  SparseCooTensor(DenseTensor&& non_zero_indices,
                  DenseTensor&& non_zero_elements,
                  const DDim& dims);

  /// \brief SparseCooTensor shallow copy constructor.
  SparseCooTensor(const SparseCooTensor& other);

  /// \brief move constructor
  SparseCooTensor(SparseCooTensor&& other);

  /// \brief SparseCooTensor shallow copy assignment.
  SparseCooTensor operator=(const SparseCooTensor& other);

  /// \brief Destroy the tensor object and release exclusive resources.
  virtual ~SparseCooTensor() = default;

  /// \brief Returns the indices of non zero elemetns in original dense tensor.
  /// \return The indices of non zero elemetns in original dense tensor.
  const DenseTensor& non_zero_indices() const { return non_zero_indices_; }

  /// \brief Returns the non zero elemetns in original dense tensor.
  /// \return The non zero elemetns in original dense tensor.
  const DenseTensor& non_zero_elements() const { return non_zero_elements_; }

  /// \brief Returns whether the indices has coalesced
  /// \return whether the indices has coalesced
  bool coalesced() const { return coalesced_; }

  /// \brief Set the coalesced
  /// \param coalesced whether the indices has coalesced
  void SetCoalesced(const bool coalesced) { coalesced_ = coalesced; }

  /// \brief Returns the name of the class for type traits.
  /// \return The name of the class.
  static const char* name() { return "SparseCooTensor"; }

  /// \brief Returns the total number of non zero elements in original
  /// DenseTensor
  int64_t nnz() const;

  /// \brief Return the number of elements contained in original dense tensor
  /// \return The number of elements contained in original dense tensor
  int64_t numel() const override { return product(dims_); }

  /// \brief Returns the dims of the original dense tensor.
  /// \return The dims of the original dense tensor.
  const DDim& dims() const noexcept override { return dims_; }

  /// \brief Returns the data type of the tensor.
  /// \return The data type of the tensor.
  DataType dtype() const noexcept override {
    return non_zero_elements_.dtype();
  }

  /// \brief Returns the data layout of the tensor.
  /// \return The data layout of the tensor.
  DataLayout layout() const noexcept override { return DataLayout::SPARSE_COO; }

  /// \brief Returns the data place of the tensor.
  /// \return The data place of the tensor.
  const Place& place() const override { return non_zero_elements_.place(); }

  /// \brief Test whether the non_zero_elements_ metadata is valid.
  /// \return Whether the non_zero_elements_ metadata is valid.
  bool valid() const noexcept override { return non_zero_elements_.valid(); }

  /// \brief Test whether the non_zero_elements_ storage is allocated.
  /// return Whether the non_zero_elements_ storage is allocated.
  bool initialized() const override { return non_zero_elements_.initialized(); }

  /// \brief resize sparse coo tensor.
  /// \param dense_dims The dims of original dense tensor.
  /// \param sparse_dim number of sparse dimensions
  /// \param non_zero_num The total number of non zero element
  void Resize(const DDim& dense_dim,
              const int64_t sparse_dim,
              const int64_t non_zero_num);

  /// \brief set the member of sparse coo tensor.
  /// \param non_zero_indices The indices of non zero elements in original dense
  /// tensor.
  /// \param non_zero_elements The non zero elements of original dense tensor.
  /// \param dims The dims of original dense tensor.
  /// \param coalesced whether the indices has coalesced.
  void SetMember(const DenseTensor& non_zero_indices,
                 const DenseTensor& non_zero_elements,
                 const DDim& dims,
                 const bool coalesced = false);

  /// \brief Get a mutable pointer of non_zero_indices_.
  /// return a mutable pointer of non_zero_indices_.
  DenseTensor* mutable_non_zero_indices() { return &non_zero_indices_; }

  /// \brief Get a mutable pointer of non_zero_elements.
  /// return a mutable pointer of non_zero_elements.
  DenseTensor* mutable_non_zero_elements() { return &non_zero_elements_; }

  /// \brief This function is not recommended
  void* AllocateFrom(Allocator* allocator,
                     DataType dtype,
                     size_t requested_size = 0) override;
  void set_dims(const DDim& dims) { this->dims_ = dims; }

 private:
  // save the indices of non zero elements in original dense tensor
  DenseTensor non_zero_indices_;
  // save the non zero elements of original dense tensor
  DenseTensor non_zero_elements_;
  /// whether the indices has coalesced
  bool coalesced_ = false;
  // save the number of non zero elements in each batch
  DDim dims_;
  /* --------------------------- */
  /*   example: non zero element is scalar */
  /* --------------------------- */
  /*
     dense_x = [[0, 1, 0, 0],
                [2, 0, 0, 3],
                [0, 0, 4, 0],
                [0, 5, 0, 6]]
     dims_ = (4, 4)
     non_zero_elements_ = [1, 2, 3, 4, 5 ,6]
     non_zero_indices_ = [[0, 1, 1, 2, 3, 3],
                          [1, 0, 3, 2, 1, 3]]
   */
  /* --------------------------- */
  /*   example: non zero element is tensor */
  /* --------------------------- */
  /*
     dense_x = [[0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 4, 0],
                [0, 0, 0, 0]]
     dims_ = (4, 4)
     non_zero_elements_ = [[0, 1, 0, 0], [0, 0, 4, 0]]
     non_zero_indices_ = [0, 2],
   */
};

}  // namespace phi
