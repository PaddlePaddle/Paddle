/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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
#include <memory>
#include <mutex>  // NOLINT
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/utils/rw_lock.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/framework/mixed_vector.h"
namespace phi {
class SelectedRowsImpl {
  /*
   * @brief We can use the SelectedRowsImpl structure to reproduce a sparse
   * table.
   *  A sparse table is a key-value structure that the key is an `int64_t`,
   *  and the value is a Tensor which the first dimension is 0.
   *  You can use the following interface to operate the sparse table, and you
   * can find
   *  some detail information from the comments of each interface:
   *
   *  HasKey(key), whether the sparse table has the specified key.
   *  Set(key, value), set a key-value pair into the sparse table.
   *  Get(keys, value*), get value by given key list and apply it to the given
   * value pointer
   *    with the specified offset.
   *
   */
 public:
  SelectedRowsImpl(const std::vector<int64_t>& rows, const int64_t& height)
      : rows_(rows), height_(height) {
    value_.reset(new DenseTensor());
    rwlock_.reset(new RWLock);
  }

  SelectedRowsImpl() {
    height_ = 0;
    value_.reset(new DenseTensor());
    rwlock_.reset(new RWLock);
  }

  const DenseTensor& value() const { return impl_->value(); }

  DenseTensor* mutable_value() { return impl_->mutable_value(); }

  int64_t height() const { return impl_->height(); }

  void set_height(int64_t height) { impl_->set_height(height); }

  const paddle::framework::Vector<int64_t>& rows() const {
    return impl_->rows();
  }

  paddle::framework::Vector<int64_t>* mutable_rows() {
    return impl_->mutable_rows();
  }

  void set_rows(const paddle::framework::Vector<int64_t>& rows) {
    impl_->set_rows(rows);
  }

  /*
   * @brief Get the index of key in rows
   *
   * @return -1 if the key does not exists.
   */
  int64_t Index(int64_t key) const { return impl_->Index(key); }

  /*
   * @brief whether has the specified key in the table.
   *
   * @return true if the key is exists.
   */
  bool HasKey(int64_t key) const { return impl_->HasKey(key); }

  /*
   * @brief Get value by the key list.
   * Note!!! this interface is only used when selected_rows is used as
   * parameters
   * for distribute lookup table.
   *
   * @return a list of pair which contains the non-exists key and the index in
   * the value
   */
  void Get(const DenseTensor& ids,
           DenseTensor* value,
           bool auto_grown = false,
           bool is_test = false) {
    impl_->Get(ids, value, auto_grown, is_test);
  }

  void* AllocateFrom(Allocator* allocator,
                     DataType dtype,
                     size_t requested_size = 0);

  /*
   * @brief Get the index of the key from id_to_index_ map. If the key not
   * exist,
   * add the key into id_to_index_.
   *
   * Note!!! this interface is only used when selected_rows is used as
   * parameters
   * for distribute lookup table.
   *
   * @return index of the key.
   */
  int64_t AutoGrownIndex(int64_t key, bool auto_grown, bool is_test = false) {
    return impl_->AutoGrownIndex(key, auto_grown, is_test);
  }

  /*
   * @brief Get the index of the key from id_to_index_ map.
   */
  inline int64_t GetIndexFromId(int64_t key) const {
    return impl_->GetIndexFromId(key);
  }

  void SyncIndex() { impl_->SyncIndex(); }
  /*
   * @brief Get complete Dims before
   */
  phi::DDim GetCompleteDims() const {
    std::vector<int64_t> dims = vectorize(value_->dims());
    dims[0] = height_;
    return phi::make_ddim(dims);
  }

  /// \brief Returns the number of elements contained in tensor.
  /// \return The number of elements contained in tensor.
  int64_t numel() const override { return impl_->numel(); };

  /// \brief Returns the dims of the tensor.
  /// \return The dims of the tensor.
  const DDim& dims() const noexcept override {
    return impl_->dims();
    // return paddle::framework::make_ddim(dims);
  }

  /// \brief Returns the data type of the tensor.
  /// \return The data type of the tensor.
  DataType dtype() const noexcept { return value_->dtype(); }

  /// \brief Returns the data layout of the tensor.
  /// \return The data layout of the tensor.
  DataLayout layout() const noexcept { return value_->layout(); }

  /// \brief Returns the data place of the tensor.
  /// \return The data place of the tensor.
  const Place& place() const { return value_->place(); }

  /// \brief Test whether the metadata is valid.
  /// \return Whether the metadata is valid.
  bool valid() const noexcept { return value_->valid(); }

  /// \brief Test whether the storage is allocated.
  /// return Whether the storage is allocated.
  bool initialized() const { return value_->initialized(); }

 private:
  // Notice: rows can be duplicate. We can have {0, 4, 7, 0, 5, 7, 9} here.
  // SelectedRowsImpl are simply concated when adding together. Until a
  // SelectedRowsImpl add a Tensor, will the duplicate rows be handled.
  paddle::framework::Vector<int64_t> rows_;
  std::unordered_map<int64_t, int64_t>
      id_to_index_;  // should not be used when rows_ has duplicate member
  std::unique_ptr<DenseTensor> value_{nullptr};
  int64_t height_;  // height indicates the underline tensor's height
  std::unique_ptr<RWLock> rwlock_{nullptr};
};

}  // namespace phi
