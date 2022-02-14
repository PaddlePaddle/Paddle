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

#include "paddle/pten/common/place.h"
#include "paddle/pten/core/ddim.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/enforce.h"
#include "paddle/pten/core/utils/rw_lock.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/framework/mixed_vector.h"

namespace egr {
class EagerTensor;
}  // namespace egr
namespace pten {
class SelectedRows : public TensorBase,
                     public TypeInfoTraits<TensorBase, SelectedRows> {
  /*
   * @brief We can use the SelectedRows structure to reproduce a sparse table.
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
  SelectedRows(const std::vector<int64_t>& rows, const int64_t& height)
      : rows_(rows), height_(height) {
    value_.reset(new DenseTensor());
    rwlock_.reset(new RWLock);
  }

  SelectedRows() {
    height_ = 0;
    value_.reset(new DenseTensor());
    rwlock_.reset(new RWLock);
  }

  const DenseTensor& value() const { return *value_; }

  DenseTensor* mutable_value() { return value_.get(); }

  int64_t height() const { return height_; }

  void set_height(int64_t height) { height_ = height; }

  const paddle::framework::Vector<int64_t>& rows() const { return rows_; }

  paddle::framework::Vector<int64_t>* mutable_rows() { return &rows_; }

  void set_rows(const paddle::framework::Vector<int64_t>& rows) {
    rows_ = rows;
  }

  /*
   * @brief Get the index of key in rows
   *
   * @return -1 if the key does not exists.
   */
  int64_t Index(int64_t key) const {
    auto it = std::find(rows_.begin(), rows_.end(), key);
    if (it == rows_.end()) {
      PADDLE_THROW(paddle::platform::errors::NotFound(
          "Input id (%lld) is not in current rows table.", key));
    }
    return static_cast<int64_t>(std::distance(rows_.begin(), it));
  }

  /*
   * @brief whether has the specified key in the table.
   *
   * @return true if the key is exists.
   */
  bool HasKey(int64_t key) const;

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
           bool is_test = false);

  void* AllocateFrom(Allocator* allocator,
                     DataType dtype,
                     size_t requested_size = 0) override;

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
  int64_t AutoGrownIndex(int64_t key, bool auto_grown, bool is_test = false);

  /*
   * @brief Get the index of the key from id_to_index_ map.
   */
  inline int64_t GetIndexFromId(int64_t key) const {
    auto iter = id_to_index_.find(key);
    if (iter == id_to_index_.end()) {
      return -1;
    } else {
      return iter->second;
    }
  }

  void SyncIndex();
  /*
   * @brief Get complete Dims before
   */
  pten::framework::DDim GetCompleteDims() const {
    std::vector<int64_t> dims = vectorize(value_->dims());
    dims[0] = height_;
    return pten::framework::make_ddim(dims);
  }

  /// \brief Returns the name of the class for type traits.
  /// \return The name of the class.
  static const char* name() { return "SelectedRows"; }

  /// \brief Returns the number of elements contained in tensor.
  /// \return The number of elements contained in tensor.
  int64_t numel() const override { return value_->numel(); };

  /// \brief Returns the dims of the tensor.
  /// \return The dims of the tensor.
  const DDim& dims() const noexcept override {
    return value_->dims();
    // return paddle::framework::make_ddim(dims);
  }

  /// \brief Returns the data type of the tensor.
  /// \return The data type of the tensor.
  DataType dtype() const noexcept override { return value_->dtype(); }

  /// \brief Returns the data layout of the tensor.
  /// \return The data layout of the tensor.
  DataLayout layout() const noexcept override { return value_->layout(); }

  /// \brief Returns the data place of the tensor.
  /// \return The data place of the tensor.
  const Place& place() const override { return value_->place(); };

  /// \brief Test whether the metadata is valid.
  /// \return Whether the metadata is valid.
  bool valid() const noexcept override { return value_->valid(); }

  /// \brief Test whether the storage is allocated.
  /// return Whether the storage is allocated.
  bool initialized() const override { return value_->initialized(); }

 private:
  // Notice: rows can be duplicate. We can have {0, 4, 7, 0, 5, 7, 9} here.
  // SelectedRows are simply concated when adding together. Until a
  // SelectedRows add a Tensor, will the duplicate rows be handled.
  paddle::framework::Vector<int64_t> rows_;
  std::unordered_map<int64_t, int64_t>
      id_to_index_;  // should not be used when rows_ has duplicate member
  std::unique_ptr<DenseTensor> value_{nullptr};
  int64_t height_;  // height indicates the underline tensor's height
  std::unique_ptr<RWLock> rwlock_{nullptr};
  // TODO(jiabin): Remove this when we don't need EagerTensor support
  // SelectedRows which is expected in next version.
  /** Why we need this weird friend class?
   * In eager mode, since some of ops doesn't support C++ API for now we need to
   *use 'imperative::TraceOp' to run it.
   * So, we need to support get a SelectedRows from egr::EagerTensor's
   *framework::Variable obj and used it to reconstruct
   * a new paddle::experimental::Tensor to support framework usage. However, we
   *got 2 problems here.
   * First, we got 2 unique_ptr in SelectedRows so that we can't support
   *std::make_shared in EagerTensor's SetImplWithSelectedRows method,
   * since we have to construct a shared_ptr for paddle::experimental::Tensor's
   *impl.
   * Second, when we are trying to support move constructor for SelectedRows we
   *found that we can't get its rvalue from
   * framework::Variable because it holds an obj of target type.
   *
   *
   * The only three way to solve this problem is:
   * 1. Just like what we have done, using friend class and just copy/move each
   *member. In this way, we can avoid additional API
   * and symbols.
   * 2. Make pten::SelectedRows's member from unique_ptr to shared_ptr. However,
   *this may cause some cost of performance.
   * 3. Add some api to return or move member of framework::SelectedRows.
   *However, it's not as safe as first solution.
   * 4. Support all framework::SelectedRows related ops and make sure
   *EagerTensor never holds framework::SelectedRows.
   *
   * If anyone got better ideas, welcome to contact JiabinYang, we are open for
   *your help.
  **/
  friend class egr::EagerTensor;
};

}  // namespace pten
