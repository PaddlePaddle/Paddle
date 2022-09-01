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

#include "paddle/phi/core/selected_rows.h"

#include "paddle/phi/core/utils/data_type.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/memory/memcpy.h"

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

  const DenseTensor& value() const { return *value_; }

  DenseTensor* mutable_value() { return value_.get(); }

  int64_t height() const { return height_; }

  void set_height(int64_t height) { height_ = height; }

  const std::vector<int64_t>& rows() const { return rows_; }

  std::vector<int64_t>* mutable_rows() { return &rows_; }

  void set_rows(const std::vector<int64_t>& rows) { rows_ = rows; }

  /*
   * @brief Get the index of key in rows
   *
   * @return -1 if the key does not exists.
   */
  int64_t Index(int64_t key) const {
    auto it = std::find(rows_.begin(), rows_.end(), key);
    if (it == rows_.end()) {
      PADDLE_THROW(phi::errors::NotFound(
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
  phi::DDim GetCompleteDims() const {
    std::vector<int64_t> dims = vectorize(value_->dims());
    dims[0] = height_;
    return phi::make_ddim(dims);
  }

  /// \brief Returns the number of elements contained in tensor.
  /// \return The number of elements contained in tensor.
  int64_t numel() const { return value_->numel(); }

  /// \brief Returns the dims of the tensor.
  /// \return The dims of the tensor.
  const DDim& dims() const noexcept { return value_->dims(); }

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
  std::vector<int64_t> rows_;
  std::unordered_map<int64_t, int64_t>
      id_to_index_;  // should not be used when rows_ has duplicate member
  std::unique_ptr<DenseTensor> value_{nullptr};
  int64_t height_;  // height indicates the underline tensor's height
  std::unique_ptr<RWLock> rwlock_{nullptr};
};

struct ReAllocateVisitor {
  ReAllocateVisitor(const phi::DDim& dims, phi::DenseTensor* tensor)
      : dims_(dims), tensor_(tensor) {}

  template <typename T>
  void operator()() const {
    phi::DenseTensor cpu_tensor;
    phi::CPUPlace cpu;
    T* ptr = cpu_tensor.mutable_data<T>(dims_, cpu);
    const T* old_ptr =
        tensor_->memory_size() == 0 ? nullptr : tensor_->data<T>();
    if (old_ptr != nullptr) {
      std::copy(old_ptr, old_ptr + tensor_->numel(), ptr);
    }
    tensor_->ShareDataWith(cpu_tensor);
  }

  phi::DDim dims_;
  phi::DenseTensor* tensor_;
};

struct TensorCopyVisitor {
  TensorCopyVisitor(phi::DenseTensor* dst,
                    int64_t dst_offset,
                    const phi::DenseTensor src,
                    int64_t src_offset,
                    int64_t size)
      : dst_(dst),
        dst_offset_(dst_offset),
        src_(src),
        src_offset_(src_offset),
        size_(size) {}

  template <typename T>
  void apply() const {
    // TODO(Yancey1989): support other place
    phi::CPUPlace cpu;
    std::memcpy(dst_->mutable_data<T>(cpu) + dst_offset_,
                src_.data<T>() + src_offset_,
                size_ * sizeof(T));
  }

  phi::DenseTensor* dst_;
  int64_t dst_offset_;
  phi::DenseTensor src_;
  int64_t src_offset_;
  int64_t size_;
};

struct TensorFillVisitor {
  TensorFillVisitor(phi::DenseTensor* dst,
                    int64_t dst_offset,
                    int64_t size,
                    float value)
      : dst_(dst), dst_offset_(dst_offset), size_(size) {}

  template <typename T>
  void apply() const {
    // TODO(qiao): support other place
    phi::CPUPlace cpu;
    auto* tensor_data = dst_->mutable_data<T>(cpu);
    auto* start = tensor_data + dst_offset_;
    auto* end = start + size_;
    std::fill(start, end, static_cast<T>(0.0));
  }

  phi::DenseTensor* dst_;
  int64_t dst_offset_;
  int64_t size_;
};

void* SelectedRowsImpl::AllocateFrom(Allocator* allocator,
                                     DataType dtype,
                                     size_t requested_size) {
  return value_->AllocateFrom(allocator, dtype, requested_size);
}

bool SelectedRowsImpl::HasKey(int64_t key) const {
  return std::find(rows_.begin(), rows_.end(), key) == rows_.end() ? false
                                                                   : true;
}

int64_t SelectedRowsImpl::AutoGrownIndex(int64_t key,
                                         bool auto_grown,
                                         bool is_test) {
  if (is_test) {
    auto iter = id_to_index_.find(key);
    if (iter == id_to_index_.end()) {
      return -1;
    } else {
      return iter->second;
    }
  }

  rwlock_->RDLock();
  auto iter = id_to_index_.find(key);
  if (iter == id_to_index_.end()) {
    rwlock_->UNLock();
    PADDLE_ENFORCE_EQ(
        auto_grown,
        true,
        phi::errors::NotFound("Input key(%lld) is not found.", key));
    rwlock_->WRLock();
    auto map_size = id_to_index_.size();
    auto vector_size = rows_.size();
    if (map_size != vector_size) {
      rwlock_->UNLock();
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Row map size(%zu) should be equal to rows size(%zu).",
          map_size,
          vector_size));
    }
    auto write_iter = id_to_index_.find(key);
    if (write_iter == id_to_index_.end()) {
      int row_num = rows_.size();
      if (row_num == value_->dims()[0]) {
        rwlock_->UNLock();
        PADDLE_THROW(phi::errors::InvalidArgument(
            "Selected rows is full, then length exceed the length of first "
            "dimension (%d).",
            row_num));
      }
      // key logic to put a key into id_to_index_
      rows_.push_back(key);
      auto index = static_cast<int64_t>(rows_.size() - 1);
      id_to_index_[key] = index;
      rwlock_->UNLock();
      return index;
    } else {
      auto index = write_iter->second;
      rwlock_->UNLock();
      return index;
    }
  } else {
    auto index = iter->second;
    rwlock_->UNLock();
    return index;
  }
}

void SelectedRowsImpl::SyncIndex() {
  rwlock_->WRLock();
  id_to_index_.clear();
  for (size_t i = 0; i < rows_.size(); ++i) {
    id_to_index_[rows_[i]] = i;
  }
  rwlock_->UNLock();
}

void SelectedRowsImpl::Get(const phi::DenseTensor& ids,
                           phi::DenseTensor* value,
                           bool auto_grown,
                           bool is_test) {
  PADDLE_ENFORCE_EQ(value->IsInitialized(),
                    true,
                    paddle::platform::errors::InvalidArgument(
                        "The value tensor is not initialized."));
  if (ids.numel() == 0) {
    VLOG(3) << "keys is empty, please check data!";
  } else {
    int64_t value_width = value_->numel() / value_->dims()[0];
    PADDLE_ENFORCE_EQ(
        value_width,
        value->numel() / value->dims()[0],
        phi::errors::InvalidArgument(
            "Output tensor should have the same shape with table "
            "except the first dimmension, excepted value width not counting "
            "the first dimension is %d, actual value width is %d.",
            value_width,
            value->numel() / value->dims()[0]));
    for (int i = 0; i < ids.numel(); ++i) {
      auto id = ids.data<int64_t>()[i];
      int64_t index = AutoGrownIndex(id, auto_grown, is_test);
      if (index < 0) {
        VLOG(5) << "id " << id << " not in the table, return 0";
        phi::VisitDataType(
            value_->dtype(),
            TensorFillVisitor(value, i * value_width, value_width, 0.0));
      } else {
        phi::VisitDataType(value_->dtype(),
                           TensorCopyVisitor(value,
                                             i * value_width,
                                             *value_.get(),
                                             index * value_width,
                                             value_width));
      }
    }
  }
}

SelectedRows::SelectedRows(const std::vector<int64_t>& rows,
                           const int64_t& height)
    : impl_(std::make_shared<phi::SelectedRowsImpl>(rows, height)) {}

SelectedRows::SelectedRows()
    : impl_(std::make_shared<phi::SelectedRowsImpl>()) {}

const DenseTensor& SelectedRows::value() const { return impl_->value(); }

DenseTensor* SelectedRows::mutable_value() { return impl_->mutable_value(); }

int64_t SelectedRows::height() const { return impl_->height(); }

void SelectedRows::set_height(int64_t height) { impl_->set_height(height); }

const std::vector<int64_t>& SelectedRows::rows() const { return impl_->rows(); }

std::vector<int64_t>* SelectedRows::mutable_rows() {
  return impl_->mutable_rows();
}

void SelectedRows::set_rows(const std::vector<int64_t>& rows) {
  impl_->set_rows(rows);
}

int64_t SelectedRows::Index(int64_t key) const { return impl_->Index(key); }

bool SelectedRows::HasKey(int64_t key) const { return impl_->HasKey(key); }

void SelectedRows::Get(const DenseTensor& ids,
                       DenseTensor* value,
                       bool auto_grown,
                       bool is_test) {
  impl_->Get(ids, value, auto_grown, is_test);
}

void* SelectedRows::AllocateFrom(Allocator* allocator,
                                 DataType dtype,
                                 size_t requested_size) {
  return impl_->AllocateFrom(allocator, dtype, requested_size);
}

int64_t SelectedRows::AutoGrownIndex(int64_t key,
                                     bool auto_grown,
                                     bool is_test) {
  return impl_->AutoGrownIndex(key, auto_grown, is_test);
}

int64_t SelectedRows::GetIndexFromId(int64_t key) const {
  return impl_->GetIndexFromId(key);
}

void SelectedRows::SyncIndex() { impl_->SyncIndex(); }

DDim SelectedRows::GetCompleteDims() const { return impl_->GetCompleteDims(); }

int64_t SelectedRows::numel() const { return impl_->numel(); }

const DDim& SelectedRows::dims() const noexcept { return impl_->dims(); }

DataType SelectedRows::dtype() const noexcept { return impl_->dtype(); }

DataLayout SelectedRows::layout() const noexcept { return impl_->layout(); }

const Place& SelectedRows::place() const { return impl_->place(); }

bool SelectedRows::valid() const noexcept { return impl_->valid(); }

bool SelectedRows::initialized() const { return impl_->initialized(); }

}  // namespace phi
