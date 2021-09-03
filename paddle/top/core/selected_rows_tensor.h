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

#include <algorithm>
#include <memory>
#include <mutex>  // NOLINT
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/top/core/dense_tensor.h"
#include "paddle/top/core/tensor_interface.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/framework/rw_lock.h"

namespace pt {

template <typename T>
using Vector = paddle::framework::Vector<T>;
using RWLock = paddle::framework::RWLock;

/**
 * SelectedRowsTensor: compatible with SelectedRows in fluid and related
 * operators.
 *
 * SelectedRowsTensor is not a typical design of sparse Tensor, and may
 * no longer be recommended for use in the future, and there may be new
 * SparseTensor later.
 */

// TODO(chenweihang): add other methods later

class SelectedRowsTensor : public TensorInterface {
 public:
  SelectedRowsTensor() = delete;

  // SelectedRowsTensor(const SelectedRowsTensor&) = delete;
  // SelectedRowsTensor& operator=(const SelectedRowsTensor&) = delete;
  SelectedRowsTensor(SelectedRowsTensor&&) = delete;
  SelectedRowsTensor& operator=(SelectedRowsTensor&&) = delete;

  SelectedRowsTensor(const TensorMeta& meta,
                     const TensorStatus& status,
                     const std::vector<int64_t>& rows,
                     int64_t height) {
    value_.reset(new DenseTensor(meta, status));
    rows_ = rows;
    height_ = height;
  }

  ~SelectedRowsTensor() override {}

  int64_t numel() const override { return value_->numel(); }

  DDim dims() const override {
    std::vector<int64_t> dims = vectorize(value_->dims());
    dims[0] = height_;
    return paddle::framework::make_ddim(dims);
  }

  DataType type() const override { return value_->type(); }

  DataLayout layout() const override { return value_->layout(); }

  Place place() const override { return value_->place(); }

  Backend backend() const override { return value_->backend(); }

  bool initialized() const override { return value_->initialized(); }

  const DenseTensor& value() const { return *value_; }

  DenseTensor* mutable_value() { return value_.get(); }

  const Vector<int64_t>& rows() const { return rows_; }

  Vector<int64_t>* mutable_rows() { return &rows_; }

  void set_rows(const Vector<int64_t>& rows) { rows_ = rows; }

  int64_t height() const { return height_; }

  void set_height(int64_t height) { height_ = height; }

 private:
  std::unique_ptr<DenseTensor> value_{nullptr};

  Vector<int64_t> rows_;
  int64_t height_;

  std::unordered_map<int64_t, int64_t> id_to_index_;
  std::unique_ptr<RWLock> rwlock_{nullptr};
};

}  // namespace pt
