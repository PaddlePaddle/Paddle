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

#include "paddle/pten/core/base_tensor.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/framework/rw_lock.h"

namespace pt {

using Vector = paddle::framework::Vector;

/**
 * SelectedRows: compatible with SelectedRows in fluid and related operators.
 */
class SelectedRows final : public BaseTensor {
 public:
  SelectedRows() = delete;

  SelectedRows(const SelectedRows&) = delete;
  SelectedRows& operator=(const SelectedRows&) = delete;
  SelectedRows(SelectedRows&&) = delete;
  SelectedRows& operator=(SelectedRows&&) = delete;

  SelectedRows(const std::vector<int64_t>& rows,
               int64_t height,
               TensorMeta&& meta)
      : rows_(rows), height_(height), BaseTensor(meta) {}

  const Vector<int64_t>& rows() const { return rows_; }

  Vector<int64_t>* mutable_rows() { return &rows_; }

  void set_rows(const Vector<int64_t>& rows)()

      int64_t height() const {
    return height_;
  }

  void set_height(int64_t height) { height_ = height; }

 private:
  Vector<int64_t> rows_;
  int64_t height_;

  std::unordered_map<int64_t, int64_t> id_to_index_;
  std::unique_ptr<RWLock> rwlock_{nullptr};
};

}  // namespace pt
