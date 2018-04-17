/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include <vector>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace framework {

class SelectedRows {
 public:
  SelectedRows(const std::vector<int64_t>& rows, const int64_t& height)
      : rows_(rows), height_(height) {
    value_.reset(new Tensor());
  }

  SelectedRows() {
    height_ = 0;
    value_.reset(new Tensor());
  }

  platform::Place place() const { return value_->place(); }

  const Tensor& value() const { return *value_; }

  Tensor* mutable_value() { return value_.get(); }

  int64_t height() const { return height_; }

  void set_height(int64_t height) { height_ = height; }

  const Vector<int64_t>& rows() const { return rows_; }

  Vector<int64_t>* mutable_rows() { return &rows_; }

  void set_rows(const Vector<int64_t>& rows) { rows_ = rows; }

  /*
   * @brief wheter has the specified key in the table.
   *
   * @return true if the key is exists.
   */
  bool HasKey(int64_t key) const;

  /*
   * @brief Get a value by the specified key, if the
   * key does not exists, this function would throw an exception.
   *
   * @return true if the Get operation successed.
   */

  bool Get(int64_t key, framework::Tensor* tensor, int64_t row = 0) const;

  /*
   * @brief Set a key-value pair into the table.
   *  This function will double the value memory if it's not engouth.
   *
   * @note:
   *    1. The first dim of the value should be 1
   *    2. The value should be initialized and the data type
   *       should be the same with the table.
   *
   * @return true if the key is a new one, otherwise false
   *
   */
  bool Set(int64_t key, const Tensor& value);

  /*
   * @brief Get the index of key in rows
   *
   * @return -1 if the key does not exists.
   */
  int64_t Index(int64_t key) const {
    auto it = std::find(rows_.begin(), rows_.end(), key);
    if (it == rows_.end()) {
      return static_cast<int64_t>(-1);
    }
    return static_cast<int64_t>(std::distance(rows_.begin(), it));
  }

  DDim GetCompleteDims() const {
    std::vector<int64_t> dims = vectorize(value_->dims());
    dims[0] = height_;
    return make_ddim(dims);
  }

 private:
  // Notice: rows can be duplicate. We can have {0, 4, 7, 0, 5, 7, 9} here.
  // SelectedRows are simply concated when adding together. Until a
  // SelectedRows add a Tensor, will the duplicate rows be handled.
  Vector<int64_t> rows_;
  std::unique_ptr<Tensor> value_{nullptr};
  int64_t height_;
};

/*
 * Serialize/Desiralize SelectedRows to std::ostream
 * You can pass ofstream or ostringstream to serilize to file
 * or to a in memory string. GPU tensor will be copied to CPU.
 */
void SerializeToStream(std::ostream& os, const SelectedRows& selected_rows,
                       const platform::DeviceContext& dev_ctx);
void DeserializeFromStream(std::istream& is, SelectedRows* selected_rows,
                           const platform::DeviceContext& dev_ctx);

}  // namespace framework
}  // namespace paddle
