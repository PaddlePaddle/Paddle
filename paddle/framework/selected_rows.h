/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
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
#include "paddle/framework/tensor.h"
#include "paddle/platform/cpu_gpu_vector.h"

namespace paddle {
namespace framework {

class SelectedRows {
 public:
  SelectedRows(const platform::CPUGPUVector& rows, const int64_t& height)
      : rows_(rows), height_(height) {
    value_.reset(new Tensor());
  }

  SelectedRows() { value_.reset(new Tensor()); }

  platform::Place place() const { return value_->place(); }

  const Tensor& value() const { return *value_; }

  Tensor* mutable_value() { return value_.get(); }

  int64_t height() const { return height_; }

  void set_height(int64_t height) { height_ = height; }

  const std::vector<int64_t>& rows() const { return rows_; }

  void set_rows(const std::vector<int64_t>& rows) { rows_ = rows; }

  DDim GetCompleteDims() const {
    std::vector<int64_t> dims = vectorize(value_->dims());
    dims[0] = height_;
    return make_ddim(dims);
  }

 private:
  platform::CPUGPUVector<int64_t> rows_;
  std::unique_ptr<Tensor> value_{nullptr};
  int64_t height_;
};

}  // namespace framework
}  // namespace paddle
