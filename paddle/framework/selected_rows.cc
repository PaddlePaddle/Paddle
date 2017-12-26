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

#include "paddle/framework/selected_rows.h"

namespace paddle {
namespace framework {
void SelectedRows::SerializeToStream(std::ostream &os,
                                     const platform::DeviceContext &dev_ctx) {
  PADDLE_ENFORCE_NOT_NULL(
      value_, "serialize SelectedRows failed since Tensor is nullptr.");
  value_->SerializeToStream(os, dev_ctx);
  {
    // serialize rows information
    uint64_t size = rows_.size();
    os.write(reinterpret_cast<const char *>(&size), sizeof(size));
    for (uint64_t i = 0; i < size; ++i) {
      os.write(reinterpret_cast<const char *>(&rows_[i]), sizeof(rows_[i]));
    }
  }
  {
    // serialize height field
    os.write(reinterpret_cast<const char *>(&this->height_), sizeof(int64_t));
  }
}

void SelectedRows::DeserializeFromStream(std::istream &is) {
  value_.reset(new Tensor());
  value_->DeserializeFromStream(is);
  {
    // deserialize rows information
    uint64_t size;
    is.read(reinterpret_cast<char *>(&size), sizeof(size));
    rows_.resize(size);
    for (uint64_t i = 0; i < size; ++i) {
      int64_t tmp;
      is.read(reinterpret_cast<char *>(&tmp), sizeof(int64_t));
      rows_[i] = tmp;
    }
  }
  {
    // deserialize height field
    is.read(reinterpret_cast<char *>(&this->height_), sizeof(int64_t));
  }
}

}  // namespace framework
}  // namespace paddle
