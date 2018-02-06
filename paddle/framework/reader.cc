//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/framework/reader.h"

namespace paddle {
namespace framework {

DDim FileReader::shape(size_t idx) const {
  PADDLE_ENFORCE_LT(
      idx, shapes_.size(),
      "Cannot get the %d'th shape, 'shapes_' only has %d elements.", idx,
      shapes_.size());
  return shapes_[idx];
}

void ShuffleReader::ReadNext(std::vector<LoDtensor>* out) {
  if (iteration_pos_ >= buffer_.size()) {
    // Reload buffer with new data
    buffer_.clear();
    buffer_.reverse(buffer_size_);
    for (int i = 0; i < buffer_size_; ++i) {
      if (reader_->HasNext()) {
        buffer.push_back(std::vector<LoDTensor>());
        reader_->ReadNext(&buffer.back());
      } else {
        break;
      }
    }
    std::random_shuffle(buffer_.begin(), buffer_.end());
    iteration_pos_ = 0;
  }
  out->clear();
  if (!buffer_.empty()) {
    std::swap(*out, buffer_[iteration_pos_++]);
  }
  // if buffer_ is empty, the 'out' will return as an empty vector.
}

void BatchReader::ReadNext(std::vector<LoDtensor>* out) {
  buffer_.clear();
  buffer_.reserve(batch_size_);
  for (int i = 0; i < batch_size_; ++i) {
    if (reader_->HasNext()) {
      buffer_.push_back(std::vector<LoDtensor>());
      reader_->ReadNext(&buffer_.back());
    } else {
      break;
    }
  }
  // Concat instances
  out.clear();
  if (buffer_.empty()) {
    // if buffer_ is empty, the 'out' will return as an empty vector.
    return;
  }
  int out_num = buffer_[0].size();
  out->reserve(out_num);
  for (int j = 0; j < out_num; ++j) {
    // Merge shape and check date type
    std::type_index batch_type = buffer_[0][j].type();
    DDim batch_shape = buffer_[0][j].dims();
    for (size_t i = 1; i < buffer_.size(); ++i) {
      std::type_index ins_type = buffer_[i][j].type();
      DDim ins_shape = buffer_[i][j].dims();
      PADDLE_ENFORCE_EQ(batch_type, ins_type);
      PADDLE_ENFORCE_EQ(slice_ddim(batch_shape, 1, batch_shape.size()),
                        slice_ddim(ins_shape, 1, ins_shape.size()));
      PADDLE_ENFORCE_GT(ins_shape[0], 0);
      batch_shape[0] += ins_shape[0];
    }

    LoDTensor out_tensor;
    out_tensor.Resize(batch_shape);
    out_tensor.mutable_data(platform::CPUPlace(), batch_type);
    int64_t dst_offset = 0;

    // Merge lod and data
    LoD batch_lod;
    std::vector<size_t> top_level_lod({0});
    for (size_t i = 0; i < buffer_.size(); ++i) {
      DDim ins_shape = buffer_[i][j].dims();
      LoD ins_lod = buffer_[i][j].lod();
      if (i == 0) {
        batch_lod = ins_lod;
      } else {
        PADDLE_ENFORCE_EQ(batch_lod.size(), ins_lod.size());
        for (size_t level_idx = 0; level_idx < batch_lod.size(); ++level_idx) {
          auto& lod_level = batch_lod[level_idx];
          for (size_t k = 1; k < ins_lod[level_idx].size(); ++k) {
            lod_level.push_back(ins_lod[level_idx][k] + lod_level.back());
          }
        }
      }
      top_level_lod.push_back(
          top_level_lod.back() +
          (ins_lod.empty() ? ins_shape[0] : (ins_lod[0].size() - 1)));

      Tensor dst = out_tensor.Slice(dst_offset, dst_offset + ins_shape[0]);
      Copy(buffer_[i][j], platform::CPUPlace(), &dst);
      dst_offset += ins_shape[0];
    }
    batch_lod.insert(batch_lod.begin(), top_level_lod);
    out_tensor.set_lod(batch_lod);
    out->push_back(out_tensor);
  }
}
}  // namespace framework
}  // namespace paddle
