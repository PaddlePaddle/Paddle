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

#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/saver.pb.h"

#include "paddle/memory/memcpy.h"
#include "paddle/memory/memory.h"

#include <stdint.h>
#include <string.h>
#include <algorithm>
#include <iterator>

#include <glog/logging.h>

namespace paddle {
namespace framework {

LoD SliceLevels(const LoD& in, size_t level_begin, size_t level_end) {
  LoD new_lod;
  new_lod.reserve(level_end - level_begin);
  for (size_t i = level_begin; i < level_end; i++) {
    new_lod.emplace_back(in.at(i));
  }
  // transform the lowest level to absolute offset.
  LoD abs_offset_lod = ToAbsOffset(in);
  new_lod.back() = abs_offset_lod[level_end - 1];
  return new_lod;
}

LoD SliceInLevel(const LoD& in, size_t level, size_t elem_begin,
                 size_t elem_end) {
  PADDLE_ENFORCE_LT(level, in.size());
  PADDLE_ENFORCE_LT(elem_end, in[level].size());

  LoD res;
  res.resize(in.size() - level);
  // copy the first level
  res[0].assign(in[level].begin() + elem_begin,
                in[level].begin() + elem_end + 1);
  for (size_t lvl = 1; lvl < res.size(); lvl++) {
    const auto& in_level = in[level + lvl];
    const auto& above_level = res[lvl - 1];
    auto& out_level = res[lvl];
    out_level.assign(in_level.begin() + above_level.front(),
                     in_level.begin() + above_level.back() + 1);
  }
  for (size_t lvl = 0; lvl < res.size(); lvl++) {
    // to make the first offset equals 0, all the elements minus the first
    // element
    size_t front = res[lvl].front();
    for (auto& ele : res[lvl]) {
      ele -= front;
    }
  }
  return res;
}

LoD ToAbsOffset(const LoD& in) {
  // the lowest level stores relative offsets
  if (in.empty() || in.size() == 1) return in;
  LoD result = in;
  for (int level = result.size() - 2; level >= 0; level--) {
    for (auto& ele : result[level]) {
      ele = result[level + 1][ele];
    }
  }
  return result;
}

bool operator==(const LoD& a, const LoD& b) {
  if (a.size() != b.size()) {
    return false;
  }

  for (size_t i = 0; i < a.size(); i++) {
    const auto& a_level = a[i];
    const auto& b_level = b[i];
    if (a_level.size() != b_level.size()) {
      return false;
    }
    for (size_t j = 0; j < a_level.size(); j++) {
      if (a_level[j] != b_level[j]) {
        return false;
      }
    }
  }
  return true;
}

size_t LoDTensor::NumElements(size_t level, size_t idx) const {
  PADDLE_ENFORCE_LT(level, NumLevels());
  PADDLE_ENFORCE_LT(idx, NumElements(level));
  return lod_[level][idx + 1] - lod_[level][idx];
}

void LoDTensor::ShrinkLevels(size_t level_begin, size_t level_end) {
  auto new_lod = framework::SliceLevels(lod_, level_begin, level_end);
  lod_ = new_lod;
}

void LoDTensor::ShrinkInLevel(size_t level, size_t elem_begin,
                              size_t elem_end) {
  PADDLE_ENFORCE_LT(level, NumLevels());
  PADDLE_ENFORCE_LT(elem_begin, NumElements(level));
  PADDLE_ENFORCE_LT(elem_end, NumElements(level) + 1);

  auto new_lod = framework::SliceInLevel(lod_, level, elem_begin, elem_end);
  lod_ = new_lod;
}

std::string LoDTensor::SerializeToString() const {
  LoDTensorProto desc;

  // set data_type
  if (this->type() == typeid(int8_t)) desc.set_data_type(DataType::BOOL);
  if (this->type() == typeid(int16_t)) desc.set_data_type(DataType::INT16);
  if (this->type() == typeid(int32_t)) desc.set_data_type(DataType::INT32);
  if (this->type() == typeid(int64_t)) desc.set_data_type(DataType::INT64);
  // FIXME(dzh): there is no fp16 in standard c++

  if (this->type() == typeid(float))  // NOLINT
    desc.set_data_type(DataType::FP32);
  if (this->type() == typeid(double))  // NOLINT
    desc.set_data_type(DataType::FP64);

  for (int i = 0; i < dims().size(); ++i) {
    desc.add_dims(dims()[i]);
  }
  // for (auto& dim : dims) {
  //   desc.add_dims(dims()dwim);
  // }

  // set lod information
  desc.set_lod_level(this->NumLevels());
  for (size_t i = 0; i < this->NumLevels(); ++i) {
    LoDInfo* lod = desc.add_levels();
    for (size_t j = 0; j < lod_[i].size(); ++j) {
      lod->add_level(lod_[i][j]);
    }
  }

  desc.set_version(0);

  std::string desc_bytes = desc.SerializeAsString();

  // FIXME(dzh) : implement fix chunk size buffer.
  size_t DESC_SIZE = desc_bytes.size();
  size_t DATA_SIZE = holder_->size() - offset_;

  const size_t BUFFER_SIZE = DESC_SIZE + DATA_SIZE + 2 * sizeof(size_t);
  char* buffer =
      static_cast<char*>(memory::Alloc(platform::CPUPlace(), BUFFER_SIZE));

  // format: desc_size data_size, desc_bytes, data_bytes.
  platform::CPUPlace src_place;
  platform::CPUPlace dst_place;

  memory::Copy(dst_place, buffer, src_place, &DESC_SIZE, sizeof(size_t));
  memory::Copy(dst_place, buffer + sizeof(size_t), src_place, &DATA_SIZE,
               sizeof(size_t));
  memory::Copy(dst_place, buffer + sizeof(size_t) * 2, src_place,
               desc_bytes.c_str(), desc_bytes.size());

  PADDLE_ENFORCE(this->numel() != 0, " Serialize a empty Tensor!");

  platform::Place place = holder_->place();
  int element_width = holder_->size() / this->numel();

  if (platform::is_cpu_place(place)) {
    memory::Copy(dst_place, buffer + sizeof(size_t) * 2 + desc_bytes.size(),
                 boost::get<platform::CPUPlace>(place),
                 static_cast<char*>(holder_->ptr()) + offset_ / element_width,
                 DATA_SIZE);
  }
#ifdef PADDLE_WITH_GPU
  if (platform::is_gpu_place(place)) {
    memory::Copy(dst_place, buffer + sizeof(size_t) * 2 + desc_bytes.size(),
                 boost::get<platform::GPUPlace>(place),
                 static_cast<char*>(holder_->ptr()) + offset_ / element_width,
                 DATA_SIZE);
  }
#endif

  std::string ret(buffer, BUFFER_SIZE);
  memory::Free(platform::CPUPlace(), buffer);
  return ret;
}

void LoDTensor::DeserializeFromString(const std::string& s,
                                      const platform::Place& dst_place) {
  size_t DESC_SIZE, DATA_SIZE;
  platform::CPUPlace src_place;

  memory::Copy(src_place, &DESC_SIZE, src_place, s.c_str(), sizeof(size_t));
  memory::Copy(src_place, &DATA_SIZE, src_place, s.c_str() + sizeof(size_t),
               sizeof(size_t));

  // parse LoDTensorDesc
  LoDTensorProto desc;
  desc.ParseFromArray(s.c_str() + sizeof(size_t) * 2, DESC_SIZE);

  std::vector<int64_t> dims;
  std::copy(desc.dims().begin(), desc.dims().end(), std::back_inserter(dims));
  this->Resize(make_ddim(dims));

  // parse data type
  void* ptr = nullptr;
  if (desc.data_type() == DataType::BOOL)
    ptr = this->mutable_data<bool>(dst_place);
  if (desc.data_type() == DataType::INT16)
    ptr = this->mutable_data<int16_t>(dst_place);
  if (desc.data_type() == DataType::INT32)
    ptr = this->mutable_data<int32_t>(dst_place);
  if (desc.data_type() == DataType::INT64)
    ptr = this->mutable_data<int64_t>(dst_place);
  // FIXME(dzh): there is no fp16 in standard c++

  if (desc.data_type() == DataType::FP32)
    ptr = this->mutable_data<float>(dst_place);
  if (desc.data_type() == DataType::FP64)
    ptr = this->mutable_data<double>(dst_place);

  LoD lod;
  std::vector<size_t> levels;
  for (int i = 0; i < desc.levels().size(); ++i) {
    auto current_level = desc.levels()[i].level();
    std::copy(current_level.begin(), current_level.end(),
              std::back_inserter(levels));
    lod.emplace_back(levels);
    levels.clear();
  }

  this->set_lod(lod);

  if (platform::is_cpu_place(dst_place)) {
    memory::Copy(boost::get<platform::CPUPlace>(dst_place), ptr, src_place,
                 s.c_str() + sizeof(size_t) * 2 + DESC_SIZE, DATA_SIZE);
  }
#ifdef PADDLE_WITH_GPU
  if (platform::is_gpu_place(dst_place)) {
    memory::Copy(boost::get<platform::GPUPlace>(dst_place), ptr, src_place,
                 s.c_str() + sizeof(size_t) * 2 + DESC_SIZE, DATA_SIZE);
  }
#endif
}

}  // namespace framework
}  // namespace paddle
