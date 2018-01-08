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
#include "paddle/framework/data_type.h"
#include "paddle/framework/framework.pb.h"

#include "paddle/memory/memcpy.h"
#include "paddle/memory/memory.h"

#include <stdint.h>
#include <string.h>
#include <algorithm>
#include <iterator>

#include <glog/logging.h>

namespace paddle {
namespace framework {

std::ostream &operator<<(std::ostream &os, const LoD &lod) {
  os << "{";
  for (auto &v : lod) {
    os << "{";
    for (auto &i : v) {
      os << i << ",";
    }
    os << "}";
  }
  os << "}";

  return os;
}

std::ostream &operator<<(std::ostream &os, const LoDTensor &t) {
  PADDLE_ENFORCE(platform::is_cpu_place(t.place()));
  PADDLE_ENFORCE(t.type().hash_code() == typeid(float).hash_code());

  os << "dim: " << t.dims() << "\n";
  os << "lod: " << t.lod() << "\n";

  // only print first ten elements
  int64_t size = t.numel() < 10 ? t.numel() : 10;
  for (int64_t i = 0; i < size; ++i) {
    os << t.data<float>()[i] << " ";
  }

  return os;
}

LoD SliceInLevel(const LoD &in, size_t level, size_t elem_begin,
                 size_t elem_end) {
  PADDLE_ENFORCE_LT(level, in.size());
  PADDLE_ENFORCE_LT(elem_end, in[level].size());

  LoD res;
  res.resize(in.size() - level);
  // copy the first level
  res[0].assign(in[level].begin() + elem_begin,
                in[level].begin() + elem_end + 1);
  for (size_t lvl = 1; lvl < res.size(); lvl++) {
    const auto &in_level = in[level + lvl];
    const auto &above_level = res[lvl - 1];
    auto &out_level = res[lvl];
    out_level.assign(in_level.begin() + above_level.front(),
                     in_level.begin() + above_level.back() + 1);
  }
  for (size_t lvl = 0; lvl < res.size(); lvl++) {
    // to make the first offset equals 0, all the elements minus the first
    // element
    size_t front = res[lvl].front();
    for (auto &ele : res[lvl]) {
      ele -= front;
    }
  }
  return res;
}

LoD ToAbsOffset(const LoD &in) {
  // the lowest level stores relative offsets
  if (in.empty() || in.size() == 1) return in;
  LoD result = in;
  for (int level = result.size() - 2; level >= 0; level--) {
    for (auto &ele : result[level]) {
      ele = result[level + 1][ele];
    }
  }
  return result;
}

bool operator==(const LoD &a, const LoD &b) {
  if (a.size() != b.size()) {
    return false;
  }

  for (size_t i = 0; i < a.size(); i++) {
    const auto &a_level = a[i];
    const auto &b_level = b[i];
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

using LoDAndOffset = std::pair<LoD, std::pair<size_t, size_t>>;
LoDAndOffset GetSubLoDAndAbsoluteOffset(const LoD &lod, size_t start_idx,
                                        size_t end_idx, size_t start_level) {
  LoD sub_lod;

  for (size_t level_idx = start_level; level_idx < lod.size(); ++level_idx) {
    PADDLE_ENFORCE_LE(start_idx, end_idx);
    PADDLE_ENFORCE_LT(end_idx, lod[level_idx].size());
    std::vector<size_t> level_lens;
    for (size_t i = start_idx; i < end_idx; ++i) {
      level_lens.push_back(lod[level_idx][i + 1] - lod[level_idx][i]);
    }
    sub_lod.emplace_back(level_lens);
    start_idx = lod[level_idx][start_idx];
    end_idx = lod[level_idx][end_idx];
  }

  return LoDAndOffset{sub_lod, {start_idx, end_idx}};
}

void AppendLoD(LoD *lod, const LoD &lod_length) {
  PADDLE_ENFORCE(
      lod->empty() || lod->size() == lod_length.size(),
      "The lod_length should has the same size with the appended lod.");
  if (lod->empty()) {
    for (size_t i = 0; i < lod_length.size(); ++i) {
      lod->emplace_back(1, 0);  // size = 1, value = 0;
    }
    *lod = LoD(lod_length.size(), std::vector<size_t>({0}));
  }
  for (size_t i = 0; i < lod->size(); ++i) {
    auto &level = (*lod)[i];
    for (size_t len : lod_length[i]) {
      level.push_back(level.back() + len);
    }
  }
}

void SerializeToStream(std::ostream &os, const LoDTensor &tensor,
                       const platform::DeviceContext &dev_ctx) {
  {  // the 1st field, uint32_t version for LoDTensor
    constexpr uint32_t version = 0;
    os.write(reinterpret_cast<const char *>(&version), sizeof(version));
  }
  {
    // the 2st field, LoD information
    // uint64_t lod_level
    // uint64_t lod_level_1 size in byte.
    // int*     lod_level_1 data
    // ...
    auto lod = tensor.lod();
    uint64_t size = lod.size();
    os.write(reinterpret_cast<const char *>(&size), sizeof(size));

    for (auto &each : lod) {
      size = each.size() * sizeof(framework::LoD::value_type::value_type);
      os.write(reinterpret_cast<const char *>(&size), sizeof(size));
      os.write(reinterpret_cast<const char *>(each.data()),
               static_cast<std::streamsize>(size));
    }
  }
  // the 3st field, Tensor
  SerializeToStream(os, static_cast<Tensor>(tensor), dev_ctx);
}

void DeserializeFromStream(std::istream &is, LoDTensor *tensor,
                           const platform::DeviceContext &dev_ctx) {
  {
    // the 1st field, unit32_t version for LoDTensor
    uint32_t version;
    is.read(reinterpret_cast<char *>(&version), sizeof(version));
    PADDLE_ENFORCE_EQ(version, 0U, "Only version 0 is supported");
  }
  {
    // the 2st field, LoD information
    uint64_t lod_level;
    is.read(reinterpret_cast<char *>(&lod_level), sizeof(lod_level));
    auto &lod = *tensor->mutable_lod();
    lod.resize(lod_level);
    for (uint64_t i = 0; i < lod_level; ++i) {
      uint64_t size;
      is.read(reinterpret_cast<char *>(&size), sizeof(size));
      std::vector<size_t> tmp(size / sizeof(size_t));
      is.read(reinterpret_cast<char *>(tmp.data()),
              static_cast<std::streamsize>(size));
      lod[i] = tmp;
    }
  }
  // the 3st filed, Tensor
  DeserializeFromStream(is, static_cast<Tensor *>(tensor), dev_ctx);
}

std::vector<LoDTensor> LoDTensor::SplitLoDTensor(
    const std::vector<platform::Place> places) const {
  check_memory_size();
  //  PADDLE_ENFORCE(lod().empty() || (lod().size() == 1 && lod()[0].empty())
  //                 , "Disable parallel lod for now");
  PADDLE_ENFORCE(lod().empty(), "Disable parallel lod for now");
  PADDLE_ENFORCE(dims()[0] % places.size() == 0,
                 "Batch size should be divided by places size");

  std::vector<LoDTensor> lods;
  for (size_t place_idx = 0; place_idx < places.size(); ++place_idx) {
    size_t begin = place_idx * dims()[0] / places.size();
    size_t end = (place_idx + 1) * dims()[0] / places.size();
    auto src = Slice(static_cast<int>(begin), static_cast<int>(end));

    LoDTensor dst;
    dst.Resize(src.dims());
    auto &dst_place = places[place_idx];
    auto dst_ptr = dst.mutable_data(dst_place, src.type());

    // TODO(tonyyang-svail):
    //   change the following to framework::CopyFrom
    auto src_place = src.place();
    auto src_ptr = src.data<void>();
    auto size = src.numel() * SizeOfType(src.type());
    if (platform::is_cpu_place(src_place) &&
        platform::is_cpu_place(dst_place)) {
      memory::Copy(boost::get<platform::CPUPlace>(dst_place), dst_ptr,
                   boost::get<platform::CPUPlace>(src_place), src_ptr, size);
    } else {
      PADDLE_THROW("Not Implemented");
    }

    lods.emplace_back(dst);
  }

  return lods;
}

void LoDTensor::MergeLoDTensor(
    const std::vector<const LoDTensor *> &lod_tensors, platform::Place place) {
  PADDLE_ENFORCE(platform::is_cpu_place(place));
  PADDLE_ENFORCE(!lod_tensors.empty());

  framework::DDim new_dim = lod_tensors[0]->dims();
  std::type_index new_type = lod_tensors[0]->type();
  for (auto *lod : lod_tensors) {
    PADDLE_ENFORCE(new_dim == lod->dims());
    PADDLE_ENFORCE(new_type == lod->type());
    PADDLE_ENFORCE(platform::is_cpu_place(lod->place()));
  }
  new_dim[0] *= lod_tensors.size();
  Resize(new_dim);

  auto *dst_ptr = reinterpret_cast<uint8_t *>(mutable_data(place, new_type));
  for (auto *src : lod_tensors) {
    auto size = src->numel() * SizeOfType(src->type());
    memory::Copy(boost::get<platform::CPUPlace>(place), dst_ptr,
                 boost::get<platform::CPUPlace>(src->place()),
                 src->data<void>(), size);
    dst_ptr += size;
  }
}

}  // namespace framework
}  // namespace paddle
