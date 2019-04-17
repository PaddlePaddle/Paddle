/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <stdint.h>
#include <string.h>
#include <algorithm>
#include <iterator>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/framework/version.h"

#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/memory/memory.h"

#include "paddle/fluid/recordio/scanner.h"
#include "paddle/fluid/recordio/writer.h"

namespace paddle {
namespace framework {

std::ostream &operator<<(std::ostream &os, const LoD &lod) {
  os << "{";
  for (auto &v : lod) {
    os << "{";
    bool is_first = true;
    for (auto &i : v) {
      if (is_first) {
        os << i;
        is_first = false;
      } else {
        os << ", " << i;
      }
    }
    os << "}";
  }
  os << "}";

  return os;
}

std::ostream &operator<<(std::ostream &os, const LoDTensor &t) {
  if (!platform::is_cpu_place(t.place())) {
    LoDTensor cpu_tensor;
    cpu_tensor.set_lod(t.lod());
    framework::TensorCopy(t, platform::CPUPlace(), &cpu_tensor);
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(t.place());
    dev_ctx.Wait();

    os << cpu_tensor;
    return os;
  }

  os << "dim: " << t.dims() << "\n";
  os << "lod: " << t.lod() << "\n";

  // only print first ten elements
  int64_t size = t.numel() < 10 ? t.numel() : 10;
  for (int64_t i = 0; i < size; ++i) {
    if (t.type() == proto::VarType::FP32) {
      os << t.data<float>()[i] << " ";
    } else if (t.type() == proto::VarType::INT64) {
      os << t.data<int64_t>()[i] << " ";
    } else {
      PADDLE_THROW("LoDTensor data type not in [float, int64_t]");
    }
  }

  return os;
}

std::string LoDToString(const LoD &lod) {
  std::ostringstream stream;
  stream << lod;
  return stream.str();
}

LoD SliceInLevel(const LoD &in, size_t level, size_t elem_begin,
                 size_t elem_end) {
  PADDLE_ENFORCE_LT(level, in.size());
  PADDLE_ENFORCE_LT(elem_begin, elem_end);
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
  for (auto level = static_cast<int>(in.size() - 2); level >= 0; level--) {
    for (size_t i = 0; i < in[level].size(); ++i) {
      size_t index = in[level][i];
      result[level][i] = result[level + 1][index];
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

bool CheckLoD(const LoD &in, int tensor_height) {
  if (in.empty()) return true;
  for (const auto &level : in) {
    // check: there should be more than 2 offsets existing in each level.
    if (level.size() < 2) return false;
    // check: the first offset(the begin offset) of each level should be 0.
    if (level.front() != 0) return false;
    // check: all the offsets in a level should be ascending
    if (!std::is_sorted(level.begin(), level.end())) {
      return false;
    }
  }
  // check: the lowest level's last offset should equals `tensor_height` if
  //        tensor_height>0.
  if (tensor_height > 0 && (size_t)tensor_height != in.back().back())
    return false;

  // check: the higher level's last offset should equals the lower level's
  // size-1.
  // NOTE LoD store the levels from top to bottom, so the higher level goes
  // first.
  for (size_t level = 0; level < in.size() - 1; level++) {
    if (in[level].back() != in[level + 1].size() - 1) return false;
  }
  return true;
}

bool CheckAbsLoD(const LoD &in, int tensor_height) {
  if (in.empty()) return true;
  for (const auto &level : in) {
    // check: all the offsets in a level should be ascending(no same items
    // allows).
    if (!std::is_sorted(level.begin(), level.begin(), [](size_t a, size_t b) {
          if (a <= b) return true;
          return false;
        })) {
      return false;
    }

    // check: there should be more than 2 offsets existing in each level.
    if (level.size() < 2) return false;

    // check: the first offset of each level should be 0, and the last should be
    // the same(the height of underlying tensor).
    if (level.front() != 0) return false;
    if (tensor_height < 0) {
      tensor_height = level.back();
    } else if ((size_t)tensor_height != level.back()) {
      return false;
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
    os.write(reinterpret_cast<const char *>(&kCurTensorVersion),
             sizeof(kCurTensorVersion));
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
  TensorToStream(os, static_cast<Tensor>(tensor), dev_ctx);
}

void DeserializeFromStream(std::istream &is, LoDTensor *tensor,
                           const platform::DeviceContext &dev_ctx) {
  {
    // the 1st field, unit32_t version for LoDTensor
    uint32_t version;
    is.read(reinterpret_cast<char *>(&version), sizeof(version));
    PADDLE_ENFORCE(framework::IsTensorVersionSupported(version),
                   "tensor version %u is not supported.", version);
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
  TensorFromStream(is, static_cast<Tensor *>(tensor), dev_ctx);
}

void WriteToRecordIO(recordio::Writer *writer,
                     const std::vector<LoDTensor> &tensor,
                     const platform::DeviceContext &dev_ctx) {
  std::stringstream buffer;
  size_t sz = tensor.size();
  buffer.write(reinterpret_cast<const char *>(&sz), sizeof(uint32_t));
  for (auto &each : tensor) {
    SerializeToStream(buffer, each, dev_ctx);
  }
  writer->Write(buffer.str());
}

bool ReadFromRecordIO(recordio::Scanner *scanner,
                      const platform::DeviceContext &dev_ctx,
                      std::vector<LoDTensor> *result_ptr) {
  if (!scanner->HasNext()) {
    return false;
  }
  std::istringstream sin(scanner->Next());
  uint32_t sz;
  sin.read(reinterpret_cast<char *>(&sz), sizeof(uint32_t));
  auto &result = *result_ptr;
  result.resize(sz);
  for (uint32_t i = 0; i < sz; ++i) {
    DeserializeFromStream(sin, &result[i], dev_ctx);
  }

  return true;
}

std::vector<LoDTensor> LoDTensor::SplitLoDTensor(
    const std::vector<platform::Place> places) const {
  check_memory_size();
  int batch_size =
      lod().empty() ? dims()[0] : static_cast<int>(lod()[0].size()) - 1;
  size_t result_size = std::min(static_cast<size_t>(batch_size), places.size());
  size_t remainder = batch_size % places.size();

  std::vector<LoDTensor> results;
  results.reserve(result_size);

  int step_width = static_cast<int>(batch_size / result_size);
  for (size_t i = 0; i < result_size; ++i) {
    int begin = static_cast<int>(i * step_width);
    int end = static_cast<int>((i + 1) * step_width);
    if (i + 1 == places.size()) {  // last
      end += remainder;
    }

    LoDTensor dst;
    if (lod().empty()) {
      auto src = Slice(begin, end);
      auto &dst_place = places[i];
      framework::TensorCopy(src, dst_place, &dst);
    } else {
      auto lod_and_offset = GetSubLoDAndAbsoluteOffset(lod(), begin, end, 0);

      auto &offset = lod_and_offset.second;
      auto src = Slice(offset.first, offset.second);
      auto &dst_place = places[i];
      framework::TensorCopy(src, dst_place, &dst);

      LoD my_lod;
      for (auto &l : lod_and_offset.first) {
        std::vector<size_t> v{0};
        for (auto &ll : l) {
          v.push_back(ll + v.back());
        }
        my_lod.emplace_back(v);
      }
      dst.set_lod(my_lod);
    }
    results.emplace_back(dst);
  }

  return results;
}

void LoDTensor::MergeLoDTensor(
    const std::vector<const LoDTensor *> &lod_tensors,
    platform::Place dst_place) {
  PADDLE_ENFORCE(!lod_tensors.empty());

  framework::DDim new_dim = lod_tensors[0]->dims();
  auto new_type = lod_tensors[0]->type();
  framework::DataLayout new_layout = lod_tensors[0]->layout();
  LoD new_lod = lod_tensors[0]->lod();
  for (size_t i = 1; i < lod_tensors.size(); ++i) {
    auto *t = lod_tensors[i];
    PADDLE_ENFORCE_EQ(new_type, t->type());
    PADDLE_ENFORCE_EQ(new_layout, t->layout());

    PADDLE_ENFORCE_EQ(framework::product(new_dim) / new_dim[0],
                      framework::product(t->dims()) / t->dims()[0]);
    new_dim[0] += t->dims()[0];

    auto &lod = t->lod();
    PADDLE_ENFORCE_EQ(new_lod.size(), lod.size());
    for (size_t j = 0; j < lod.size(); ++j) {
      auto &sub_lod = new_lod[j];
      size_t offset = sub_lod.back();
      for (size_t k = 1; k < lod[j].size(); ++k) {
        sub_lod.push_back(lod[j][k] + offset);
      }
    }
  }
  Resize(new_dim);
  set_layout(new_layout);
  set_lod(new_lod);
  mutable_data(dst_place, new_type);

  int begin = 0;
  for (auto *src : lod_tensors) {
    int end = begin + src->dims()[0];
    auto dst = Slice(begin, end);
    framework::TensorCopy(*src, dst_place, &dst);
    begin = end;
  }
}

LoD ConvertToLengthBasedLoD(const LoD &offset_lod) {
  LoD length_lod;
  length_lod.reserve(offset_lod.size());
  for (size_t lvl = 0; lvl < offset_lod.size(); ++lvl) {
    std::vector<size_t> level;
    if (offset_lod[lvl].size() > 0) {
      level.reserve(offset_lod[lvl].size() - 1);
    }
    for (size_t idx = 0; idx < offset_lod[lvl].size() - 1; ++idx) {
      level.push_back(offset_lod[lvl][idx + 1] - offset_lod[lvl][idx]);
    }
    length_lod.push_back(level);
  }
  return length_lod;
}

LoD ConvertToOffsetBasedLoD(const LoD &length_lod) {
  LoD offset_lod;
  offset_lod.reserve(length_lod.size());
  for (size_t lvl = 0; lvl < length_lod.size(); ++lvl) {
    std::vector<size_t> level;
    level.reserve(length_lod[lvl].size() + 1);
    size_t tmp = 0;
    level.push_back(tmp);
    for (size_t idx = 0; idx < length_lod[lvl].size(); ++idx) {
      tmp += length_lod[lvl][idx];
      level.push_back(tmp);
    }
    offset_lod.push_back(level);
  }
  return offset_lod;
}

}  // namespace framework
}  // namespace paddle
