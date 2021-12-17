// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/utils/io_utils.h"

#include <fcntl.h>

#include <utility>

#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/utils/shape_range_info.pb.h"

namespace paddle {
namespace inference {

// =========================================================
//       Item        |        Type       |      Bytes
// ---------------------------------------------------------
//      Version      |      uint32_t     |        4
// ---------------------------------------------------------
//   Bytes of `Name` |      uint64_t     |        8
//        Name       |        char       |  Bytes of `Name`
// ---------------------------------------------------------
//      LoD Level    |      uint64_t     |        8
//  Bytes of `LoD[0]`|      uint64_t     |        8
//       LoD[0]      |      uint64_t     | Bytes of `LoD[0]`
//        ...        |         ...       |       ...
// ---------------------------------------------------------
//   Dims of `Shape` |      uint64_t     |        8
//       Shape       |      uint64_t     |    Dims * 4
// ---------------------------------------------------------
//       Dtype       |       int32_t     |        4
//  Bytes of `Data`  |      uint64_t     |        8
//        Data       |        Dtype      |  Bytes of `Data`
// =========================================================
void SerializePDTensorToStream(std::ostream *os, const PaddleTensor &tensor) {
  // 1. Version
  os->write(reinterpret_cast<const char *>(&kCurPDTensorVersion),
            sizeof(kCurPDTensorVersion));
  // 2. Name
  uint64_t name_bytes = tensor.name.size();
  os->write(reinterpret_cast<char *>(&name_bytes), sizeof(name_bytes));
  os->write(tensor.name.c_str(), name_bytes);
  // 3. LoD
  auto lod = tensor.lod;
  uint64_t lod_size = lod.size();
  os->write(reinterpret_cast<const char *>(&lod_size), sizeof(lod_size));
  for (auto &each : lod) {
    auto size = each.size() * sizeof(size_t);
    os->write(reinterpret_cast<const char *>(&size), sizeof(size));
    os->write(reinterpret_cast<const char *>(each.data()),
              static_cast<std::streamsize>(size));
  }
  // 4. Shape
  size_t dims = tensor.shape.size();
  os->write(reinterpret_cast<const char *>(&dims), sizeof(dims));
  os->write(reinterpret_cast<const char *>(tensor.shape.data()),
            sizeof(int) * dims);
  // 5. Data
  os->write(reinterpret_cast<const char *>(&tensor.dtype),
            sizeof(tensor.dtype));
  uint64_t length = tensor.data.length();
  os->write(reinterpret_cast<const char *>(&length), sizeof(size_t));
  os->write(reinterpret_cast<const char *>(tensor.data.data()), length);
}

void DeserializePDTensorToStream(std::istream &is, PaddleTensor *tensor) {
  // 1. Version
  uint32_t version;
  is.read(reinterpret_cast<char *>(&version), sizeof(version));
  // 2. Name
  uint64_t name_bytes;
  is.read(reinterpret_cast<char *>(&name_bytes), sizeof(name_bytes));
  std::vector<char> bytes(name_bytes);
  is.read(bytes.data(), name_bytes);
  tensor->name = std::string(bytes.data(), name_bytes);
  // 3. LoD
  uint64_t lod_level;
  is.read(reinterpret_cast<char *>(&lod_level), sizeof(lod_level));
  auto *lod = &(tensor->lod);
  lod->resize(lod_level);
  for (uint64_t i = 0; i < lod_level; ++i) {
    uint64_t size;
    is.read(reinterpret_cast<char *>(&size), sizeof(size));
    std::vector<size_t> tmp(size / sizeof(size_t));
    is.read(reinterpret_cast<char *>(tmp.data()),
            static_cast<std::streamsize>(size));
    (*lod)[i] = tmp;
  }
  // 4. Shape
  size_t dims;
  is.read(reinterpret_cast<char *>(&dims), sizeof(dims));
  tensor->shape.resize(dims);
  is.read(reinterpret_cast<char *>(tensor->shape.data()), sizeof(int) * dims);
  // 5. Data
  uint64_t length;
  is.read(reinterpret_cast<char *>(&tensor->dtype), sizeof(tensor->dtype));
  is.read(reinterpret_cast<char *>(&length), sizeof(length));
  tensor->data.Resize(length);
  is.read(reinterpret_cast<char *>(tensor->data.data()), length);
}

// =========================================================
//       Item        |        Type       |      Bytes
// ---------------------------------------------------------
//      Version      |      uint32_t     |        4
// ---------------------------------------------------------
//   Size of Tensors |      uint64_t     |        8
//      Tensors      |        ----       |       ---
// ---------------------------------------------------------
void SerializePDTensorsToStream(std::ostream *os,
                                const std::vector<PaddleTensor> &tensors) {
  // 1. Version
  os->write(reinterpret_cast<const char *>(&kCurPDTensorVersion),
            sizeof(kCurPDTensorVersion));
  // 2. Tensors
  uint64_t num = tensors.size();
  os->write(reinterpret_cast<char *>(&num), sizeof(num));
  for (const auto &tensor : tensors) {
    SerializePDTensorToStream(os, tensor);
  }
}

void DeserializePDTensorsToStream(std::istream &is,
                                  std::vector<PaddleTensor> *tensors) {
  // 1. Version
  uint32_t version;
  is.read(reinterpret_cast<char *>(&version), sizeof(version));
  // 2. Tensors
  uint64_t num;
  is.read(reinterpret_cast<char *>(&num), sizeof(num));
  tensors->resize(num);
  for (auto &tensor : *tensors) {
    DeserializePDTensorToStream(is, &tensor);
  }
}

void SerializePDTensorsToFile(const std::string &path,
                              const std::vector<PaddleTensor> &tensors) {
  std::ofstream fout(path, std::ios::binary);
  SerializePDTensorsToStream(&fout, tensors);
  fout.close();
}

void DeserializePDTensorsToFile(const std::string &path,
                                std::vector<PaddleTensor> *tensors) {
  bool is_present = analysis::FileExists(path);
  PADDLE_ENFORCE_EQ(is_present, true, platform::errors::InvalidArgument(
                                          "Cannot open %s to read", path));
  std::ifstream fin(path, std::ios::binary);
  DeserializePDTensorsToStream(fin, tensors);
  fin.close();
}

void SerializeShapeRangeInfo(
    const std::string &path,
    const paddle::inference::proto::ShapeRangeInfos &info) {
  int out_fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  google::protobuf::io::FileOutputStream *os =
      new google::protobuf::io::FileOutputStream(out_fd);
  google::protobuf::TextFormat::Print(info, os);
  delete os;
  close(out_fd);
}

void SerializeShapeRangeInfo(
    const std::string &path,
    const std::map<std::string, std::vector<int32_t>> &min_shape,
    const std::map<std::string, std::vector<int32_t>> &max_shape,
    const std::map<std::string, std::vector<int32_t>> &opt_shape) {
  paddle::inference::proto::ShapeRangeInfos shape_range_infos;
  for (auto it : min_shape) {
    auto *s = shape_range_infos.add_shape_range_info();
    s->set_name(it.first);
    for (size_t i = 0; i < it.second.size(); ++i) {
      s->add_min_shape(it.second[i]);
      s->add_max_shape(max_shape.at(it.first)[i]);
      s->add_opt_shape(opt_shape.at(it.first)[i]);
    }
  }

  inference::SerializeShapeRangeInfo(path, shape_range_infos);
}
void DeserializeShapeRangeInfo(
    const std::string &path, paddle::inference::proto::ShapeRangeInfos *info) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd == -1) {
    PADDLE_THROW(platform::errors::NotFound("File [%s] is not found.", path));
  }
  google::protobuf::io::FileInputStream *is =
      new google::protobuf::io::FileInputStream(fd);
  google::protobuf::TextFormat::Parse(is, info);
  delete is;
  close(fd);
}

void DeserializeShapeRangeInfo(
    const std::string &path,
    std::map<std::string, std::vector<int32_t>> *min_shape,
    std::map<std::string, std::vector<int32_t>> *max_shape,
    std::map<std::string, std::vector<int32_t>> *opt_shape) {
  paddle::inference::proto::ShapeRangeInfos shape_range_infos;
  DeserializeShapeRangeInfo(path, &shape_range_infos);
  for (int i = 0; i < shape_range_infos.shape_range_info_size(); ++i) {
    auto info = shape_range_infos.shape_range_info(i);
    auto name = info.name();
    if (min_shape->count(name) || max_shape->count(name) ||
        opt_shape->count(name)) {
      continue;
    } else {
      std::vector<int32_t> tmp(info.min_shape_size());
      for (size_t k = 0; k < tmp.size(); ++k) tmp[k] = info.min_shape(k);
      min_shape->insert(std::make_pair(name, tmp));

      tmp.resize(info.max_shape_size());
      for (size_t k = 0; k < tmp.size(); ++k) tmp[k] = info.max_shape(k);
      max_shape->insert(std::make_pair(name, tmp));

      tmp.resize(info.opt_shape_size());
      for (size_t k = 0; k < tmp.size(); ++k) tmp[k] = info.opt_shape(k);
      opt_shape->insert(std::make_pair(name, tmp));
    }
  }
}

void UpdateShapeRangeInfo(
    const std::string &path,
    const std::map<std::string, std::vector<int32_t>> &min_shape,
    const std::map<std::string, std::vector<int32_t>> &max_shape,
    const std::map<std::string, std::vector<int32_t>> &opt_shape,
    const std::vector<std::string> &names) {
  paddle::inference::proto::ShapeRangeInfos shape_range_infos;
  DeserializeShapeRangeInfo(path, &shape_range_infos);

  for (int i = 0; i < shape_range_infos.shape_range_info_size(); ++i) {
    auto *info = shape_range_infos.mutable_shape_range_info(i);
    for (const auto &name : names) {
      if (info->name() == name) {
        info->clear_min_shape();
        info->clear_max_shape();
        info->clear_opt_shape();
        for (size_t j = 0; j < min_shape.at(name).size(); ++j)
          info->add_min_shape(min_shape.at(name)[j]);
        for (size_t j = 0; j < max_shape.at(name).size(); ++j)
          info->add_max_shape(max_shape.at(name)[j]);
        for (size_t j = 0; j < opt_shape.at(name).size(); ++j)
          info->add_opt_shape(opt_shape.at(name)[j]);
        break;
      }
    }
  }
  inference::SerializeShapeRangeInfo(path, shape_range_infos);
}

}  // namespace inference
}  // namespace paddle
