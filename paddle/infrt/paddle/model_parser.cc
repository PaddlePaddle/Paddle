// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/infrt/paddle/model_parser.h"

#include <fstream>
#include <vector>

#include "paddle/infrt/common/common.h"
#include "paddle/infrt/common/string.h"
#include "paddle/infrt/common/target.h"
#include "paddle/infrt/common/type.h"

#ifdef INFRT_WITH_PHI
#include "paddle/phi/common/data_type.h"
#endif

namespace infrt {
namespace paddle {

int SizeOfType(framework_proto::VarType::Type type) {
  using Type = framework_proto::VarType::Type;
  switch (static_cast<int>(type)) {
#define DO(desc, type)            \
  case Type::VarType_Type_##desc: \
    return sizeof(type);
    DO(BOOL, bool);
    DO(FP16, float);
    DO(FP32, float);
    DO(INT8, int8_t);
    DO(INT16, int16_t);
    DO(INT32, int);
    DO(INT64, int64_t);
#undef DO
    default:
      LOG(FATAL) << "unknown data type " << type;
  }
  return -1;
}

void TensorFromStream(std::istream &is,
                      _Tensor_ *tensor,
                      const common::Target &target) {
  using Type = framework_proto::VarType::Type;
  uint32_t version;
  is.read(reinterpret_cast<char *>(&version), sizeof(version));
  CHECK_EQ(version, 0U) << "Only version 0 is supported";
  // read tensor desc
  framework_proto::VarType::TensorDesc desc;
  {
    // int32_t size
    // proto buffer
    int32_t size;
    is.read(reinterpret_cast<char *>(&size), sizeof(size));
    std::unique_ptr<char[]> buf(new char[size]);
    is.read(reinterpret_cast<char *>(buf.get()), size);
    CHECK(desc.ParseFromArray(buf.get(), size)) << "Cannot parse tensor desc";
  }

  // read tensor
  std::vector<int32_t> dims_vec;
  std::copy(
      desc.dims().begin(), desc.dims().end(), std::back_inserter(dims_vec));
  Shape dims(dims_vec);
  tensor->Resize(dims);
  void *buf;
  size_t size = tensor->shape().numel() * SizeOfType(desc.data_type());
  // alllocate memory
  if (target.arch == Target::Arch::X86) {
    switch (static_cast<int>(desc.data_type())) {
#define SET_TENSOR(desc, type, precision)     \
  case Type::VarType_Type_##desc:             \
    buf = tensor->mutable_data<type>(target); \
    tensor->set_type(precision);              \
    break

      SET_TENSOR(FP32, float, Float(32));
      SET_TENSOR(INT8, int8_t, Int(8));
      SET_TENSOR(INT16, int16_t, Int(16));
      SET_TENSOR(INT32, int32_t, Int(32));
      SET_TENSOR(INT64, int64_t, Int(64));
#undef SET_TENSOR
      default:
        LOG(FATAL) << "unknown type " << desc.data_type();
    }
    // tensor->set_persistable(true);
    is.read(static_cast<char *>(buf), size);
  } else if (target.arch == Target::Arch::NVGPU) {
#ifdef INFRT_WITH_CUDA
    if (desc.data_type() != Type::VarType_Type_FP32)
      LOG(FATAL) << "[CUDA] The type is not fp32!!";
    auto *data = tensor->mutable_data<float>(target);
    tensor->set_type(infrt::common::Float(32));
    std::vector<float> temp(tensor->shape().numel());
    // LOG(INFO) <<"[CUDA] The tensor's size is "<< tensor->shape().numel();
    is.read(reinterpret_cast<char *>(temp.data()), size);
    CUDA_CALL(cudaMemcpy(reinterpret_cast<void *>(data),
                         temp.data(),
                         tensor->shape().numel() * sizeof(float),
                         cudaMemcpyHostToDevice));
#else
    LOG(FATAL) << "To use CUDA backends, you need to set WITH_CUDA ON!";
#endif
  } else {
    INFRT_NOT_IMPLEMENTED
  }
}

void LoadLoDTensor(std::istream &is, _Variable *var, const Target &target) {
  auto &tensor = var->get<Tensor>();
  uint32_t version{};
  is.read(reinterpret_cast<char *>(&version), sizeof(version));
  VLOG(3) << "model version " << version;

  // Load LoD information
  uint64_t lod_level{};
  is.read(reinterpret_cast<char *>(&lod_level), sizeof(lod_level));

  for (uint64_t i = 0; i < lod_level; ++i) {
    uint64_t size;
    is.read(reinterpret_cast<char *>(&size), sizeof(size));
    std::vector<uint64_t> tmp(size / sizeof(uint64_t));
    is.read(reinterpret_cast<char *>(tmp.data()),
            static_cast<std::streamsize>(size));
    // lod[i] = tmp;
  }

  TensorFromStream(is, tensor.operator->(), target);
}

void ReadBinaryFile(const std::string &filename, std::string *contents) {
  std::ifstream fin(filename, std::ios::in | std::ios::binary);
  CHECK(fin.is_open()) << "Cannot open file: " << filename;
  fin.seekg(0, std::ios::end);
  auto size = fin.tellg();
  contents->clear();
  contents->resize(size);
  fin.seekg(0, std::ios::beg);
  fin.read(&(contents->at(0)), contents->size());
  fin.close();
}

std::unique_ptr<framework_proto::ProgramDesc> LoadProgram(
    const std::string &path, bool program_from_memory) {
  std::unique_ptr<framework_proto::ProgramDesc> main_program(
      new framework_proto::ProgramDesc);
  if (!program_from_memory) {
    std::string desc_str;
    ReadBinaryFile(path, &desc_str);
    main_program->ParseFromString(desc_str);
  } else {
    main_program->ParseFromString(path);
  }
  return main_program;
}

void LoadParams(const std::string &path) {}

// Load directly to CPU, and latter transfer to other devices.
void LoadParam(const std::string &path, _Variable *out, const Target &target) {
  std::ifstream fin(path, std::ios::binary);
  CHECK(fin.is_open()) << "failed to open file " << path;
  LoadLoDTensor(fin, out, target);
}

#ifdef INFRT_WITH_PHI
namespace framework_proto = ::paddle::framework::proto;

inline ::phi::DataType PhiDataType(framework_proto::VarType::Type type) {
  using Type = framework_proto::VarType::Type;
  switch (static_cast<int>(type)) {
    case Type::VarType_Type_BOOL:
      return ::phi::DataType::BOOL;
    case Type::VarType_Type_INT8:
      return ::phi::DataType::INT8;
    case Type::VarType_Type_UINT8:
      return ::phi::DataType::UINT8;
    case Type::VarType_Type_INT16:
      return ::phi::DataType::INT16;
    case Type::VarType_Type_INT32:
      return ::phi::DataType::INT32;
    case Type::VarType_Type_INT64:
      return ::phi::DataType::INT64;
    case Type::VarType_Type_SIZE_T:
      return ::phi::DataType::UINT64;
    case Type::VarType_Type_FP16:
      return ::phi::DataType::FLOAT16;
    case Type::VarType_Type_FP32:
      return ::phi::DataType::FLOAT32;
    case Type::VarType_Type_FP64:
      return ::phi::DataType::FLOAT64;
    default:
      LOG(FATAL) << "unknown data type " << type;
  }
  return ::phi::DataType::UNDEFINED;
}

inline void TensorFromStream(std::istream &is,
                             ::phi::DenseTensor *tensor,
                             const ::phi::CPUContext &ctx) {
  uint32_t version;
  is.read(reinterpret_cast<char *>(&version), sizeof(version));
  CHECK_EQ(version, 0U);
  framework_proto::VarType::TensorDesc desc;
  {  // int32_t size
     // proto buffer
    int32_t size = -1;
    is.read(reinterpret_cast<char *>(&size), sizeof(size));
    CHECK_EQ(is.good(), true);
    CHECK_GE(size, 0);
    std::unique_ptr<char[]> buf(new char[size]);
    is.read(reinterpret_cast<char *>(buf.get()), size);
    CHECK_EQ(desc.ParseFromArray(buf.get(), size), true);
  }
  {  // read tensor
    std::vector<int64_t> dims;
    dims.reserve(static_cast<size_t>(desc.dims().size()));
    std::copy(desc.dims().begin(), desc.dims().end(), std::back_inserter(dims));
    tensor->Resize(::phi::make_ddim(dims));
    void *buf;
    size_t size = tensor->numel() * SizeOfType(desc.data_type());
    ctx.HostAlloc(tensor, PhiDataType(desc.data_type()), size);
    buf = tensor->data();
    is.read(static_cast<char *>(buf), size);
  }
}

void DeserializeFromStream(std::istream &is,
                           ::phi::DenseTensor *tensor,
                           const ::phi::CPUContext &dev_ctx) {
  {
    // the 1st field, unit32_t version for LoDTensor
    uint32_t version;
    is.read(reinterpret_cast<char *>(&version), sizeof(version));
    CHECK_EQ(version, 0U);
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
  TensorFromStream(is, tensor, dev_ctx);
}
#endif

}  // namespace paddle
}  // namespace infrt
