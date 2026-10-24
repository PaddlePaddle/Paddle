/* Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/phi/kernels/funcs/tensor_to_npy.h"

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi::funcs {
namespace {

std::string LittleEndianDescr(const std::string& suffix) {
  return std::string(1, static_cast<char>(60)) + suffix;
}

std::string AddBf16MarkerToNpyPath(const std::string& file_path) {
  constexpr char kNpySuffix[] = ".npy";
  constexpr char kBf16Marker[] = ".bf16";
  if (file_path.size() >= 4 &&
      file_path.compare(file_path.size() - 4, 4, kNpySuffix) == 0) {
    return file_path.substr(0, file_path.size() - 4) + kBf16Marker + kNpySuffix;
  }
  return file_path + kBf16Marker;
}

std::string NpyDescr(DataType dtype) {
  switch (dtype) {
    case DataType::BOOL:
      return "|b1";
    case DataType::UINT8:
      return "|u1";
    case DataType::INT8:
      return "|i1";
    case DataType::UINT16:
      return LittleEndianDescr("u2");
    case DataType::INT16:
      return LittleEndianDescr("i2");
    case DataType::UINT32:
      return LittleEndianDescr("u4");
    case DataType::INT32:
      return LittleEndianDescr("i4");
    case DataType::UINT64:
      return LittleEndianDescr("u8");
    case DataType::INT64:
      return LittleEndianDescr("i8");
    case DataType::FLOAT16:
      return LittleEndianDescr("f2");
    case DataType::BFLOAT16:
    case DataType::FLOAT32:
      return LittleEndianDescr("f4");
    case DataType::FLOAT64:
      return LittleEndianDescr("f8");
    case DataType::COMPLEX64:
      return LittleEndianDescr("c8");
    case DataType::COMPLEX128:
      return LittleEndianDescr("c16");
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Saving DenseTensor with dtype %s to numpy npy is not supported.",
          DataTypeToString(dtype)));
  }
}

std::string ShapeString(const DDim& dims) {
  std::ostringstream os;
  os << "(";
  for (int i = 0; i < dims.size(); ++i) {
    if (i > 0) {
      os << ", ";
    }
    os << dims[i];
  }
  if (dims.size() == 1) {
    os << ",";
  }
  os << ")";
  return os.str();
}

std::string BuildNpyHeader(const DenseTensor& tensor, bool use_v2_header) {
  std::ostringstream os;
  os << "{'descr': '" << NpyDescr(tensor.dtype())
     << "', 'fortran_order': False, 'shape': " << ShapeString(tensor.dims())
     << ", }";

  std::string header = os.str();
  const size_t prefix_size = use_v2_header ? 12 : 10;
  size_t padding = 16 - ((prefix_size + header.size() + 1) % 16);
  if (padding == 16) {
    padding = 0;
  }
  header.append(padding, ' ');
  header.push_back('\n');
  return header;
}

void WriteNpyHeader(std::ofstream* outfile, const DenseTensor& tensor) {
  std::string header = BuildNpyHeader(tensor, false);
  bool use_v2_header = header.size() > UINT16_MAX;
  if (use_v2_header) {
    header = BuildNpyHeader(tensor, true);
  }

  outfile->write("\x93NUMPY", 6);
  const char major = use_v2_header ? 2 : 1;
  const char minor = 0;
  outfile->put(major);
  outfile->put(minor);

  if (use_v2_header) {
    uint32_t header_len = static_cast<uint32_t>(header.size());
    outfile->write(reinterpret_cast<const char*>(&header_len),
                   sizeof(header_len));
  } else {
    uint16_t header_len = static_cast<uint16_t>(header.size());
    outfile->write(reinterpret_cast<const char*>(&header_len),
                   sizeof(header_len));
  }
  outfile->write(header.data(), static_cast<std::streamsize>(header.size()));
}

const DenseTensor& GetCpuTensor(const DenseTensor& tensor,
                                DenseTensor* cpu_tensor) {
  if (tensor.place().GetType() == AllocationType::CPU) {
    if (tensor.meta().is_contiguous()) {
      return tensor;
    }
    CPUPlace cpu_place;
    DeviceContextPool& pool = DeviceContextPool::Instance();
    auto dev_ctx = pool.Get(tensor.place());
    phi::Copy(*dev_ctx, tensor, cpu_place, true, cpu_tensor);
    return *cpu_tensor;
  }

  CPUPlace cpu_place;
  DeviceContextPool& pool = DeviceContextPool::Instance();
  auto dev_ctx = pool.Get(tensor.place());
  phi::Copy(*dev_ctx, tensor, cpu_place, true, cpu_tensor);
  return *cpu_tensor;
}

void WriteNpyData(std::ofstream* outfile, const DenseTensor& tensor) {
  if (tensor.dtype() == DataType::BFLOAT16) {
    std::vector<float> fp32_data(static_cast<size_t>(tensor.numel()));
    const auto* bf16_data = static_cast<const phi::bfloat16*>(tensor.data());
    for (int64_t i = 0; i < tensor.numel(); ++i) {
      fp32_data[static_cast<size_t>(i)] = static_cast<float>(bf16_data[i]);
    }
    if (!fp32_data.empty()) {
      outfile->write(
          reinterpret_cast<const char*>(fp32_data.data()),
          static_cast<std::streamsize>(fp32_data.size() * sizeof(float)));
    }
    return;
  }

  const size_t bytes =
      static_cast<size_t>(tensor.numel()) * phi::SizeOf(tensor.dtype());
  if (bytes > 0) {
    outfile->write(reinterpret_cast<const char*>(tensor.data()),
                   static_cast<std::streamsize>(bytes));
  }
}

}  // namespace

void TensorToNpySaver::Save(const DenseTensor& tensor,
                            const std::string& file_path) {
  PADDLE_ENFORCE_EQ(
      tensor.has_allocation(),
      true,
      common::errors::InvalidArgument(
          "DenseTensor must have allocation before saving to numpy npy."));

  DenseTensor cpu_tensor;
  const DenseTensor& saved_tensor = GetCpuTensor(tensor, &cpu_tensor);

  const std::string saved_path = saved_tensor.dtype() == DataType::BFLOAT16
                                     ? AddBf16MarkerToNpyPath(file_path)
                                     : file_path;
  std::ofstream outfile(saved_path, std::ios::binary | std::ios::out);
  PADDLE_ENFORCE_EQ(outfile.is_open(),
                    true,
                    common::errors::Unavailable(
                        "Cannot open npy file %s for writing.", saved_path));

  WriteNpyHeader(&outfile, saved_tensor);
  WriteNpyData(&outfile, saved_tensor);

  PADDLE_ENFORCE_EQ(
      outfile.good(),
      true,
      common::errors::Unavailable("Failed to write DenseTensor to npy file %s.",
                                  saved_path));
}

void SaveDenseTensorToNpy(const DenseTensor& tensor,
                          const std::string& file_path) {
  TensorToNpySaver saver;
  saver.Save(tensor, file_path);
}

}  // namespace phi::funcs
