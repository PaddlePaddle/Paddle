/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/phi/kernels/funcs/tensor_formatter.h"

#include <string>

#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/tensor_utils.h"

namespace paddle {
namespace funcs {

void TensorFormatter::SetPrintTensorType(bool print_tensor_type) {
  print_tensor_type_ = print_tensor_type;
}

void TensorFormatter::SetPrintTensorShape(bool print_tensor_shape) {
  print_tensor_shape_ = print_tensor_shape;
}

void TensorFormatter::SetPrintTensorLod(bool print_tensor_lod) {
  print_tensor_lod_ = print_tensor_lod;
}

void TensorFormatter::SetPrintTensorLayout(bool print_tensor_layout) {
  print_tensor_layout_ = print_tensor_layout;
}

void TensorFormatter::SetSummarize(int64_t summarize) {
  summarize_ = summarize;
}

void TensorFormatter::Print(const phi::DenseTensor& print_tensor,
                            const std::string& tensor_name,
                            const std::string& message) {
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  std::cout << Format(print_tensor, tensor_name, message);
}

std::string TensorFormatter::Format(const phi::DenseTensor& print_tensor,
                                    const std::string& tensor_name,
                                    const std::string& message) {
  std::stringstream log_stream;
  if (!tensor_name.empty()) {
    log_stream << "Variable: " << tensor_name << std::endl;
  }

  if (!message.empty()) {
    log_stream << "  - message: " << message << std::endl;
  }

  if (print_tensor_lod_) {
    log_stream << "  - lod: {";
    const phi::LegacyLoD& lod = print_tensor.lod();
    for (auto const& level : lod) {
      log_stream << "{";
      bool is_first = true;
      for (auto i : level) {
        if (is_first) {
          log_stream << i;
          is_first = false;
        } else {
          log_stream << ", " << i;
        }
      }
      log_stream << "}";
    }
    log_stream << "}" << std::endl;
  }

  log_stream << "  - place: " << print_tensor.place() << std::endl;

  if (print_tensor_shape_) {
    log_stream << "  - shape: " << print_tensor.dims().to_str() << std::endl;
  }

  if (print_tensor_layout_) {
    log_stream << "  - layout: " << print_tensor.layout() << std::endl;
  }

  auto dtype = print_tensor.dtype();
  if (print_tensor_type_) {
    log_stream << "  - dtype: " << dtype << std::endl;
  }

  if (dtype == phi::DataType::FLOAT32) {
    FormatData<float>(print_tensor, log_stream);
  } else if (dtype == phi::DataType::FLOAT64) {
    FormatData<double>(print_tensor, log_stream);
  } else if (dtype == phi::DataType::INT32) {
    FormatData<int>(print_tensor, log_stream);
  } else if (dtype == phi::DataType::INT64) {
    FormatData<int64_t>(print_tensor, log_stream);
  } else if (dtype == phi::DataType::BOOL) {
    FormatData<bool>(print_tensor, log_stream);
  } else if (dtype == phi::DataType::FLOAT16) {
    FormatData<phi::dtype::float16>(print_tensor, log_stream);
  } else if (dtype == phi::DataType::BFLOAT16) {
    FormatData<phi::dtype::bfloat16>(print_tensor, log_stream);
  } else if (dtype == phi::DataType::FLOAT8_E4M3FN) {
    FormatData<phi::dtype::float8_e4m3fn>(print_tensor, log_stream);
  } else if (dtype == phi::DataType::FLOAT8_E5M2) {
    FormatData<phi::dtype::float8_e5m2>(print_tensor, log_stream);
  } else {
    log_stream << "  - data: unprintable type: " << dtype << std::endl;
  }
  return log_stream.str();
}

template <typename T>
void TensorFormatter::FormatData(const phi::DenseTensor& print_tensor,
                                 std::stringstream& log_stream) {
  int64_t print_size = summarize_ == -1
                           ? print_tensor.numel()
                           : std::min(summarize_, print_tensor.numel());
  const T* data = nullptr;
  phi::DenseTensor cpu_tensor;
  if (print_tensor.place().GetType() == phi::AllocationType::CPU) {
    data = print_tensor.data<T>();
  } else {
    phi::CPUPlace cpu_place;

    phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
    auto dev_ctx = pool.Get(print_tensor.place());

    phi::Copy(*dev_ctx, print_tensor, cpu_place, true, &cpu_tensor);
    data = cpu_tensor.data<T>();
  }

  log_stream << "  - data: [";
  if (print_size > 0) {
    log_stream << data[0];
    for (int64_t i = 1; i < print_size; ++i) {
      log_stream << " " << static_cast<float>(data[i]);
    }
  }
  log_stream << "]" << std::endl;
}

template void TensorFormatter::FormatData<bool>(
    const phi::DenseTensor& print_tensor, std::stringstream& log_stream);
template void TensorFormatter::FormatData<float>(
    const phi::DenseTensor& print_tensor, std::stringstream& log_stream);
template void TensorFormatter::FormatData<double>(
    const phi::DenseTensor& print_tensor, std::stringstream& log_stream);
template void TensorFormatter::FormatData<int>(
    const phi::DenseTensor& print_tensor, std::stringstream& log_stream);
template void TensorFormatter::FormatData<int64_t>(
    const phi::DenseTensor& print_tensor, std::stringstream& log_stream);
template void TensorFormatter::FormatData<phi::dtype::float16>(
    const phi::DenseTensor& print_tensor, std::stringstream& log_stream);
template void TensorFormatter::FormatData<phi::dtype::bfloat16>(
    const phi::DenseTensor& print_tensor, std::stringstream& log_stream);

}  // namespace funcs
}  // namespace paddle
