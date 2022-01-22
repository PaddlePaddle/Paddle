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

#include "paddle/fluid/operators/tensor_formatter.h"

#include <string>

namespace paddle {
namespace operators {

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

void TensorFormatter::Print(const framework::LoDTensor& print_tensor,
                            const std::string& tensor_name,
                            const std::string& message) {
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  std::cout << Format(print_tensor, tensor_name, message);
}

std::string TensorFormatter::Format(const framework::LoDTensor& print_tensor,
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
    const framework::LoD& lod = print_tensor.lod();
    for (auto level : lod) {
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
    log_stream << "  - layout: "
               << framework::DataLayoutToString(print_tensor.layout())
               << std::endl;
  }

  std::type_index dtype = framework::ToTypeIndex(print_tensor.type());
  if (print_tensor_type_) {
    log_stream << "  - dtype: " << platform::demangle(dtype.name())
               << std::endl;
  }

  if (framework::IsType<const float>(dtype)) {
    FormatData<float>(print_tensor, log_stream);
  } else if (framework::IsType<const double>(dtype)) {
    FormatData<double>(print_tensor, log_stream);
  } else if (framework::IsType<const int>(dtype)) {
    FormatData<int>(print_tensor, log_stream);
  } else if (framework::IsType<const int64_t>(dtype)) {
    FormatData<int64_t>(print_tensor, log_stream);
  } else if (framework::IsType<const bool>(dtype)) {
    FormatData<bool>(print_tensor, log_stream);
  } else {
    log_stream << "  - data: unprintable type: " << dtype.name() << std::endl;
  }
  return log_stream.str();
}

template <typename T>
void TensorFormatter::FormatData(const framework::LoDTensor& print_tensor,
                                 std::stringstream& log_stream) {
  int64_t print_size = summarize_ == -1
                           ? print_tensor.numel()
                           : std::min(summarize_, print_tensor.numel());
  const T* data = nullptr;
  framework::LoDTensor cpu_tensor;
  if (paddle::platform::is_cpu_place(print_tensor.place())) {
    data = print_tensor.data<T>();
  } else {
    platform::CPUPlace cpu_place;
    paddle::framework::TensorCopy(print_tensor, cpu_place, &cpu_tensor);
#ifdef PADDLE_WITH_ASCEND_CL
    if (platform::is_npu_place(print_tensor.place())) {
      platform::DeviceContextPool::Instance().Get(print_tensor.place())->Wait();
    }
#endif
    data = cpu_tensor.data<T>();
  }

  log_stream << "  - data: [";
  if (print_size > 0) {
    log_stream << data[0];
    for (int64_t i = 1; i < print_size; ++i) {
      log_stream << " " << data[i];
    }
  }
  log_stream << "]" << std::endl;
}

template void TensorFormatter::FormatData<bool>(
    const framework::LoDTensor& print_tensor, std::stringstream& log_stream);
template void TensorFormatter::FormatData<float>(
    const framework::LoDTensor& print_tensor, std::stringstream& log_stream);
template void TensorFormatter::FormatData<double>(
    const framework::LoDTensor& print_tensor, std::stringstream& log_stream);
template void TensorFormatter::FormatData<int>(
    const framework::LoDTensor& print_tensor, std::stringstream& log_stream);
template void TensorFormatter::FormatData<int64_t>(
    const framework::LoDTensor& print_tensor, std::stringstream& log_stream);

}  // namespace operators
}  // namespace paddle
