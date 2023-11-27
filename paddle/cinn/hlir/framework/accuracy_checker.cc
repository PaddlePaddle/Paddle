// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/framework/accuracy_checker.h"

#include <iomanip>
#include <limits>

#ifdef CINN_WITH_CUDA
#include <cuda_runtime.h>
#endif

PD_DECLARE_int64(cinn_self_check_accuracy_num);

namespace cinn {
namespace hlir {
namespace framework {

using cinn::common::bfloat16;
using cinn::common::float16;

template <typename T>
std::string PrintValue(const T& value) {
  std::stringstream ss;
  if (std::is_floating_point<T>::value) {
    ss << std::showpoint;
  }
  ss << std::setprecision(std::numeric_limits<T>::max_digits10);

  if (std::is_integral<T>::value) {
    if (std::is_unsigned<T>::value) {
      ss << static_cast<uint64_t>(value);
    } else {
      ss << static_cast<int64_t>(value);
    }
  } else {
    ss << value;
  }
  return ss.str();
}

template <>
std::string PrintValue<bool>(const bool& value) {
  std::stringstream ss;
  ss << std::boolalpha << value;
  return ss.str();
}

template <typename T, typename Alloc = std::allocator<T>>
std::ostream& operator<<(std::ostream& os, const std::vector<T, Alloc>& vec) {
  os << "{";
  bool is_first = true;
  for (auto e : vec) {
    if (is_first) {
      is_first = false;
    } else {
      os << ", ";
    }
    os << PrintValue(e);
  }
  os << "}";
  return os;
}

template <typename T>
std::string GetTypeString() {
  if (std::is_same<T, float>::value) {
    return "float";
  } else if (std::is_same<T, double>::value) {
    return "double";
  } else if (std::is_same<T, bfloat16>::value) {
    return "bfloat16";
  } else if (std::is_same<T, float16>::value) {
    return "float16";
  } else if (std::is_same<T, int8_t>::value) {
    return "int8_t";
  } else if (std::is_same<T, int16_t>::value) {
    return "int16_t";
  } else if (std::is_same<T, int32_t>::value) {
    return "int32_t";
  } else if (std::is_same<T, int64_t>::value) {
    return "int64_t";
  } else if (std::is_same<T, uint8_t>::value) {
    return "uint8_t";
  } else if (std::is_same<T, uint16_t>::value) {
    return "uint16_t";
  } else if (std::is_same<T, uint32_t>::value) {
    return "uint32_t";
  } else if (std::is_same<T, uint64_t>::value) {
    return "uint64_t";
  } else if (std::is_same<T, bool>::value) {
    return "bool";
  } else {
    CHECK(false) << "Not supported data type.";
    return "";
  }
}

template <typename T>
std::string DebugString(const Tensor& cpu_tensor,
                        const std::string& name,
                        const CheckResult& res) {
  std::stringstream ss;
  ss << "name=" << name << ", dtype=" << GetTypeString<T>()
     << ", shape=" << cpu_tensor->shape().data() << ", data=[";
  size_t numel = cpu_tensor->shape().numel();
  const T* data = cpu_tensor->data<T>();
  size_t print_num = 5L;
  if (FLAGS_cinn_self_check_accuracy_num < 0) {
    print_num = numel;
  } else if (FLAGS_cinn_self_check_accuracy_num > 0) {
    print_num = FLAGS_cinn_self_check_accuracy_num;
  }

  if (numel <= 2 * print_num) {
    for (size_t i = 0; i < numel; ++i) {
      if (i > 0) {
        ss << ", ";
      }
      ss << PrintValue(data[i]);
    }
  } else {
    for (size_t i = 0; i < print_num; ++i) {
      if (i > 0) {
        ss << ", ";
      }
      ss << PrintValue(data[i]);
    }
    ss << ", ... , ";
    for (size_t i = numel - print_num; i < numel; ++i) {
      ss << PrintValue(data[i]);
      if (i != numel - 1) {
        ss << ", ";
      }
    }
  }
  ss << "]";
  if (res == CheckResult::kZero) {
    ss << ", Zero";
  } else if (res == CheckResult::kOne) {
    ss << ", One";
  } else if (res == CheckResult::kNaN) {
    ss << ", NaN";
  } else if (res == CheckResult::kInf) {
    ss << ", Inf";
  } else {
    ss << ", OK";
  }
  return ss.str();
}

std::string AccuracyChecker::operator()(const std::string& arg_name) {
  auto tensor = scope_->GetTensor(arg_name);
  if (tensor->type().is_float(32)) {
    return CheckTensor<float>(tensor, arg_name);
  } else if (tensor->type().is_float(64)) {
    return CheckTensor<double>(tensor, arg_name);
  } else if (tensor->type().is_bfloat16()) {
    return CheckTensor<bfloat16>(tensor, arg_name);
  } else if (tensor->type().is_float16()) {
    return CheckTensor<float16>(tensor, arg_name);
  } else if (tensor->type().is_int(8)) {
    return CheckTensor<int8_t>(tensor, arg_name);
  } else if (tensor->type().is_int(16)) {
    return CheckTensor<int16_t>(tensor, arg_name);
  } else if (tensor->type().is_int(32)) {
    return CheckTensor<int32_t>(tensor, arg_name);
  } else if (tensor->type().is_int(64)) {
    return CheckTensor<int64_t>(tensor, arg_name);
  } else if (tensor->type().is_uint(8)) {
    return CheckTensor<uint8_t>(tensor, arg_name);
  } else if (tensor->type().is_uint(16)) {
    return CheckTensor<uint16_t>(tensor, arg_name);
  } else if (tensor->type().is_uint(32)) {
    return CheckTensor<uint32_t>(tensor, arg_name);
  } else if (tensor->type().is_uint(64)) {
    return CheckTensor<uint64_t>(tensor, arg_name);
  } else if (tensor->type().is_bool()) {
    return CheckTensor<bool>(tensor, arg_name);
  } else {
    CHECK(false) << "Not supported data type.";
    return "";
  }
}

std::string AccuracyChecker::operator()(
    const std::map<std::string, cinn_pod_value_t>* name2podargs,
    const std::string& arg_name) {
  CHECK(name2podargs) << "name2podargs should not be nullptr.";
  const cinn_buffer_t* buffer = cinn_pod_value_to_buffer_p(
      const_cast<cinn_pod_value_t*>(&name2podargs->at(arg_name)));
  if (buffer->type == cinn_float32_t()) {
    return CheckBuffer<float>(buffer, arg_name);
  } else if (buffer->type == cinn_float64_t()) {
    return CheckBuffer<double>(buffer, arg_name);
  } else if (buffer->type == cinn_bfloat16_t()) {
    return CheckBuffer<bfloat16>(buffer, arg_name);
  } else if (buffer->type == cinn_float16_t()) {
    return CheckBuffer<float16>(buffer, arg_name);
  } else if (buffer->type == cinn_int8_t()) {
    return CheckBuffer<int8_t>(buffer, arg_name);
  } else if (buffer->type == cinn_int16_t()) {
    return CheckBuffer<int16_t>(buffer, arg_name);
  } else if (buffer->type == cinn_int32_t()) {
    return CheckBuffer<int32_t>(buffer, arg_name);
  } else if (buffer->type == cinn_int64_t()) {
    return CheckBuffer<int64_t>(buffer, arg_name);
  } else if (buffer->type == cinn_uint8_t()) {
    return CheckBuffer<uint8_t>(buffer, arg_name);
  } else if (buffer->type == cinn_uint16_t()) {
    return CheckBuffer<uint16_t>(buffer, arg_name);
  } else if (buffer->type == cinn_uint32_t()) {
    return CheckBuffer<uint32_t>(buffer, arg_name);
  } else if (buffer->type == cinn_uint64_t()) {
    return CheckBuffer<uint64_t>(buffer, arg_name);
  } else if (buffer->type == cinn_bool_t()) {
    return CheckBuffer<bool>(buffer, arg_name);
  } else {
    CHECK(false) << "Not supported data type.";
    return "";
  }
}

template <typename T>
std::string AccuracyChecker::CheckTensor(const Tensor& tensor,
                                         const std::string& arg_name) {
  Tensor cpu_tensor;
  cpu_tensor->Resize(tensor->shape());
  T* dst = cpu_tensor->mutable_data<T>(common::DefaultHostTarget());

  const T* src = tensor->data<T>();
  size_t numel = tensor->shape().numel();
  MemcpyDeviceToHost(src, numel, dst);

  auto res = CheckNanOrInf<T>(cpu_tensor);
  auto result_str = DebugString<T>(cpu_tensor, arg_name, res);
  return result_str;
}

template <typename T>
std::string AccuracyChecker::CheckBuffer(const cinn_buffer_t* buffer,
                                         const std::string& arg_name) {
  std::vector<int> shape;
  shape.resize(buffer->dimensions);
  for (size_t i = 0; i < shape.size(); ++i) {
    shape[i] = buffer->dims[i];
  }

  Tensor cpu_tensor;
  cpu_tensor->Resize(Shape(shape));
  T* dst = cpu_tensor->mutable_data<T>(common::DefaultHostTarget());

  const T* src = reinterpret_cast<const T*>(buffer->memory);
  size_t numel = cpu_tensor->shape().numel();
  MemcpyDeviceToHost(src, numel, dst);

  auto res = CheckNanOrInf<T>(cpu_tensor);
  auto result_str = DebugString<T>(cpu_tensor, arg_name, res);
  return result_str;
}

template <typename T>
void AccuracyChecker::MemcpyDeviceToHost(const T* src, size_t numel, T* dst) {
#ifdef CINN_WITH_CUDA
  if (target_ == common::DefaultNVGPUTarget()) {
    cudaMemcpy(dst, src, numel * sizeof(T), cudaMemcpyDeviceToHost);
    return;
  }
#endif
  if (target_ == common::DefaultHostTarget()) {
    for (size_t i = 0; i < numel; ++i) {
      dst[i] = src[i];
    }
  } else {
    CHECK(false) << "Not supported target type.";
  }
}

template <typename T>
CheckResult AccuracyChecker::CheckNanOrInf(const Tensor& cpu_tensor) {
  bool zero_flag = true;
  bool one_flag = true;
  size_t numel = cpu_tensor->shape().numel();
  const T* data = cpu_tensor->data<T>();
  for (size_t i = 0; i < numel; ++i) {
    if (std::isnan(data[i])) {
      return CheckResult::kNaN;
    } else if (std::isinf(data[i])) {
      return CheckResult::kInf;
    }
    if (data[i] != static_cast<T>(0)) {
      zero_flag = false;
    }
    if (data[i] != static_cast<T>(1)) {
      one_flag = false;
    }
  }
  if (zero_flag) {
    return CheckResult::kZero;
  } else if (one_flag) {
    return CheckResult::kOne;
  }
  return CheckResult::kOK;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
