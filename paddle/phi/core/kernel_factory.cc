//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/kernel_factory.h"

#include "glog/logging.h"
#include "paddle/phi/core/enforce.h"
#if defined(PADDLE_WITH_XPU) && !defined(PADDLE_WITH_XPU_KP)
#include "paddle/phi/backends/xpu/xpu_op_list.h"
#include "paddle/phi/core/compat/convert_utils.h"
#endif
#include "paddle/phi/core/compat/op_utils.h"
#include "paddle/utils/string/string_helper.h"

DECLARE_bool(enable_api_kernel_fallback);

namespace phi {

const static Kernel empty_kernel;  // NOLINT

std::string KernelSelectionErrorMessage(const std::string& kernel_name,
                                        const KernelKey& target_key);

uint32_t KernelKey::Hash::operator()(const KernelKey& key) const {
  uint32_t hash_value = 0;
  // |----31-20------|---19-12---|---11-8----|---7-0---|
  // | For extension | DataType | DataLayout | Backend |
  hash_value |= static_cast<uint8_t>(key.backend());
  hash_value |=
      (static_cast<uint8_t>(key.layout()) << KernelKey::kBackendBitLength);
  hash_value |=
      (static_cast<uint16_t>(key.dtype())
       << (KernelKey::kBackendBitLength + KernelKey::kDataLayoutBitLength));
  return hash_value;
}

KernelFactory& KernelFactory::Instance() {
  static KernelFactory g_op_kernel_factory;
  return g_op_kernel_factory;
}

bool KernelFactory::HasCompatiblePhiKernel(const std::string& op_type) const {
  if (deprecated_op_names.find(op_type) == deprecated_op_names.end()) {
    if (phi::OpUtilsMap::Instance().Contains(op_type)) {
      return true;
    } else if (kernels_.find(op_type) != kernels_.end()) {
      return true;
    }
  }
  return false;
}

const Kernel& KernelFactory::SelectKernel(const std::string& kernel_name,
                                          const KernelKey& kernel_key) const {
  auto iter = kernels_.find(kernel_name);
  if (iter == kernels_.end()) {
    return empty_kernel;
  }
  auto kernel_iter = iter->second.find(kernel_key);
  if (kernel_iter == iter->second.end() &&
      kernel_key.layout() != phi::DataLayout::ALL_LAYOUT) {
    phi::KernelKey any_layout_kernel_key(
        kernel_key.backend(), phi::DataLayout::ALL_LAYOUT, kernel_key.dtype());
    kernel_iter = iter->second.find(any_layout_kernel_key);
  }

  if (kernel_iter == iter->second.end()) {
    return empty_kernel;
  }

  return kernel_iter->second;
}

KernelKeyMap KernelFactory::SelectKernelMap(
    const std::string& kernel_name) const {
  auto iter = kernels_.find(kernel_name);
  if (iter == kernels_.end()) {
    return KernelKeyMap();
  }
  return iter->second;
}

bool KernelFactory::HasKernel(const std::string& kernel_name,
                              const KernelKey& kernel_key) const {
  auto iter = kernels_.find(kernel_name);
  PADDLE_ENFORCE_NE(
      iter,
      kernels_.end(),
      phi::errors::NotFound("The kernel `%s` is not registered.", kernel_name));

  auto kernel_iter = iter->second.find(kernel_key);
  if (kernel_iter == iter->second.end()) {
    return false;
  }
  return true;
}

KernelResult KernelFactory::SelectKernelOrThrowError(
    const std::string& kernel_name, const KernelKey& const_kernel_key) const {
  auto iter = kernels_.find(kernel_name);
  PADDLE_ENFORCE_NE(
      iter,
      kernels_.end(),
      phi::errors::NotFound("The kernel `%s` is not registered.", kernel_name));

  KernelKey kernel_key = KernelKey(const_kernel_key.backend(),
                                   phi::DataLayout::ALL_LAYOUT,
                                   const_kernel_key.dtype());
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (kernel_key.backend() == Backend::GPUDNN) {
    auto kernel_iter = iter->second.find(
        {Backend::GPUDNN, phi::DataLayout::ALL_LAYOUT, kernel_key.dtype()});
    if (kernel_iter != iter->second.end()) {
      return {kernel_iter->second, false};
    }
    kernel_key =
        KernelKey(Backend::GPU, kernel_key.layout(), kernel_key.dtype());
  }
#endif
  auto kernel_iter = iter->second.find(kernel_key);

  PADDLE_ENFORCE_NE(
      kernel_iter == iter->second.end() && kernel_key.backend() == Backend::CPU,
      true,
      phi::errors::NotFound(
          "The kernel with key %s of kernel `%s` is not registered. %s",
          kernel_key,
          kernel_name,
          KernelSelectionErrorMessage(kernel_name, kernel_key)));

#if defined(PADDLE_WITH_XPU) && !defined(PADDLE_WITH_XPU_KP)
  VLOG(6) << "fluid_op_name: " << TransToFluidOpName(kernel_name);
  if ((FLAGS_enable_api_kernel_fallback && kernel_iter == iter->second.end()) ||
      !phi::backends::xpu::is_xpu_support_op(TransToFluidOpName(kernel_name),
                                             kernel_key.dtype())
#else
  if ((FLAGS_enable_api_kernel_fallback && kernel_iter == iter->second.end())
#endif
  ) {
    // Fallback CPU backend
    phi::KernelKey cpu_kernel_key(
        phi::Backend::CPU, kernel_key.layout(), kernel_key.dtype());
    kernel_iter = iter->second.find(cpu_kernel_key);

    PADDLE_ENFORCE_NE(
        kernel_iter,
        iter->second.end(),
        phi::errors::NotFound(
            "The kernel with key %s of kernel `%s` is not registered and "
            "fail to fallback to CPU one. %s",
            kernel_key,
            kernel_name,
            KernelSelectionErrorMessage(kernel_name, kernel_key)));

    VLOG(3) << "missing " << kernel_key.backend() << " kernel: " << kernel_name
            << ", expected_kernel_key:" << kernel_key
            << ", fallbacking to CPU one!";

    return {kernel_iter->second, true};
  }

  PADDLE_ENFORCE_NE(
      kernel_iter,
      iter->second.end(),
      phi::errors::NotFound(
          "The kernel with key %s of kernel `%s` is not registered. %s "
          "The current value of FLAGS_enable_api_kernel_fallback(bool,"
          " default true) is false. If you want to fallback this kernel"
          " to CPU one, please set the flag true before run again.",
          kernel_key,
          kernel_name,
          KernelSelectionErrorMessage(kernel_name, kernel_key)));

  return {kernel_iter->second, false};
}

const KernelArgsDef& KernelFactory::GetFirstKernelArgsDef(
    const std::string& kernel_name) const {
  auto iter = kernels_.find(kernel_name);
  PADDLE_ENFORCE_NE(
      iter,
      kernels_.end(),
      phi::errors::NotFound("The kernel `%s` is not registered.", kernel_name));
  return iter->second.cbegin()->second.args_def();
}

std::ostream& operator<<(std::ostream& os, AttributeType attr_type) {
  switch (attr_type) {
    case AttributeType::BOOL:
      os << "bool";
      break;
    case AttributeType::INT32:
      os << "int";
      break;
    case AttributeType::INT64:
      os << "int64_t";
      break;
    case AttributeType::FLOAT32:
      os << "float";
      break;
    case AttributeType::FLOAT64:
      os << "double";
      break;
    case AttributeType::STRING:
      os << "string";
      break;
    case AttributeType::BOOLS:
      os << "vector<bool>";
      break;
    case AttributeType::INT32S:
      os << "vector<int>";
      break;
    case AttributeType::INT64S:
      os << "vector<int64_t>";
      break;
    case AttributeType::FLOAT32S:
      os << "vector<float>";
      break;
    case AttributeType::FLOAT64S:
      os << "vector<double>";
      break;
    case AttributeType::STRINGS:
      os << "vector<string>";
      break;
    case AttributeType::SCALAR:
      os << "Scalar";
      break;
    case AttributeType::SCALARS:
      os << "vector<Scalar>";
      break;
    case AttributeType::INT_ARRAY:
      os << "IntArray";
      break;
    case AttributeType::DATA_TYPE:
      os << "DataType";
      break;
    case AttributeType::DATA_LAYOUT:
      os << "DataLayout";
      break;
    case AttributeType::PLACE:
      os << "Place";
      break;
    default:
      os << "Undefined";
  }
  return os;
}

// print kernel info with json format:
// {
//   "(CPU, Undefined(AnyLayout), complex64)": {
//   "input": ["CPU, NCHW, complex64", "CPU, NCHW, complex64"],
//   "output": ["CPU, NCHW, complex64"],
//   "attribute": ["i"]
// }
std::ostream& operator<<(std::ostream& os, const Kernel& kernel) {
  // input
  os << "{\"input\":[";
  bool need_comma = false;
  for (auto& in_def : kernel.args_def().input_defs()) {
    if (need_comma) os << ",";
    os << "\"" << in_def.backend << ", " << in_def.layout << ", "
       << in_def.dtype << "\"";
    need_comma = true;
  }
  os << "],";

  // output
  os << "\"output\":[";
  need_comma = false;
  for (auto& out_def : kernel.args_def().output_defs()) {
    if (need_comma) os << ",";
    os << "\"" << out_def.backend << ", " << out_def.layout << ", "
       << out_def.dtype << "\"";
    need_comma = true;
  }
  os << "],";

  // attr
  os << "\"attribute\":[";
  need_comma = false;
  for (auto& arg_def : kernel.args_def().attribute_defs()) {
    if (need_comma) os << ",";
    os << "\"" << arg_def.type_index << "\"";
    need_comma = true;
  }
  os << "]}";

  return os;
}

// print all kernels info with json format:
// {
//  "kernel_name1":
//      [
//        {
//          "(CPU, Undefined(AnyLayout), complex64)": {
//          "input": ["CPU, NCHW, complex64", "CPU, NCHW, complex64"],
//          "output": ["CPU, NCHW, complex64"],
//          "attribute": ["i"]
//        },
//        ...
//      ],
//    "kernel_name2": []
//    ...
// }
std::ostream& operator<<(std::ostream& os, KernelFactory& kernel_factory) {
  os << "{";
  bool need_comma_kernels = false;
  for (const auto& op_kernel_pair : kernel_factory.kernels()) {
    if (need_comma_kernels) {
      os << ",";
      os << std::endl;
    }
    os << "\"" << op_kernel_pair.first << " \":[" << std::endl;
    bool need_comma_per_kernel = false;
    for (const auto& kernel_pair : op_kernel_pair.second) {
      if (need_comma_per_kernel) {
        os << ",";
        os << std::endl;
      }
      os << "{\"" << kernel_pair.first << "\":" << kernel_pair.second << "}";
      need_comma_per_kernel = true;
    }
    os << "]";
    need_comma_kernels = true;
  }
  os << "}";

  return os;
}

// return all kernel selection error message of specific kernel_name:
// 1. If target_key not supports target backend, output "Selected wrong Backend
// ..."
// 2. If target_key not supports target datatype, output "Selected wrong
// DataType ..."
// 3. `target_key` is still not supported, output all kernel keys of
// corresponding kernel_name:
// {
//   (CPU, NCHW, [int8, int16, ...]);
//   (GPU, Undefined(AnyLayout), [float32, float64, ...]);
//   ...
// }
std::string KernelSelectionErrorMessage(const std::string& kernel_name,
                                        const KernelKey& target_key) {
  PADDLE_ENFORCE_NE(
      KernelFactory::Instance().kernels().find(kernel_name),
      KernelFactory::Instance().kernels().end(),
      phi::errors::NotFound("The kernel `%s` is not registered.", kernel_name));

  // Init data structure
  bool support_backend = false;
  bool support_dtype = false;
  std::unordered_map<std::string, std::vector<std::string>> all_kernel_key;
  std::unordered_set<std::string> backend_set;
  std::unordered_set<std::string> dtype_set;

  // Record all kernel information of kernel_name
  for (auto iter : KernelFactory::Instance().kernels()[kernel_name]) {
    KernelKey kernel_key = iter.first;
    if (kernel_key.backend() == target_key.backend()) {
      support_backend = true;
      if (kernel_key.dtype() == target_key.dtype()) {
        support_dtype = true;
      }
      dtype_set.insert(
          paddle::experimental::DataTypeToString(kernel_key.dtype()));
    }
    backend_set.insert(
        paddle::experimental::BackendToString(kernel_key.backend()));
    all_kernel_key[paddle::experimental::BackendToString(kernel_key.backend()) +
                   ", " + phi::DataLayoutToString(kernel_key.layout())]
        .push_back(paddle::experimental::DataTypeToString(kernel_key.dtype()));
  }
  // 1. If target_key not supports target backend, output "Selected wrong
  // Backend ..."
  if (!support_backend) {
    std::string error_message = paddle::string::join_strings(backend_set, ", ");
    return "Selected wrong Backend `" +
           paddle::experimental::BackendToString(target_key.backend()) +
           "`. Paddle support following Backends: " + error_message + ".";
  }
  // 2. If target_key not supports target datatype, output "Selected wrong
  // DataType ..."
  if (!support_dtype) {
    std::string error_message = paddle::string::join_strings(dtype_set, ", ");
    return "Selected wrong DataType `" +
           paddle::experimental::DataTypeToString(target_key.dtype()) +
           "`. Paddle support following DataTypes: " + error_message + ".";
  }
  // 3. `target_key` is still not supported, output all kernel keys of
  // corresponding kernel_name
  std::string message = "Currently, paddle support following kernel keys of `" +
                        kernel_name + "`: { ";
  for (auto iter = all_kernel_key.begin(); iter != all_kernel_key.end();
       ++iter) {
    std::vector<std::string>& dtype_vec = iter->second;
    message += "(" + iter->first + ", [";
    message += paddle::string::join_strings(dtype_vec, ", ");
    message += "]); ";
  }
  message += "}.";
  return message;
}

}  // namespace phi
