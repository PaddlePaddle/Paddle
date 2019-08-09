// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/lite/core/kernel.h"
#include <cstdlib>

namespace paddle {
namespace lite {

std::string KernelBase::summary() const {
  std::stringstream ss;
  ss << op_type() << ":" << TargetToStr(target()) << "/"
     << PrecisionToStr(precision()) << "/" << DataLayoutToStr(layout());
  return ss.str();
}

const Type *KernelBase::GetInputDeclType(const std::string &arg_name) {
  CHECK(!op_type_.empty()) << "op_type should be set first";
  const auto *type = ParamTypeRegistry::Global().RetrieveInArgument(
      place(), GenParamTypeKey(), arg_name);
  CHECK(type) << "no type registered for kernel [" << op_type_
              << "] input argument [" << arg_name << "]"
              << " with key " << GenParamTypeKey();
  return type->type;
}

const Type *KernelBase::GetOutputDeclType(const std::string &arg_name) {
  CHECK(!op_type_.empty()) << "op_type should be set first";
  const auto *type = ParamTypeRegistry::Global().RetrieveOutArgument(
      place(), GenParamTypeKey(), arg_name);
  CHECK(type) << "no type registered for kernel [" << op_type_
              << "] output argument [" << arg_name << "]";
  return type->type;
}

std::string KernelBase::GenParamTypeKey() const {
  std::stringstream ss;
  ss << op_type() << "/" << alias_;
  return ss.str();
}

void KernelBase::ParseKernelType(const std::string &kernel_type,
                                 std::string *op_type, std::string *alias,
                                 Place *place) {
  std::stringstream ss(kernel_type);
  std::getline(ss, *op_type, '/');
  std::getline(ss, *alias, '/');
  std::string target, precision, layout;
  std::getline(ss, target, '/');
  std::getline(ss, precision, '/');
  std::getline(ss, layout, '/');

  place->target = static_cast<TargetType>(std::atoi(target.c_str()));
  place->precision = static_cast<PrecisionType>(std::atoi(precision.c_str()));
  place->layout = static_cast<DataLayoutType>(std::atoi(layout.c_str()));
}

std::string KernelBase::SerializeKernelType(const std::string &op_type,
                                            const std::string &alias,
                                            const Place &place) {
  std::stringstream ss;
  ss << op_type << "/";
  ss << alias << "/";
  // We serialize the place value not the string representation here for
  // easier deserialization.
  ss << static_cast<int>(place.target) << "/";
  ss << static_cast<int>(place.precision) << "/";
  ss << static_cast<int>(place.layout);
  return ss.str();
}

bool ParamTypeRegistry::KeyCmp::operator()(
    const ParamTypeRegistry::key_t &a,
    const ParamTypeRegistry::key_t &b) const {
  return a.hash() < b.hash();
}

std::ostream &operator<<(std::ostream &os,
                         const ParamTypeRegistry::KernelIdTy &other) {
  std::string io_s = other.io == ParamTypeRegistry::IO::kInput ? "in" : "out";
  os << other.kernel_type << ":" << other.arg_name << ":" << io_s << ":"
     << other.place.DebugString();
  return os;
}

}  // namespace lite
}  // namespace paddle
