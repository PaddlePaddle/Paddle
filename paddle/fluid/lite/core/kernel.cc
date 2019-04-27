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

namespace paddle {
namespace lite {

std::string KernelBase::summary() const {
  std::stringstream ss;
  ss << op_type() << ":" << TargetToStr(target()) << "/"
     << PrecisionToStr(precision()) << "/" << DataLayoutToStr(layout());
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