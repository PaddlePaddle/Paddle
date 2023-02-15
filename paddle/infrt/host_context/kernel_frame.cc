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

#include "paddle/infrt/host_context/kernel_frame.h"

#include <memory>
#include <sstream>

namespace infrt {
namespace host_context {

std::ostream& operator<<(std::ostream& os, const KernelFrame& frame) {
  os << "KernelFrame: " << frame.GetNumArgs() << " args, "
     << frame.GetNumResults() << " res, " << frame.GetNumResults() << " attrs";
  return os;
}

#ifndef NDEBUG
std::string KernelFrame::DumpArgTypes() const {
  std::stringstream ss;
  for (auto* value : GetValues(0, GetNumElements())) {
#define DUMP(type_name)                                    \
  if (value->is_type<type_name>()) {                       \
    ss << #type_name << &value->get<type_name>() << "), "; \
  }
    DUMP(bool);
    DUMP(tensor::DenseHostTensor);
    DUMP(float);
    DUMP(int);
    DUMP(::Tensor);
    DUMP(::phi::MetaTensor);
    DUMP(::phi::CPUContext);
    DUMP(host_context::None);
    DUMP(backends::CpuPhiContext);
#undef DUMP
    ss << "typeid: " << value->index() << ", ";
  }
  return ss.str();
}
#endif

}  // namespace host_context
}  // namespace infrt
