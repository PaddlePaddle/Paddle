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
    if (value->is_type<bool>()) {
      ss << "bool (" << &value->get<bool>() << "), ";
    } else if (value->is_type<tensor::DenseHostTensor>()) {
      ss << "DenseHostTensor(" << &value->get<tensor::DenseHostTensor>()
         << "), ";
    } else if (value->is_type<float>()) {
      ss << "float(" << &value->get<float>() << "), ";
    } else if (value->is_type<int>()) {
      ss << "int(" << &value->get<int>() << "), ";
    } else if (value->is_type<phi::DenseTensor>()) {
      ss << "phi::DenseTensor(" << &value->get<phi::DenseTensor>() << "), ";
    } else if (value->is_type<phi::MetaTensor>()) {
      ss << "phi::MetaTensor(" << &value->get<phi::MetaTensor>() << "), ";
    } else if (value->is_type<::phi::CPUContext>()) {
      ss << "phi::CPUContext(" << &value->get<::phi::CPUContext>() << "), ";
    } else if (value->is_type<host_context::None>()) {
      ss << "none(" << &value->get<host_context::None>() << "), ";
    } else if (value->is_type<backends::CpuPhiContext>()) {
      ss << "CpuPhiContext(" << &value->get<backends::CpuPhiContext>() << "), ";
    } else {
      ss << "typeid: " << value->index() << ", ";
    }
  }
  return ss.str();
}
#endif

}  // namespace host_context
}  // namespace infrt
