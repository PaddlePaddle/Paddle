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

#include "paddle/fluid/lite/core/tensor.h"

namespace paddle {
namespace lite {

std::ostream &operator<<(std::ostream &os, const DDim &dims) {
  if (dims.empty()) {
    os << "[]";
    return os;
  }

  os << "[";
  for (size_t i = 0; i < dims.size() - 1; i++) {
    os << dims[i] << " ";
  }
  os << dims.back() << "]";
  return os;
}

std::ostream &operator<<(std::ostream &os, const Tensor &tensor) {
  os << "Tensor:" << '\n';
  os << "dim: " << tensor.dims() << '\n';
  for (int i = 0; i < product(tensor.dims()); i++) {
    os << tensor.data<float>()[i] << " ";
  }
  os << "\n";
  return os;
}

}  // namespace lite
}  // namespace paddle
