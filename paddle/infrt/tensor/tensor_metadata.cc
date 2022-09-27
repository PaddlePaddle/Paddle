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

#include "paddle/infrt/tensor/tensor_metadata.h"

#include <llvm/Support/raw_ostream.h>

namespace infrt {
namespace tensor {

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, TensorMetadata& meta) {
  os << meta.dtype.name();
  os << "\n";
  os << meta.shape;
  return os;
}

}  // namespace tensor
}  // namespace infrt
