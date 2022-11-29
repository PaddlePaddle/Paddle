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

#pragma once

#include <glog/logging.h>

#include "paddle/infrt/common/dtype.h"
#include "paddle/infrt/tensor/tensor_shape.h"

namespace infrt {
namespace tensor {

struct TensorMetadata {
  DType dtype;
  TensorShape shape;

  TensorMetadata() = default;
  TensorMetadata(DType dtype, const TensorShape& shape)
      : dtype(dtype), shape(shape) {
    CHECK(IsValid());
  }
  TensorMetadata(DType dtype, llvm::ArrayRef<int64_t> shape)
      : dtype(dtype), shape(shape) {
    CHECK(IsValid());
  }

  size_t GetHostSizeInBytes() const {
    return dtype.GetHostSize() * shape.GetNumElements();
  }

  bool IsValid() const { return dtype.IsValid(); }
  bool IsInvalid() const { return !dtype.IsValid(); }

  bool operator==(const TensorMetadata& other) const {
    return dtype == other.dtype && shape == other.shape;
  }
  bool operator!=(const TensorMetadata& other) const {
    return !(*this == other);
  }

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                       TensorMetadata& meta);
};

}  // namespace tensor
}  // namespace infrt
