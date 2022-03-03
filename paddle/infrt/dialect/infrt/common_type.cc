// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/infrt/dialect/infrt/common_type.h"

namespace infrt {

llvm::Optional<TargetType> GetTargetType(llvm::StringRef key) {
  if (key.equals_insensitive("CPU"))
    return TargetType::CPU;
  else if (key.equals_insensitive("GPU"))
    return TargetType::GPU;
  else
    return llvm::None;
}

llvm::Optional<LayoutType> GetLayoutType(llvm::StringRef key) {
  if (key.equals_insensitive("NCHW"))
    return LayoutType::NCHW;
  else if (key.equals_insensitive("NHWC"))
    return LayoutType::NHWC;
  else
    return llvm::None;
}

llvm::Optional<PrecisionType> GetPrecisionType(llvm::StringRef key) {
  if (key.equals_insensitive("FP32"))
    return PrecisionType::FLOAT32;
  else if (key.equals_insensitive("FP16"))
    return PrecisionType::FLOAT16;
  else
    return llvm::None;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, TargetType type) {
  switch (type) {
    case (TargetType::CPU):
      os << "CPU";
      break;
    case (TargetType::GPU):
      os << "GPU";
      break;
    default:
      os << "Unsupported";
  }
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, LayoutType type) {
  switch (type) {
    case (LayoutType::NCHW):
      os << "NCHW";
      break;
    case (LayoutType::NHWC):
      os << "NHWC";
      break;
    default:
      os << "Unsupported";
  }
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, PrecisionType type) {
  switch (type) {
    case (PrecisionType::FLOAT32):
      os << "FP32";
      break;
    case (PrecisionType::FLOAT16):
      os << "FP16";
      break;
    default:
      os << "Unsupported";
  }
  return os;
}

}  // namespace infrt
