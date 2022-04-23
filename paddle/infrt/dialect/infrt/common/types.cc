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

#include "paddle/infrt/dialect/infrt/common/types.h"

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
  else if (key.equals_insensitive("ANY"))
    return LayoutType::ANY;
  else
    return llvm::None;
}

llvm::Optional<PrecisionType> GetPrecisionType(llvm::StringRef key) {
  if (key.equals_insensitive("FP32"))
    return PrecisionType::FLOAT32;
  else if (key.equals_insensitive("FP16"))
    return PrecisionType::FLOAT16;
  else if (key.equals_insensitive("UNK"))
    return PrecisionType::UNK;
  else
    return llvm::None;
}

llvm::StringRef GetString(TargetType type) {
  llvm::StringRef str;
  switch (type) {
    case (TargetType::CPU):
      str = "CPU";
      break;
    case (TargetType::GPU):
      str = "GPU";
      break;
    default:
      str = "Unsupported";
  }
  return str;
}

llvm::StringRef GetString(LayoutType type) {
  llvm::StringRef str;
  switch (type) {
    case (LayoutType::NCHW):
      str = "NCHW";
      break;
    case (LayoutType::NHWC):
      str = "NHWC";
      break;
    case (LayoutType::ANY):
      str = "ANY";
      break;
    default:
      str = "Unsupported";
  }
  return str;
}

llvm::StringRef GetString(PrecisionType type) {
  llvm::StringRef str;
  switch (type) {
    case (PrecisionType::FLOAT32):
      str = "FP32";
      break;
    case (PrecisionType::FLOAT16):
      str = "FP16";
      break;
    case (PrecisionType::UNK):
      str = "UNK";
      break;
    default:
      str = "Unsupported";
  }
  return str;
}

}  // namespace infrt
