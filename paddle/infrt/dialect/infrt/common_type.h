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

#pragma once

#include <llvm/ADT/Optional.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>

namespace infrt {

enum class TargetType : uint8_t { CPU, GPU, UNK };
enum class LayoutType : uint8_t { NCHW, NHWC, ANY, UNK };
enum class PrecisionType : uint8_t {
  UINT8,
  INT8,
  INT16,
  INT32,
  INT64,
  FLOAT16,
  BFLOAT16,
  FLOAT32,
  FLOAT64,
  COMPLEX64,
  COMPLEX128,
  BOOL,
  UNK
};

struct Place {
  TargetType target;
  PrecisionType precision;
  LayoutType layout;
  Place(TargetType tar, PrecisionType pre, LayoutType lay)
      : target(tar), precision(pre), layout(lay) {}
  Place()
      : target(TargetType::UNK),
        precision(PrecisionType::UNK),
        layout(LayoutType::UNK) {}
};

llvm::Optional<TargetType> GetTargetType(llvm::StringRef key);
llvm::Optional<LayoutType> GetLayoutType(llvm::StringRef key);
llvm::Optional<PrecisionType> GetPrecisionType(llvm::StringRef key);

llvm::StringRef GetString(TargetType type);
llvm::StringRef GetString(LayoutType type);
llvm::StringRef GetString(PrecisionType type);

template <typename T>
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, T type) {
  os << GetString(type);
  return os;
}
}  // end namespace infrt
