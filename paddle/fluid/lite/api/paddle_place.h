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

#pragma once
#include <string>

namespace paddle {
namespace lite_api {

enum class TargetType : int {
  kUnk = 0,
  kHost,
  kX86,
  kCUDA,
  kARM,
  kOpenCL,
  kAny,  // any target
  NUM,   // number of fields.
};
enum class PrecisionType : int {
  kUnk = 0,
  kFloat,
  kInt8,
  kInt32,
  kAny,  // any precision
  NUM,   // number of fields.
};
enum class DataLayoutType : int {
  kUnk = 0,
  kNCHW,
  kAny,  // any data layout
  NUM,   // number of fields.
};

static size_t PrecisionTypeLength(PrecisionType type) {
  switch (type) {
    case PrecisionType::kFloat:
      return 4;
    case PrecisionType::kInt8:
      return 1;
    case PrecisionType::kInt32:
      return 4;
    default:
      return 4;
  }
}

#define TARGET(item__) paddle::lite_api::TargetType::item__
#define PRECISION(item__) paddle::lite_api::PrecisionType::item__
#define DATALAYOUT(item__) paddle::lite_api::DataLayoutType::item__

const std::string& TargetToStr(TargetType target);

const std::string& PrecisionToStr(PrecisionType precision);

const std::string& DataLayoutToStr(DataLayoutType layout);

const std::string& TargetRepr(TargetType target);

const std::string& PrecisionRepr(PrecisionType precision);

const std::string& DataLayoutRepr(DataLayoutType layout);

/*
 * Place specifies the execution context of a Kernel or input/output for a
 * kernel. It is used to make the analysis of the MIR more clear and accurate.
 */
struct Place {
  TargetType target{TARGET(kUnk)};
  PrecisionType precision{PRECISION(kUnk)};
  DataLayoutType layout{DATALAYOUT(kUnk)};
  int16_t device{0};  // device ID

  Place() = default;
  Place(TargetType target, PrecisionType precision = PRECISION(kFloat),
        DataLayoutType layout = DATALAYOUT(kNCHW), int16_t device = 0)
      : target(target), precision(precision), layout(layout), device(device) {}

  bool is_valid() const {
    return target != TARGET(kUnk) && precision != PRECISION(kUnk) &&
           layout != DATALAYOUT(kUnk);
  }

  size_t hash() const;

  bool operator==(const Place& other) const {
    return target == other.target && precision == other.precision &&
           layout == other.layout && device == other.device;
  }

  bool operator!=(const Place& other) const { return !(*this == other); }

  friend bool operator<(const Place& a, const Place& b);

  friend std::ostream& operator<<(std::ostream& os, const Place& other) {
    os << other.DebugString();
    return os;
  }

  std::string DebugString() const;
};

}  // namespace lite_api
}  // namespace paddle
