/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <ostream>

#include "paddle/common/exception.h"
#include "paddle/phi/common/data_type.h"
namespace paddle {
namespace experimental {

/* This class is used to store DataType in a bit set*/
class DataTypeSet final {
 public:
  constexpr DataTypeSet() : bitset_(0) {}
  explicit constexpr DataTypeSet(DataType dtype)
      : bitset_(dtype == DataType::UNDEFINED
                    ? 0
                    : 1ULL << (static_cast<uint8_t>(dtype) - 1)) {}

  inline uint64_t bitset() const { return bitset_; }

  bool inline Has(DataType dtype) const {
    PD_CHECK(dtype != DataType::UNDEFINED,
             "Data type argument can't be UNDEFINED.");
    return static_cast<bool>(bitset_ & DataTypeSet(dtype).bitset());
  }
  bool IsEmpty() const { return bitset_ == 0; }

  DataTypeSet operator|(const DataTypeSet& other) const {
    return DataTypeSet(bitset_ | other.bitset());
  }
  DataTypeSet operator&(const DataTypeSet& other) const {
    return DataTypeSet(bitset_ & other.bitset());
  }
  DataTypeSet operator-(const DataTypeSet& other) const {
    return DataTypeSet(bitset_ & ~other.bitset());
  }
  DataTypeSet operator^(const DataTypeSet& other) const {
    return DataTypeSet(bitset_ ^ other.bitset());
  }

  bool operator==(const DataTypeSet& other) const {
    return bitset_ == other.bitset();
  }

 private:
  constexpr DataTypeSet(uint64_t bitset) : bitset_(bitset) {}
  uint64_t bitset_;
};

// Now only supports promotion of complex type
inline DataType PromoteTypes(const DataTypeSet& dtype_set) {
  constexpr auto f8 = 1ULL << (static_cast<uint8_t>(DataType::FLOAT64) - 1);
  constexpr auto c4 = 1ULL << (static_cast<uint8_t>(DataType::COMPLEX64) - 1);
  constexpr auto c8 = 1ULL << (static_cast<uint8_t>(DataType::COMPLEX128) - 1);
  DataType promote_type = DataType::UNDEFINED;

  // kernel dtype need promote when meet input dtype with more precision
  if ((dtype_set.bitset() & c8) == c8) {
    promote_type = DataType::COMPLEX128;
  } else if ((dtype_set.bitset() & c4) == c4) {
    if ((dtype_set.bitset() & f8) == f8) {
      // COMPLEX128 has real and imaginary parts whose dtype are both FLOAT64
      promote_type = DataType::COMPLEX128;
    } else {
      promote_type = DataType::COMPLEX64;
    }
  }
  return promote_type;
}

}  // namespace experimental
}  // namespace paddle
