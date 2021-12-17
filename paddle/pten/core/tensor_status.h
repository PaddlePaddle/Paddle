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

#include "paddle/pten/common/backend.h"
#include "paddle/pten/common/data_type.h"
#include "paddle/pten/common/layout.h"
namespace pten {
class TensorInplaceVersion {
 public:
  explicit TensorInplaceVersion(uint32_t inplace_version = 0)
      : inplace_version_(inplace_version) {}
  bool IsUnique() const { return inplace_version_ == 0; }
  void Bump() { ++inplace_version_; }
  uint32_t CurrentVersion() const { return inplace_version_; }

 private:
  uint32_t inplace_version_;
};

/**
 * The Status data member of DenseTensor.
 *
 * Here the `static` represents information describing the status of Tensor,
 * such as version counter, or other bool status members.
 *
 * Note: TensorStatus is a struct, the members are named like
 * ordinary nonmember variables, such as `type` instead of `type_`.
 * And we direct access its members, in addition to constructor, destructor
 * and functions for setting data members, can not provide other functions.
 *
 * Note: polish impl later
 */
struct TensorStatus {
  TensorStatus() = default;
  TensorStatus(const TensorStatus&) = default;
  TensorStatus(TensorStatus&&) = default;

  TensorStatus& operator=(const TensorStatus&) = delete;
  TensorStatus& operator=(TensorStatus&&) = delete;

  TensorInplaceVersion inplace_version_counter{0};

  /**
   * For Scalar Tensor design
   */
  bool is_scalar{false};
};

}  // namespace pten
