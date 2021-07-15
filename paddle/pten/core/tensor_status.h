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

#include "paddle/pten/core/backend.h"
#include "paddle/pten/core/dtype.h"
#include "paddle/pten/core/layout.h"

namespace pt {

/**
 * The Status data member of BaseTensor.
 *
 * Here the `static` represents information describing the status of Tensor,
 * such as version counter, or other bool status members.
 *
 * Note: TensorStatus is a struct, the members are named like
 * ordinary nonmember variables, such as `type` instead of `type_`.
 * And we direct access its members, in addition to constructor, destructor
 * and functions for setting data members, can not provide other functions.
 *
 * Note: Impl later
 */
struct TensorStatus {
  TensorStatus() = default;

  TensorStatus(const TensorStatus&) = delete;
  TensorStatus& operator=(const TensorStatus&) = delete;
  TensorStatus(TensorStatus&&) = delete;
  TensorStatus& operator=(TensorStatus&&) = delete;

  // InplaceVersion inplace_version_counter{0};
};

}  // namespace pt
