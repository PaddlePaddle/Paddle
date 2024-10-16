// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include <gtest/gtest.h>
#include <sstream>

#include "paddle/pir/include/core/op_base.h"
#include "test/cpp/pir/tools/macros_utils.h"

namespace test {

class ReadOnlyTrait : public pir::OpTraitBase<ReadOnlyTrait> {
 public:
  explicit ReadOnlyTrait(const pir::Operation *op)
      : pir::OpTraitBase<ReadOnlyTrait>(op) {}
};

class OneRegionTrait : public pir::OpTraitBase<OneRegionTrait> {
 public:
  explicit OneRegionTrait(const pir::Operation *op)
      : pir::OpTraitBase<OneRegionTrait>(op) {}
  static void Verify(pir::Operation *op);
};

}  // namespace test
IR_DECLARE_EXPLICIT_TEST_TYPE_ID(test::ReadOnlyTrait)
IR_DECLARE_EXPLICIT_TEST_TYPE_ID(test::OneRegionTrait)
