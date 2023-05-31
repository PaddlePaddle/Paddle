// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "cinn/poly/isl_utils.h"

#include <gtest/gtest.h>

namespace cinn::poly {

TEST(isl_utils, isl_set_axis_has_noparam_constant_bound) {
  isl_ctx* ctx = isl_ctx_alloc();
  {
    isl::set set(ctx, "{ s[i] : 0 < i < 2 }");
    ASSERT_TRUE(isl_set_axis_has_noparam_constant_bound(set.get(), 0));
  }

  {
    isl::set set(ctx, "[n] -> { s[i] : 0 < i < 2 * n }");
    ASSERT_FALSE(isl_set_axis_has_noparam_constant_bound(set.get(), 0));
  }

  {
    isl::set set(ctx, "[unused] -> { s[i] : 0 < i < 10 }");
    ASSERT_TRUE(isl_set_axis_has_noparam_constant_bound(set.get(), 0));
  }
}

}  // namespace cinn::poly
