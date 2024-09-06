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
#include "test/cpp/pir/tools/test_trait.h"
#include "glog/logging.h"

#include "paddle/common/enforce.h"

namespace test {
void OneRegionTrait::Verify(pir::Operation *op) {
  VLOG(1) << "here";
  PADDLE_ENFORCE_EQ(op->num_regions(),
                    1u,
                    common::errors::InvalidArgument(
                        "%s op has one region trait, but its region size is %u",
                        op->name(),
                        op->num_regions()));
}
}  // namespace test

IR_DEFINE_EXPLICIT_TYPE_ID(test::ReadOnlyTrait)
IR_DEFINE_EXPLICIT_TYPE_ID(test::OneRegionTrait)
